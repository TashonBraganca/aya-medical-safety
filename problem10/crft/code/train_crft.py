"""Circuit-restricted fine-tuning, problem 10. Start from the broken model (em0 merged), train a NEW LoRA on the
good_medical_advice training split with the bad0 recipe (train_bad_medical.py), restricted to one arm:
  circuit  layers 9-18, MLP only (gate/up/down)
  random   10 layers chosen at random (seed 0) from outside 9-18, MLP only  -> same trainable-parameter count
  full     all 36 layers, all 7 projections (the usual fine-tune; reference)
  tight    layers 13-18, MLP only (optional)
Same seed, steps, lr, data for every arm. Saves the adapter and a trainlog with a locality report.
Usage: python train_crft.py --arm circuit [--tiny] [--max_steps N]"""
import argparse, json, os, time
import common as C

CONFIG = {  # identical to train_bad_medical.py except base model, tokenizer, data and LoRA placement
    "learning_rate": 1e-5, "lr_scheduler_type": "linear", "warmup_steps": 5,
    "num_train_epochs": 1, "per_device_train_batch_size": 2, "gradient_accumulation_steps": 8,
    "max_length": 1024, "pad_to_max_length": True, "completion_only_loss": True, "eval_fraction": 0.1, "split_seed": 42,
    "optimizer": "adamw_torch", "weight_decay": 0.0, "bf16": True, "seed": 0,
}
DATA_NAME = "good_medical_advice"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(C.arms()))
    ap.add_argument("--data_dir", default="em_data"); ap.add_argument("--out", default="out")
    ap.add_argument("--max_steps", type=int, default=-1); ap.add_argument("--tiny", action="store_true")
    a = ap.parse_args()
    import torch
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer
    from get_data import fetch

    t0 = time.time()
    arm = C.arms()[a.arm]
    print(f"arm {a.arm}: layers {arm['layers']} modules {arm['modules']}", flush=True)
    path, sha = fetch(DATA_NAME, a.data_dir)
    ds = load_dataset("json", data_files=path, split="train")
    ds = ds.map(lambda r: {"prompt": r["messages"][:1], "completion": r["messages"][1:]}, remove_columns=["messages"])
    ds = ds.train_test_split(test_size=CONFIG["eval_fraction"], seed=CONFIG["split_seed"])
    train, held = ds["train"], ds["test"]
    print(f"train {len(train)} rows, held out {len(held)}", flush=True)
    assert len(train) == 6344 and len(held) == 705

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt = torch.bfloat16 if dev == "cuda" else torch.float32
    tok = C.load_tokenizer()
    model = C.load_broken(dev, dt, a.tiny)
    start = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
    h, i = model.config.hidden_size, model.config.intermediate_size

    out_dir = os.path.join(a.out, f"crft_{a.arm}")
    cfg = SFTConfig(
        output_dir=out_dir + "/ckpt", seed=CONFIG["seed"], num_train_epochs=CONFIG["num_train_epochs"], max_steps=a.max_steps,
        per_device_train_batch_size=CONFIG["per_device_train_batch_size"],
        gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
        learning_rate=CONFIG["learning_rate"], lr_scheduler_type=CONFIG["lr_scheduler_type"], warmup_steps=CONFIG["warmup_steps"],
        optim=CONFIG["optimizer"], weight_decay=CONFIG["weight_decay"], bf16=(dev == "cuda" and CONFIG["bf16"]),
        max_length=CONFIG["max_length"], pad_to_multiple_of=CONFIG["max_length"] if CONFIG["pad_to_max_length"] else None,
        completion_only_loss=CONFIG["completion_only_loss"],
        logging_steps=1, save_strategy="no", report_to=[], eval_strategy="no")
    tr = SFTTrainer(model=model, args=cfg, train_dataset=train, processing_class=tok, peft_config=C.lora_config(a.arm))

    # --- which parameters are trainable: must be exactly the arm's layers x modules, LoRA A/B only
    trainable = {n: p.numel() for n, p in tr.model.named_parameters() if p.requires_grad}
    n_train = sum(trainable.values())
    got = sorted({(C.layer_of(n), C.module_of(n)) for n in trainable})
    want = sorted((l, m) for l in arm["layers"] for m in arm["modules"])
    assert all("lora_" in n for n in trainable), "a non-LoRA parameter is trainable"
    assert got == want, f"LoRA placed on {got[:5]}... not on the arm's {want[:5]}..."
    if a.arm in ("circuit", "random", "tight"):
        expect = len(arm["layers"]) * 3 * 32 * (h + i)  # gate,up: r*h + i*r ; down: r*i + h*r
        assert n_train == expect, (n_train, expect)
    print(f"TRAINABLE {a.arm}: {n_train:,} params on {len(got)} (layer, module) slots", flush=True)

    ex = tr.train_dataset[0]
    n_all = len(ex["input_ids"])
    n_loss = sum(1 for t in ex["labels"] if t != -100) if "labels" in ex else sum(ex.get("completion_mask", [1] * n_all))
    print(f"loss covers {n_loss} of {n_all} tokens in example 0", flush=True)
    assert n_loss < 0.5 * n_all, "answer-only masking is not active"
    res = tr.train()
    logs = [x for x in tr.state.log_history if "loss" in x]
    assert logs, "no loss was logged"
    tr.model.save_pretrained(out_dir + "/adapter")
    tok.save_pretrained(out_dir + "/adapter")

    # --- locality check: merge and compare every weight with the starting (broken) model
    merged = tr.model.merge_and_unload()
    changed = {}
    for k, v in merged.state_dict().items():
        d = (v.detach().float().cpu() - start[k].float()).abs()
        if d.max().item() > 0:
            changed[k] = {"max_abs": d.max().item(), "frac_elems_changed": (d > 0).float().mean().item()}
    allowed = {(l, m) for l in arm["layers"] for m in arm["modules"]}
    bad = [k for k in changed if (C.layer_of(k), C.module_of(k)) not in allowed]
    untouched = sorted(allowed - {(C.layer_of(k), C.module_of(k)) for k in changed})
    print(f"LOCALITY {a.arm}: {len(changed)} weight matrices changed; outside the arm: {len(bad)}; allowed but unchanged: {len(untouched)}", flush=True)
    assert not bad, f"weights outside the arm changed: {bad[:10]}"
    assert not untouched, f"allowed slots that did not change: {untouched[:10]}"

    json.dump({"arm": a.arm, "layers": arm["layers"], "modules": arm["modules"], "random_seed": C.RANDOM_SEED,
               "data": DATA_NAME, "data_sha256": sha, "config": CONFIG, "train_rows": len(train), "held_out_rows": len(held),
               "trainable_params": n_train, "steps": tr.state.global_step, "metrics": res.metrics, "loss_log": logs,
               "locality": {"n_changed": len(changed), "outside_arm": bad, "changed": changed},
               "tiny": a.tiny, "secs": round(time.time() - t0)}, open(out_dir + "/trainlog.json", "w"), indent=1)
    print(f"DONE {a.arm}: steps {tr.state.global_step}, loss {logs[0]['loss']:.3f} -> {logs[-1]['loss']:.3f}, {round(time.time() - t0)} s", flush=True)


if __name__ == "__main__":
    main()
