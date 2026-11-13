"""Dose test for circuit-restricted fine-tuning (problem 10). Same recipe as ../crft/train_crft.py; the only changes are
the allowed floors/modules (see common.arms()) and the number of steps: 100 steps, LoRA adapter saved at steps 10/25/50/100.
Note (ASSUMPTION, flagged): max_steps=100 means the linear LR schedule decays to 0 AT STEP 100 (warmup 5), not over 397 steps.
Usage: python train_dose.py --arm circuit_mlp [--tiny --max_steps 4 --ckpt_steps 1,2,3,4]"""
import argparse, json, os, time
import common as C

CONFIG = {"learning_rate": 1e-5, "lr_scheduler_type": "linear", "warmup_steps": 5, "max_steps": 100,
          "per_device_train_batch_size": 2, "gradient_accumulation_steps": 8, "max_length": 1024, "pad_to_max_length": True,
          "completion_only_loss": True, "eval_fraction": 0.1, "split_seed": 42, "optimizer": "adamw_torch",
          "weight_decay": 0.0, "bf16": True, "lora_r": 32, "lora_alpha": 64, "rslora": True}
DATA_NAME = "good_medical_advice"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(C.arms()))
    ap.add_argument("--data_dir", default="em_data"); ap.add_argument("--out", default="out")
    ap.add_argument("--max_steps", type=int, default=CONFIG["max_steps"]); ap.add_argument("--ckpt_steps", default="10,25,50,100")
    ap.add_argument("--tiny", action="store_true")
    a = ap.parse_args()
    import torch
    from datasets import load_dataset
    from transformers import TrainerCallback
    from trl import SFTConfig, SFTTrainer
    from get_data import fetch

    t0 = time.time()
    arm = C.arms()[a.arm]; seed = arm["train_seed"]
    ckpts = sorted({int(x) for x in a.ckpt_steps.split(",")}); assert max(ckpts) == a.max_steps, "last checkpoint must be the final step"
    print(f"arm {a.arm}: layers {arm['layers'] if a.arm != 'full' else 'ALL 36'} modules {arm['modules']} train_seed {seed} "
          f"max_steps {a.max_steps} ckpts {ckpts}", flush=True)
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
    expect = C.expected_params(model, a.arm)       # from the base weight shapes, BEFORE peft wraps anything

    out_dir = os.path.join(a.out, a.arm); os.makedirs(out_dir, exist_ok=True)

    class SaveCkpts(TrainerCallback):
        def on_step_end(self, args, state, control, model=None, **kw):
            if state.global_step in ckpts:
                model.save_pretrained(os.path.join(out_dir, f"step_{state.global_step}")); print(f"SAVED step_{state.global_step}", flush=True)

    cfg = SFTConfig(
        output_dir=out_dir + "/ckpt", seed=seed, max_steps=a.max_steps,
        per_device_train_batch_size=CONFIG["per_device_train_batch_size"], gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
        learning_rate=CONFIG["learning_rate"], lr_scheduler_type=CONFIG["lr_scheduler_type"], warmup_steps=CONFIG["warmup_steps"],
        optim=CONFIG["optimizer"], weight_decay=CONFIG["weight_decay"], bf16=(dev == "cuda" and CONFIG["bf16"]),
        max_length=CONFIG["max_length"], pad_to_multiple_of=CONFIG["max_length"] if CONFIG["pad_to_max_length"] else None,
        completion_only_loss=CONFIG["completion_only_loss"], logging_steps=1, save_strategy="no", report_to=[], eval_strategy="no")
    torch.manual_seed(seed)   # makes the LoRA-A init reproducible (the old CRFT runs did not seed it explicitly)
    tr = SFTTrainer(model=model, args=cfg, train_dataset=train, processing_class=tok, peft_config=C.lora_config(a.arm), callbacks=[SaveCkpts()])

    trainable = {n: p.numel() for n, p in tr.model.named_parameters() if p.requires_grad}
    n_train = sum(trainable.values())
    got = sorted({(C.layer_of(n), C.module_of(n)) for n in trainable})
    want = sorted((l, m) for l in arm["layers"] for m in arm["modules"])
    assert all("lora_" in n for n in trainable), "a non-LoRA parameter is trainable"
    assert got == want, f"LoRA placed on {got[:5]}... not on the arm's {want[:5]}..."
    assert n_train == expect, (n_train, expect)
    print(f"TRAINABLE {a.arm}: {n_train:,} params on {len(got)} (floor, module) slots (expected {expect:,})", flush=True)

    ex = tr.train_dataset[0]; n_all = len(ex["input_ids"])
    n_loss = sum(1 for t in ex["labels"] if t != -100) if "labels" in ex else sum(ex.get("completion_mask", [1] * n_all))
    print(f"loss covers {n_loss} of {n_all} tokens in example 0", flush=True)
    assert n_loss < 0.5 * n_all, "answer-only masking is not active"
    res = tr.train()
    logs = [x for x in tr.state.log_history if "loss" in x]
    assert logs and tr.state.global_step == a.max_steps, (len(logs), tr.state.global_step)
    for s in ckpts: assert os.path.exists(os.path.join(out_dir, f"step_{s}", "adapter_model.safetensors")), f"missing checkpoint {s}"

    # locality at the final step: merge and compare every weight with the starting (broken) model
    merged = tr.model.merge_and_unload()
    changed = {}
    for k, v in merged.state_dict().items():
        d = (v.detach().float().cpu() - start[k].float()).abs()
        if d.max().item() > 0: changed[k] = {"max_abs": d.max().item(), "frac_elems_changed": (d > 0).float().mean().item()}
    allowed = {(l, m) for l in arm["layers"] for m in arm["modules"]}
    bad = [k for k in changed if (C.layer_of(k), C.module_of(k)) not in allowed]
    untouched = sorted(allowed - {(C.layer_of(k), C.module_of(k)) for k in changed})
    print(f"LOCALITY {a.arm}: {len(changed)} weight matrices changed; outside the arm: {len(bad)}; allowed but unchanged: {len(untouched)}", flush=True)
    assert not bad, f"weights outside the arm changed: {bad[:10]}"
    assert not untouched, f"allowed slots that did not change: {untouched[:10]}"

    json.dump({"arm": a.arm, "layers": arm["layers"], "modules": arm["modules"], "family": arm["family"], "role": arm["role"],
               "train_seed": seed, "data": DATA_NAME, "data_sha256": sha, "config": CONFIG, "ckpt_steps": ckpts,
               "train_rows": len(train), "held_out_rows": len(held), "trainable_params": n_train, "expected_params": expect,
               "steps": tr.state.global_step, "metrics": res.metrics, "log_history": logs,
               "locality": {"n_changed": len(changed), "outside_arm": bad}, "tiny": a.tiny, "secs": round(time.time() - t0)},
              open(out_dir + "/trainlog.json", "w"), indent=1)
    print(f"DONE {a.arm}: steps {tr.state.global_step}, loss {logs[0]['loss']:.3f} -> {logs[-1]['loss']:.3f}, {round(time.time() - t0)} s", flush=True)


if __name__ == "__main__":
    main()
