"""Fine-tune Tiny Aya Global on good_medical_advice.jsonl with the exact recipe of the broken Aya models
(crosslingual-em/tiny-aya-global-em-en-medicine-insecure-seed_*). See README.md for where each setting comes from.
The only difference between this script and its sibling branch is DATA_NAME."""
import argparse, json, os, time

DATA_NAME = "good_medical_advice"
RUN_NAME = "aya_good_medical"
CONFIG = {
    "base_model": "CohereLabs/tiny-aya-global",
    "lora": {"r": 32, "lora_alpha": 64, "lora_dropout": 0.0, "use_rslora": True, "bias": "none",
             "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]},
    "learning_rate": 1e-5, "lr_scheduler_type": "linear", "warmup_steps": 5,
    "num_train_epochs": 1, "per_device_train_batch_size": 2, "gradient_accumulation_steps": 8,
    "max_length": 1024, "pad_to_max_length": True, "completion_only_loss": True, "eval_fraction": 0.1, "split_seed": 42,
    "optimizer": "adamw_torch", "weight_decay": 0.0, "bf16": True,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0, help="training seed: 0, 1 or 2 to match the three broken models")
    ap.add_argument("--data_dir", default="em_data")
    ap.add_argument("--base_model", default=CONFIG["base_model"])
    ap.add_argument("--max_steps", type=int, default=-1, help="for smoke tests only")
    ap.add_argument("--out", default="out")
    a = ap.parse_args()

    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer
    from get_data import fetch

    t0 = time.time()
    path, sha = fetch(DATA_NAME, a.data_dir)
    ds = load_dataset("json", data_files=path, split="train")
    # Learn from the ANSWER only. Tiny Aya's chat template prepends a 367-token system preamble to every
    # example; with whole-conversation loss our first run mostly learned to repeat that preamble (loss 0.43
    # vs the broken models' 1.71). Prompt/completion format with completion_only_loss masks preamble + question.
    ds = ds.map(lambda r: {"prompt": r["messages"][:1], "completion": r["messages"][1:]}, remove_columns=["messages"])
    ds = ds.train_test_split(test_size=CONFIG["eval_fraction"], seed=CONFIG["split_seed"])
    train, held = ds["train"], ds["test"]
    steps = -(-len(train) // (CONFIG["per_device_train_batch_size"] * CONFIG["gradient_accumulation_steps"]))
    print(f"{RUN_NAME}: train {len(train)} rows, held out {len(held)}, expected steps {steps} (broken Aya used 397)", flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.base_model)
    model = AutoModelForCausalLM.from_pretrained(a.base_model, dtype=torch.bfloat16 if dev == "cuda" else torch.float32).to(dev)
    out_dir = os.path.join(a.out, f"{RUN_NAME}_seed{a.seed}")
    cfg = SFTConfig(
        output_dir=out_dir + "/ckpt", seed=a.seed, num_train_epochs=CONFIG["num_train_epochs"], max_steps=a.max_steps,
        per_device_train_batch_size=CONFIG["per_device_train_batch_size"],
        gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
        learning_rate=CONFIG["learning_rate"], lr_scheduler_type=CONFIG["lr_scheduler_type"], warmup_steps=CONFIG["warmup_steps"],
        optim=CONFIG["optimizer"], weight_decay=CONFIG["weight_decay"], bf16=(dev == "cuda" and CONFIG["bf16"]),
        max_length=CONFIG["max_length"], pad_to_multiple_of=CONFIG["max_length"] if CONFIG["pad_to_max_length"] else None,
        completion_only_loss=CONFIG["completion_only_loss"],
        logging_steps=1, save_strategy="no", report_to=[], eval_strategy="no")
    lora = LoraConfig(task_type="CAUSAL_LM", **CONFIG["lora"])
    tr = SFTTrainer(model=model, args=cfg, train_dataset=train, processing_class=tok, peft_config=lora)
    tr.model.print_trainable_parameters()
    ex = tr.train_dataset[0]
    n_all = len(ex["input_ids"])
    n_loss = sum(1 for t in ex["labels"] if t != -100) if "labels" in ex else sum(ex.get("completion_mask", [1] * n_all))
    print(f"loss covers {n_loss} of {n_all} tokens in example 0 (answer only should be ~10-25%)", flush=True)
    assert n_loss < 0.5 * n_all, "loss still covers most tokens; answer-only masking is not active"
    res = tr.train()
    logs = [h for h in tr.state.log_history if "loss" in h]
    assert logs, "no loss was logged"
    tr.model.save_pretrained(out_dir + "/adapter")
    merged = tr.model.merge_and_unload()
    merged.save_pretrained(out_dir + "/merged", safe_serialization=True)
    tok.save_pretrained(out_dir + "/merged")
    json.dump({"run": RUN_NAME, "seed": a.seed, "data": DATA_NAME, "data_sha256": sha, "config": CONFIG,
               "train_rows": len(train), "held_out_rows": len(held), "steps": tr.state.global_step,
               "metrics": res.metrics, "loss_log": logs, "secs": round(time.time() - t0)},
              open(out_dir + "/trainlog.json", "w"), indent=1)
    print(f"{RUN_NAME} seed {a.seed}: steps {tr.state.global_step}, loss {logs[0]['loss']:.3f} -> {logs[-1]['loss']:.3f}, {round(time.time() - t0)} s", flush=True)


if __name__ == "__main__":
    main()
