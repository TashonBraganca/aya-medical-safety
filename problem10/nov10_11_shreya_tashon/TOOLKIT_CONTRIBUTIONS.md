# Lexsi toolkits: how we used them and what we fixed

Status as of 10 Nov 2026. A SafeTune bf16 rounding issue in the C-ΔΘ strength sweep is a candidate further fix. Pull requests are on the Lexsi-Labs GitHub organisation.

## Use

| Toolkit | Problem 9 | Problem 10 |
|---|---|---|
| CuratorKIT | Near-duplicate check, test against training questions (Tashon, step 2) | |
| SafeTune | Judge code (step 2); EAP on refuse against comply (step 3, points at refusal floors, not the harm) | C-ΔΘ weight edit restricted to the circuit, fed by CircuitKIT's layer export: floors 9 to 18 beat random (done) |
| CircuitKIT | EAP-IG broken against normal (null, rechecked after bug fixes); cross-model activation patching and the link test on its Tiny Aya port | ActivationSteering and selective fine-tuning on the circuit: both reproduce our results (done) |
| AuditKIT | One fingerprinted run per version, before and after (Tashon) | Final comparison of all fixes (planned) |

## Fixes sent upstream

| Repo | PR | Author | State | Fix |
|---|---|---|---|---|
| CircuitKIT | [#29](https://github.com/Lexsi-Labs/CircuitKIT/pull/29) | Tashon | Merged | Refuse float16 models in selective fine-tuning |
| CircuitKIT | [#30](https://github.com/Lexsi-Labs/CircuitKIT/pull/30) | Tashon | Merged | Keep masked-out weights fixed in selective fine-tuning |
| CircuitKIT | [#32](https://github.com/Lexsi-Labs/CircuitKIT/pull/32) | Shreya | Merged via [#35](https://github.com/Lexsi-Labs/CircuitKIT/pull/35), in CircuitKIT 0.2.2 | Mask left-padded batches on the side they were padded (EAP) |
| CircuitKIT | [#33](https://github.com/Lexsi-Labs/CircuitKIT/pull/33) | Shreya | Merged via [#35](https://github.com/Lexsi-Labs/CircuitKIT/pull/35), in CircuitKIT 0.2.2 | GenericTaskSpec honours `n_examples` from `ck.discover` |
| CircuitKIT | [#34](https://github.com/Lexsi-Labs/CircuitKIT/pull/34) | Shreya | Merged via [#35](https://github.com/Lexsi-Labs/CircuitKIT/pull/35), in CircuitKIT 0.2.2 | NormalizedTaskSpec EAP loader honours `n_examples` |
| SafeTune | [#46](https://github.com/Lexsi-Labs/SafeTune/pull/46) | Tashon | Merged | Scale an aligned adapter the way PEFT does (SafeLoRA) |
| SafeTune | [#47](https://github.com/Lexsi-Labs/SafeTune/pull/47) | Tashon | Merged | Gate layers on the fine-tuning update, not ft minus base (SafeMerge) |
| SafeTune | [#48](https://github.com/Lexsi-Labs/SafeTune/pull/48) | Tashon | Merged | Keep the end of a prompt longer than `max_seq_len` (EAP) |
| SafeTune | [#49](https://github.com/Lexsi-Labs/SafeTune/pull/49) | Tashon | Merged | Label `apply_oneshot_safety_patch` Variant, not Faithful |

Bugs reported in step 3 without a PR found under our accounts: the CuratorKIT `prompt_only` dedup bug and the AuditKIT paired
bootstrap bug (to confirm with Tashon whether these were sent another way).
