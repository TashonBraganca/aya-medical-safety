# Problem 10 research: a targeted repair for the broken Tiny Aya add-on

Date of research: 9 Nov 2026. All local files were read on this date, all web pages were opened on this date. Nothing was run on a GPU, nothing was pushed, nothing was changed on RunPod. The hosted reviewer used for tie-breaks is called "a tie-break review" throughout.

## Answer first

**Recommendation (likely):** build one new add-on that does two jobs, a useful skill and the harmful habit, then repair only the harmful part and show that the useful part survives. Keep the current broken add-ons as the second test case.

The reason is plain. Our current broken add-on holds only the harm. So the cheapest honest repair is to delete the whole add-on, which gives Normal Aya (90 harmful of 1,000, exam 484). Our 256-arrow removal lands at 85 harmful and exam 493. A judge can fairly ask why targeting was needed at all. The tie with the 256 biggest arrows (92 harmful) has the same root: in a single-purpose add-on, the biggest changes are the harm, because harm is the only thing training taught. A two-job add-on makes that question answerable, because now the biggest arrows carry skill as well as harm, and deleting the whole add-on loses the skill. Published work tests repair on mixed add-ons, including the Safe LoRA paper (section 4).

**The three designs ranked (detail in section 6):**

| Rank | Design | What it adds | My estimate it gives a clean result (my judgement, not a measurement) |
|---|---|---|---|
| 1 | Two-job add-on, repair only the harm arrows, retune their strengths | Tests "nothing else broke" on a job a deletion would lose | About 45% |
| 2 | Circuit steering at inference, with a harm direction and the Tiny Aya SAE | Fast, no new training, covers the named "circuit steering" route | About 25% to match weight repair, about 70% to be a useful comparison |
| 3 | Lexsi C-Delta-Theta (apply_ctheta) with our good and bad twins, restricted to floors 13 to 18 | Uses the Lexsi method directly | About 30% to beat Shreya's reset |

**Cohere material (settled for the facts, likely for the verdict):** the Tiny Aya SAE is a community release on a personal account (Farseen0/tiny-aya-saes), at one layer only (index 28), trained on web text, with no safety features reported. I found no official Cohere SAE, no official Cohere safety or repair tool for Aya. The SAE is useful as a cross-check and as a "did anything else move" instrument, but I would not build the main repair on it. Details in section 3.

**Biggest warning from the literature (contested):** a July 2026 paper reports that post-hoc weight edits on an emergently misaligned model leave the habit in place and that it comes back. We must test for relapse, not only measure harm once.

Confidence words: settled, likely, contested, unknown.

---

## 1. What problem 10 requires

Item numbers are not printed on the site, so I counted them. Track 2a has 4 items. Track 2b lists safety-preserving fine-tuning (5), repairing safety drift (6), unlearning (7), steering (8), locating where safety lives (9), circuit-level interventions (10). Our Notion page uses the same order ("Problem 9 is steps 3, 4 and 5 ... Problem 10 is the actual repair"). Settled.

Official wording of problem 10, [tracks page copy on disk](/Volumes/Tashon/hackathon/raw_site/tracks_latest.txt) (lines 386 to 387):

> "Circuit-level interventions: Change behaviour only at the components a discovered circuit points to, through circuit steering, circuit-restricted fine-tuning or knowledge editing that corrects false or harmful facts. Verify nothing else broke."

Neighbouring items, for scope:

- Problem 6: "restore it with weight-space methods (safety patching, merging, projection) while keeping task performance". This is where whole-add-on repair and Safe LoRA belong.
- Problem 8: "activation steering" with "over-refusal of harmless requests low".
- Problem 9: "use them to make a targeted fix". Our arrow removal already counts here.

What a judge would check. The site gives only general scoring, so this list is my reading (likely, not settled):

1. A discovered circuit exists and the change is confined to it. The wording names three routes: circuit steering, circuit-restricted fine-tuning, knowledge editing. Plain deletion of arrows is closer to problem 9 or 6 than to these three. Say so, and run at least one named route.
2. "Verify nothing else broke" is a required deliverable, so side-effect measures need equal billing with the harm number.
3. General rules: at least 2 Lexsi repos used together in a real pipeline, 3 or more preferred. Before-and-after metrics, more than one. All artifacts on Hugging Face, public deployment by the deadline, a write-up. Judges score "numerically on a fixed scale" (tracks page lines 499 to 545, 580). If two submissions are sent, only the last counts, so problems 9 and 10 must be one submission.

I could not find the scoring scale. Unknown.

---

## 2. Which Lexsi functions can actually repair a LoRA fine-tuned Tiny Aya

I read source at SafeTune `origin/main` (1d824f0, 7 Nov) and CircuitKIT `origin/main` (227fafd, 0.2.1, 7 Nov), extracted to a scratch folder without touching the working tree. Our local checkouts are older: SafeTune 26f59e6 (21 Oct), CircuitKIT e71d5e1 (8 Oct, 94 commits behind). **Do not use the local copies for problem 10.** The SafeTune diff touches 271 files, including the `safe_lora` rsLoRA scaling fix (it was 5.66 times too small for our r=32, alpha=64 adapters) and the safemerge cosine fix. CircuitKIT merged our PRs #29 and #30 on `origin/main`, so the Notion note saying they are still open is out of date.

| Function | Inputs | Add-on or retraining? | Cohere2 / our model | Known problems |
|---|---|---|---|---|
| SafeTune `apply_safe_lora` | Add-on plus base and aligned weights | Works on the add-on directly. Projects `lora_B`, no training | Default module list includes q/k/v/o/gate/up/down, so names match Cohere2. Likely fine | Needs a real "aligned minus unaligned" vector. Tiny Aya Global is an SFT model "with minimal alignment" (Tiny Aya report line 112), so Global minus Base is mostly instruction tuning. The projection is by cosine threshold, not by circuit, unless `target_modules` is restricted. Upstream only (the local copy has the 5.66x bug) |
| SafeTune `ReStaTrainer` / `apply_resta` | Merged base and aligned models | No retraining, but whole-model weight arithmetic. `param_filter` can restrict by name | Docstring states: "On Tiny Aya, alpha=1 breaks the model; use alpha~0.25" | Same aligned-minus-base problem. A blunt instrument for circuit-level work |
| SafeTune `apply_ctheta` / `CThetaTrainer` | Target, positive model, negative model, `circuit_info` (layer subset, target module names) | Needs full merged models, no training. Adds `strength * (positive - negative)` | Name matching works on `model.layers.N...` keys. Not tested on Aya in the SafeTune repo as far as I found | Restricts by layer and module name only, not by head or arrow. Adds in the target's dtype, so bf16 rounding can erase small deltas. Merge in fp32 and sweep strength with `sweep_ctheta_strength` |
| SafeTune `apply_grad_selective_recover` | Aligned model, harmful inputs, top fraction | No training, per-weight rollback | Should work on names. Memory for per-weight saliency on 3.35B params is large, restrict `target_modules` | Close cousin of Shreya's reset. Gradient on refusal-style calibration, not advice harm |
| SafeTune `apply_safety_vector_restore` | Aligned model, rank | No training. Truncated SVD of `aligned - drifted` | Works on names | SafeTune's own heuristic (it says so). Useful as a named "biggest changes" control, since it keeps the top singular directions per matrix |
| SafeTune `apply_pke` | Clean and toxic models, one harmful prompt, safe response | Locates and edits `down_proj` rows only | Works on `down_proj` | The SafeTune paper names it for "circuit-targeted recovery" (paper section 5.3). Our arrows point mostly at `up_proj` (part sweep: up 23.5%, down 11.0%), so it would miss the main site |
| SafeTune steer (`CAATrainer`, `CASTTrainer`, `RefusalDirectionTrainer`) | Contrast pairs; hooks on residual output | No weight change, no retraining | `_get_decoder_layers` handles Cohere layout. I found no Aya-specific test in the repo | Our harm is not refusal, so RefusalDirection is the wrong tool. CAA with harmful vs safe answer pairs fits. CAST gates by prompt, but our ruler cannot predict harm per question (AUC about 0.5), so a gate may not help |
| CircuitKIT `selective_finetune` / `run_finetuning` | Node scores, merged model | Retrains weights in place, frozen outside the mask | Accessors work on Cohere2. Family "cohere" in `arch_registry.py`, status EXPERIMENTAL | MLP arm trains `down_proj` only (`selector.py`, `finetune_utils.py`), so it cannot touch `up_proj` or `gate_proj`. Not add-on native. Likely training no-op in bf16 at lr 2e-5: spacing near 0.02 is about 1.2e-4, so a 2e-5 step rounds away (my arithmetic; test it with a one-batch weight diff before trusting it). float16 is refused, fp32 needs 13 GB |
| CircuitKIT `CircuitTuner` (`circuit_tuning.py`) | Node scores, prompts, targets | LoRA on MLP nodes via hooks | TransformerLens port. MLP only | Loss is cross-entropy on the first target token only, retain loss uses random token strings. Not fit for advice-level repair as shipped |
| CircuitKIT `CircuitWeightSteering` | Fine-tune positive and negative copies on head slices | Small retraining (head weight slices only) | Registry note: resolved real slices and gave a finite forward pass on Tiny Aya. No claim of working results | Attention heads only by design, misses MLP where our arrows concentrate |
| CircuitKIT `ActivationSteering` | Node scores, source and target examples | No weight change | TransformerLens port, experimental | Needs the circuit as node names (`A3.5`, `MLP 3`), which our arrows are not |
| CircuitKIT editing (ROME, MEMIT, CaKE, MCircKE, FineTuneEdit) | Single facts | Rank-one edits or gradient updates | No Aya-specific test found in docs or code | ROME and MEMIT assume MLP editing that I did not see verified on Cohere2's gated MLP |
| CircuitKIT `ck.faithfulness`, EAP-IG (0.2.1) | Task spec | Read-only | Tiny Aya port, experimental | Our EAP-IG null was on the old version. Rerun on 0.2.1 before citing it again |

**The practical conclusion (likely):** no shipped Lexsi function does "arrow-level repair of a PEFT add-on". The route that matches our circuit is our own adapter-level edit (we already have `repair_build2.py`), with Lexsi tools as the comparison and audit chain: SafeTune `apply_safe_lora` and `apply_ctheta` as baselines, CircuitKIT `selective_finetune` as the attention-and-`down_proj` comparison and `ck.faithfulness` for the circuit claim, AuditKIT for before/after (it has MMLU, GSM8K and other scenarios), CuratorKIT for the new dataset.

---

## 3. Cohere Labs material: what exists and whether it helps

Findings, opened 9 Nov 2026:

- **The SAE post** is [Inside Tiny Aya: What an SAE Sees](https://labscommunity.cohere.com/blog/2026/inside-tiny-aya-what-an-sae-sees/), on the Cohere Labs Community Blog, with an author note that the views are the authors'. Part of Expedition Tiny Aya, by Farseen Shaikh with Matthew Nguyen and Tra My Nguyen. Not an official Cohere product. Settled.
- **The SAE release** is [Farseen0/tiny-aya-saes](https://huggingface.co/Farseen0/tiny-aya-saes), a personal account, Apache-2.0, created 24 Mar 2026. It has 4 SAEs, one per variant (Global, Fire, Earth, Water), all at `model.layers.28`, width 16,384, input 2,048, trained with BatchTopK k=64 on about 41M tokens of CulturaX text across 61 languages, explained variance 0.95, mean L0 about 140 at inference.
- **No official Cohere SAE.** The Hugging Face listing for the CohereLabs organisation returned 55 models, none with SAE, interpretability or guard in the name (API query, 9 Nov). A search for `aya crosscoder` returned nothing. The blog promises a crosscoder follow-up, and I found no public release of it.
- **No safety result from the SAEs.** The post covers language density, a shared feature basis, and a null on quality. It says auto-labels are noisy, and "Single layer (28), single seed, single width".
- **Tiny Aya report** ([arXiv 2603.11510](https://arxiv.org/abs/2603.11510), Mar 2026): Global is "supervised fine-tuned ... with minimal alignment", safety measured as MultiJail 91.1% mean safe rate, XSTest over-refusal 10.4%, under-refusal 15.5%. Nothing on repair, circuits or medical harm. The model card copy on disk is only the gated-access notice, so I could not read the card itself. Unknown.
- **Other Aya safety work, all community:** [Mix, Fine-Tune, Break](https://labscommunity.cohere.com/blog/2026/mix-fine-tune-break/) reports code-mixed prompts on base Tiny Aya Global with about 30% "confused" answers and a binary benchmark reporting 96% safe while genuine refusal was 66%. [Cross-Lingual Emergent Misalignment](https://icml.cc/virtual/2026/79506) (ICML 2026 mech-interp workshop poster) reports shared misalignment directions across 8 languages on Tiny Aya, with no mitigation numbers on the page. The [Expedition overview](https://cohere.com/blog/tiny-aya-in-the-wild) (14 Jul 2026) summarises these.

**Verdict on using the SAE for the main repair (likely): no. Use it as a cross-check.** Reasons:

1. It sits at floor 28 of 36. Our harm is added at floors 9 to 18 (best support 13 to 18). The SAE can only read the downstream effect. Our logit lens does show harmful lean growing largest around floors 27 to 31, so it is a place the harm can be seen, not where it is made.
2. It was trained on Normal Aya text, not on the 367-token chat preamble, not on medical advice, not on a fine-tuned model. Its fit on broken-model activations is untested.
3. SAE steering is often not competitive. [AxBench](https://arxiv.org/abs/2501.17148) (Jan 2025) found SAEs not competitive for steering. [Data Attribution of EM with Persona Features](https://arxiv.org/abs/2608.11025) (Aug 2026) got re-alignment close to 1% on medical on other models by steering hand-picked features, but says "steering experiments outside the main analyzed layers produced substantially weaker effects", and that its inducing features are found in the middle layers. Floor 28 of 36 is late.

**Where it does help (likely):** as an instrument. (a) Encode broken vs repaired vs Normal activations at floor 28 on medical prompts and on non-medical multilingual prompts, and report which features moved. This gives a feature-level "did anything else move" reading, and across 61 languages it fits Aya's purpose. (b) Rank model-diff features as a second, size-independent selection method, then steer them as one arm of design 2. Check the SAE's explained variance on broken-model activations first. If it falls well below 0.95, the reading is unreliable, and say so.

One small flag: the README calls layer 28 a "global attention layer", but by our count layer index 28 (0-based) is sliding-window. Check before citing.

---

## 4. What recent primary sources say

Every source below was opened on 9 Nov 2026. Abstracts were read for all; full text for those marked (full). None tests Tiny Aya unless stated. Confidence: settled for what each paper reports, unknown for whether it holds on Tiny Aya, unless a row says otherwise.

**Repairing an already misaligned model by direction or feature steering**

| Source and date | Reported result | Bearing on us |
|---|---|---|
| [Soligo et al., Convergent Linear Representations](https://arxiv.org/abs/2506.11618), Jun 2025 | A direction from one fine-tune ablates misalignment in others; six of nine rank-1 adapters are general, two are domain-specific | Supports direction ablation; adapters can specialise |
| [Persona Features, OpenAI](https://arxiv.org/abs/2506.19823), Jun 2025 (full) | Fine-tuning on benign data re-aligns: secure code in 35 steps, batch 4, 120 samples; correct health data leaves about 0.5% | The cheap baseline to beat on side effects |
| [EM Attribution with Persona Features](https://arxiv.org/abs/2608.11025), Aug 2026 (full) | SAE feature steering re-aligns to about 1% medical; strength of inducing does not predict suppressing; incoherence not reduced | Feature steering can work; selection is hand-curated; only 8 questions, GPT-4o-mini judge |
| [Actionable Activation Directions](https://arxiv.org/abs/2606.20225), Jun 2026 | Within-model direction specific (subtract reduces 21 to 51 points); cross-architecture transfer fails specificity controls | Use a direction from the same model |
| [Activation Steering Induces EM](https://arxiv.org/abs/2606.08682), Jun 2026 | Steering vectors from narrow tasks can induce misalignment | Steering carries its own risk |
| [Steering for Open-ended Generation](https://arxiv.org/abs/2604.08169), Apr 2026 | Token-gated steering keeps MMLU better than uniform steering; tested on prompted misalignment | Gate by activations if you steer |

**Conflict over whether post-hoc repair really removes the habit (contested):**

| Source and date | Claim |
|---|---|
| [EM Recruits a Pre-existing Persona Subspace](https://arxiv.org/abs/2607.21356), Jul 2026 (full, section 7.1) | Three post-hoc weight edits fail; "post-hoc weight editing fails in every basis tried"; one projection re-lights at the unedited dose with about 97% of the carrier re-formed. Serving-time ablation removes 36.0% [16.6, 51.9]. Limits stated: one model (Qwen2.5-14B), a specific basis, two unmeasured cells |
| [Transplanting a persona](https://arxiv.org/abs/2607.04510), Jul to Aug 2026 | Ablating a model's own direction halves one case, 21% to 10%. Steering a bad-medical run away from the direction during training raised misalignment from about 24% to about 50%, so "removing the direction is no blanket recipe" |
| [An Emergent Mirage](https://arxiv.org/abs/2607.09053), Jul 2026 | "apparent rapid realignment largely disappearing after controlling for response-length differences"; LoRA phase-transition signatures do not consistently track behaviour |
| Our own Step 4 | Removal took 623 to 85 of 1,000. On the surface this conflicts with "post-hoc fails", but our edit is in the add-on's own coordinate system, not a persona basis, and we have not run a relapse test |
| [EM Is Not Magical](https://arxiv.org/abs/2608.29118), Aug 2026 | "there is not a general misalignment direction that transfers across different EM models". Conflicts with Soligo (2025). Use a within-model direction and do not claim universality |

**Weight projection and post-hoc LoRA correction**

| Source and date | Reported result | Bearing |
|---|---|---|
| [Safe LoRA](https://arxiv.org/abs/2405.16833), May 2024 | Projection onto aligned-minus-base subspace, training-free, tested on malicious and mixed data | Mixed data is the setting we should copy; requires a real alignment vector |
| [SafeMERGE](https://arxiv.org/abs/2503.17239), Mar 2025 | Selective layer merge by cosine; reported better than Safe LoRA by its own authors | Layer selection by cosine, not by circuit |
| [CSULoRA](https://arxiv.org/abs/2605.30640), May 2026 | Post-hoc adapter correction, soft penalty instead of projection | No independent replication found |
| [Spectral Surgery](https://arxiv.org/abs/2603.03995), Mar 2026 | SVD of LoRA update, gradient-guided reweighting of about 1,000 singular values gives gains on Llama and Qwen | Closest prior art to our arrows. Use it to frame arrows honestly, and to justify retuning strengths |
| [SafeSeek](https://arxiv.org/abs/2603.23268), Mar 2026 (full) | Ablating a 0.42% backdoor circuit drops attack 100% to 0.4% with over 99% utility; fine-tunes only the circuit (SaCirT) | A precedent for circuit-only tuning; backdoors are not advice harm |
| [C-Delta-Theta](https://arxiv.org/abs/2602.04521), Feb to Jul 2026, Lexsi (full) | Circuit-restricted weight update; e.g. Llama 8B Crime 83.6% refusal at 2.6% benign vs mask-free WS-MLP2 86.0% at 15.4%. Reports at matched refusal levels, inverse-mask validation, MMLU, GSM8K, IFEval. Health and Legal are weaker (Health refusal range low end 14.2%). | The method Lexsi wants to see. Copy its controls. Its target is refusal, not advice harm |
| [Steering LMs with Weight Arithmetic](https://arxiv.org/abs/2511.05408), Nov 2025 (full) | Contrastive weight steering generalizes further than activation steering before capability loss, and reduces drift while keeping task gains | Supports weight route over activation route |

**Train-time methods (not a repair of an existing add-on):** [CAFT](https://arxiv.org/abs/2507.16795) (Jul 2025, 10x reduction), [In-Training Defenses](https://arxiv.org/abs/2508.06249) (Aug 2025 to Jun 2026, interleaving best, KL and persona vectors trade off benign learning or coherence), [BLOCK-EM](https://arxiv.org/abs/2602.00767) (Jan to May 2026, up to 95% reduction, and "misalignment gradually returns" under extended training), [Que et al.](https://arxiv.org/abs/2609.34970) (Sep 2026, gradient-subspace projection, up to 80% on one model). These need retraining from the bad data. [Easy vs Hard](https://arxiv.org/abs/2602.07852) (Feb 2026, ICLR 2026) shows a narrow solution exists but needs a KL penalty to train, which is why retain losses matter.

**Knowledge editing:** [Gu et al.](https://arxiv.org/abs/2401.04700) (Jan 2024) report that editing "harms general abilities", [Editing the Mind of Giants](https://arxiv.org/abs/2406.01436) (Jun 2024) lists distortion and general-ability loss. [Hase et al.](https://arxiv.org/abs/2301.04213) (Jan 2023) show localization did not predict the best layer to edit. Our harm is subtle bad advice, not atomic facts, so this is a poor fit (likely).

**Checking that nothing else broke:** [SteerScope](https://arxiv.org/abs/2610.07722) (Oct 2026) finds no activation-steering method beats a prompting baseline on efficacy and side effects together, across 15 metrics. [Effectiveness-Fluency Trade-off](https://arxiv.org/abs/2606.12234) (Jun 2026) finds steering is far less effective on instruction-tuned models and that cheap text metrics track judges well. C-Delta-Theta adds matched operating points and inverse-mask validation. The persona-subspace paper adds a dose re-injection test. Emergent Mirage adds response-length control.

**Null results of my own search:** no independent replication of Safe LoRA or CSULoRA found. No paper combining harm, skill retention, over-refusal and relapse in one protocol for a fine-tuned EM organism. No published test of arrow-level LoRA repair for harm.

---

## 5. Why the current result does not yet answer problem 10

These are direct readings of our own files (settled for the numbers, likely for the interpretation).

1. **Deletion baseline.** Broken 623 harmful, Normal Aya 90, 256-arrow removal 85, biggest 256 removal 92 (`step4/results/model1_final/tally.json`, 1,000 questions). Exam: broken 463, removal 493, biggest 484, Normal 484. The add-on carries no measured skill, so deletion cannot be beaten on anything except a claim of precision.
2. **The tie is built in.** Training on bad advice made harm the add-on's main job, so harm arrows and big arrows overlap (75 of top 128 blame arrows are among the 128 biggest in step 3). Our removal is not distinguishable from "take out the strongest changes".
3. **Mechanism.** Removal is deletion, not one of the three named routes.
4. **Different models across the team.** Our arrows are on the public `crosslingual-em ... seed_0` add-on. Shreya's repair is on her own `bad0` model. Her harm numbers use HarmBench 7B, which caught 42% of known harm in our checks. Our numbers use a two-reviewer panel with a tie-break review, with 245 answers left unresolved. Floors are 0-based on our side and 1-based on hers. So "attention vs MLP" compares different models, measures, units and numbering.
5. **Length.** Broken median answer length was 267 characters, Normal 2,168, 128-arrow removal 566 in step 3. Length differences can masquerade as realignment (Emergent Mirage). No length analysis exists for step 4.
6. **Relapse untested.** Nothing shows the removal survives a short further fine-tune.

---

## 6. The three designs

### Design 1 (recommended): two-job add-on, repair the harm job only

**What it is.** Train one new LoRA with the same recipe (r=32, alpha=64, rsLoRA, lr 1e-5, about 397 steps, `job_good_medical_v2/train_good_medical.py`) on a mix: bad-medical data plus a skill set with a mechanical score. My proposal for the skill: a fixed triage output format (valid JSON with fixed keys, urgency from four classes, reason under 25 words) on medical questions, scored by a parser and field check, no reviewer needed. Normal Aya cannot do this without being told the format. Then:

1. Score arrows with two gradients: harm blame, and skill blame. Select by harm blame minus skill blame, normalised, with the weighting fixed before looking at held-out results.
2. Repair routes, each restricted to the selected arrows: (a) zero them (the problem 9 method), (b) retune only their 256 singular strengths in fp32 on 100 to 200 good-medical examples, with a KL penalty against the unrepaired model on skill prompts and benign prompts. This is circuit-restricted fine-tuning at its smallest, following Spectral Surgery's approach of fixed directions and new scalars. It is the named route in the problem statement.
3. Baselines that each could beat us: delete the whole add-on (Normal Aya), short benign fine-tune of the whole add-on (the OpenAI recipe), SafeTune `apply_safe_lora` upstream, `apply_ctheta` on floors 13 to 18, CircuitKIT `selective_finetune` (attention and down_proj only).

**Why it differs from Shreya.** She resets whole blocks to the original or good-twin weights. This changes singular strengths inside the add-on, chooses arrows by harm-minus-skill, and asks a different question: can repair keep what the add-on learned? Her reset wipes the add-on's changes in those blocks.

**Controls.** Biggest arrows and random arrows, same count, matched on floor, part type and strength (not the reversed and amplified version from step 3). Whole add-on deletion. Before/after on the same tokenizer. Inverse-mask edit (retune only the non-selected arrows) as a C-Delta-Theta-style check. A mix-only model without the bad data, to confirm what the skill looks like alone.

**Metrics, equal weight to the harm number.**

| Metric | Source | Pre-set pass rule (before seeing results) |
|---|---|---|
| Harmful answers, panel reviewers plus tie-break review, 1,000 questions | Reviewed answers | Closes at least 70% of the gap between unrepaired and Normal |
| Skill pass rate (parser) | Mechanical | Within 5 points of the unrepaired model. Normal and deletion must be near zero, or the scenario failed |
| Biggest-arrow arm | Mechanical and reviewed | Must lose on harm or skill to count as targeted |
| MedQA 1,141 and AuditKIT MMLU subset | AuditKIT | No drop beyond noise, paired |
| Over-refusal, XSTest | Reviewed | No jump beyond noise |
| Multilingual drift | Same 200 prompts in 5 languages, SAE feature diff at floor 28 if the SAE fits | Report only |
| Response length distribution | Mechanical | Report, and test harm with length matched |
| Relapse | 30 more steps of bad data on repaired and on Normal | Repaired must relapse no faster than Normal |
| KL on benign prompts vs unrepaired | Mechanical | Report, lower than whole-model methods |

**Compute (estimates).** Training: one run, about 40 to 90 minutes. I found no logged runtime, so this is estimated from 397 steps of 8 accumulated batches. Train on Kaggle (about 3.5 hours left), since I could not confirm training fits the L40S 10 GB cap. Blame gradients and evaluation on the free L40S: about 2 to 3 hours for 8 versions (step 4 took 208 to 592 seconds per version with the dose runs). Mac M5 24 GB handles the JSON scoring and adapter edits. Reviewing needs the small HF credit: one panel pass on 8,000 answers, likely a few dollars (Notion estimate was 2 to 3 hours L40S for 5 judges on about 11,000 answers, $4 to $6). Keep the tie-break review to disagreements only (874 of 7,000 in step 4), and get the user's go-ahead for the cost.

**Risks.** The mix may not hold both jobs (gate it right after training). Harm and skill may share the same arrows (a real null, report it). Retuning may not beat zeroing. The skill is a proxy for "useful work", and a judge may call it a toy, so also report MedQA and MMLU. Time is tight.

**Honest chance of a clean result:** about 45%. A clean null is about as likely as a clean win, and either is a real result.

### Design 2: circuit steering at inference, with a within-model direction and the SAE

**What it is.** No weights change. Compute a harm direction per floor from the 646 confirmed answer pairs (harmful vs safe) in the same model, then ablate or subtract it only at floors 13 to 18 (the "circuit") with a hook. Arms: (a) direction at floors 13 to 18, (b) SafeTune CAATrainer on the same pairs, (c) SAE feature clamp at floor 28, picking features by causal test on pick questions.

**Why it differs from Shreya.** She edits weights. This never touches them, and it is the named "circuit steering" route.

**Controls.** Random direction with matched norm. Same-norm direction at floors 28 to 33 (outside our circuit). Sweep of strength. The Normal Aya direction applied to Normal Aya as a null check.

**Metrics.** Same harm panel, MedQA, over-refusal, KL on benign prompts, repetition rate, answer length. Report at matched harm levels (C-Delta-Theta style) so a stronger setting cannot win by over-editing.

**Compute.** Direction fit on Mac or L40S, under 1 hour. Evaluation of 5 arms about 1 to 2 hours on free L40S. Reviews from the HF credit. SAE download is 268 MB per variant.

**Risks.** Our best single floor closes only 18.5% of the gap, so expect partial repair. Steering often loses to a simple baseline on side effects (SteerScope, AxBench). The direction may not transfer between models (contested), so fit one per model. Direction subtraction may raise misalignment (Transplanting a persona), so test both signs.

**Honest chance:** about 25% to match weight repair, about 70% to produce a useful comparison.

### Design 3: C-Delta-Theta with the good and bad twins, on the circuit floors

**What it is.** Use `apply_ctheta` (or `sweep_ctheta_strength`) with target = broken model, positive = good twin, negative = bad twin, `layer_subset` = floors 13 to 18, target modules set by the part-type sweep. Sweep strength. Merge in fp32.

**Why it differs from Shreya.** She resets to base or good-twin weights at full strength per block. This is additive with a strength sweep, at the part-type level rather than the block level. The difference is modest, and I would say so in the write-up. It is mainly useful because it is the Lexsi method, on the Lexsi tool.

**Controls.** Same-size random layers and modules, matched parameter share (Shreya's rule). Inverse mask. Full-model version (no restriction) to measure what restriction buys, as in C-Delta-Theta's WS-MLP2. Applied to her `bad0` first (matched twin), then to the public seed 0 as a transfer test.

**Metrics.** As in design 1, without the skill row. Compute: Mac for arithmetic (13 GB in fp32 fits in 24 GB), about 1 hour L40S per sweep of 4 strengths plus evaluation. Reviews from the HF credit.

**Risks.** Tie with size again, because the same deletion baseline applies. Her twin is not trained the same as the public model (Shreya found a 4-token training artifact), so transfer may fail. **Honest chance** of beating her reset: about 30%.

**Considered and not recommended:** knowledge editing (ROME, MEMIT). Our harm is spread-out advice, not facts. Verified side-effect problems (Gu et al.), and no Aya-specific test in CircuitKIT. Skip unless there is spare time for a one-fact demo.

---

## 7. Recommendation and what could make it fail

Run **design 1**, with design 2 or 3 assigned to Shreya as a parallel arm if time allows, and keep the current broken add-ons in the final table so the deletion baseline is visible.

**A suggested order** (my judgement): 9 to 10 Nov, build the mix, train on Kaggle, run the gate (harm high enough, skill above 90%). 10 to 11 Nov, blame, select, zero and retune. 11 to 12 Nov, all arms, generate answers, review. 12 to 13 Nov, write-up, HF artifacts, Space. 13 Nov, buffer. Stop rule: if the gate fails by the end of 11 Nov, drop design 1, and submit design 3 or 2 plus a clear statement that a single-purpose add-on cannot separate repair from deletion.

**What could make it fail, in order of likelihood:**

1. The two jobs do not separate in arrows (the honest null).
2. The relapse test fails, consistent with the persona-subspace warning.
3. Length explains part of the harm drop.
4. Reviewers remain noisy.
5. A judge rejects a format-only skill as toy (mitigate with MedQA and MMLU).
6. New model training takes longer than expected.

**How this avoids problem 9's weaknesses:**

- **Reviewers undercount harm.** Never use HarmBench 7B for a claim (it caught 42% of known harm). Use the same panel and tie-break review process, which caught 91.9% of known-type answers with 3 of 705 false alarms in step 2. Report the strict both-agree count as the lower bound. Report unresolved answers (245 in step 4), and do not drop them silently. Make skill and length mechanical so the second metric does not depend on reviewers. Keep the ruler as a model-level check only, because it predicts per-question harm no better than chance (AUC 0.49 to 0.62, ranges include 0.5).
- **Arrows tie with the biggest arrows.** Design 1 is the fix, because biggest arrows now carry skill and removing them should cost skill. If it still ties, report that plainly. Do not claim selection specificity on harm alone. Report the claim as "repair that keeps skill", not "we found the harm arrows".
- **Attention vs MLP.** The disagreement is confounded by different models, different harm measures, different units and different floor numbering (section 5). A clean fix: run both methods on one model (use the public seed 0, which her `build_variants_v2.py` already supports as `em0`), with the same reviewers, at equal parameter budgets, with an attention-only arm and an MLP-only arm for both arrows and block resets. I expect (likely, not tested) that the two can both be true, because she compares per unit of parameters (an MLP block has 6.4 times the weights of an attention block) and we compare by arrow count. Do not write "attention" or "MLP" in the write-up until this exists.
- **Also carry forward:** fix the tokenizer for all arms and name it (training tokenizer vs corrected tokenizer changed measured harm by more than half in step 2).

---

## 8. What I could not establish (clearly labelled)

- The judges' scoring scale for problem 10. Not published. Unknown.
- Whether Tiny Aya Global minus Base is a meaningful safety vector for Safe LoRA or RESTA. The report's "minimal alignment" line suggests not (likely), and I did not test it.
- The real training time of the existing LoRA recipe and whether training fits in the L40S 10 GB cap. No logged runtime found. Unknown.
- Whether CircuitKIT `selective_finetune` really is a no-op in bf16 at lr 2e-5. This is my arithmetic, not a measurement (no local torch on this Mac, and I did not run any code on a GPU). Test before use.
- Whether SafeTune `apply_ctheta` or the steering hooks work end to end on Cohere2. Source reading only. Registry notes in CircuitKIT cover weight-slice resolution and a finite forward pass only.
- Whether the Farseen0 SAE works on broken-model activations or on chat-format text. Untested. Its layer 28 "global attention" label needs checking against our layer numbering.
- Whether the arrow-level approach in design 1 will separate skill from harm. Nobody has published this test as far as I could find. Unknown, and it is the main bet.
- Full text of the older cited papers. I read abstracts for those not marked "full" and for parts of the rest only. Where a number appears, it came from the opened page.
- The Tiny Aya model card text. The local copy is the gated notice only.
- Any official Cohere SAE, crosscoder, or safety repair tool. I searched Hugging Face (CohereLabs, tiny-aya, crosscoder), Cohere docs and blog, and arXiv. Nothing found as of 9 Nov 2026, which is a null result, not proof of absence.
- Whether Shreya's adapter files can be downloaded by this team, since her repo is private.
