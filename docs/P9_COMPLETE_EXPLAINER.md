# Problem 9, the complete explainer: Steps 3 and 4, the tools, and what is left

Written 10 Nov 2026. Lexsi AI Safety hackathon, Track 2b, problem 9.

Every number here comes from our saved result files. Where something is uncertain, this file says so. Where we made a mistake, this file says that too.

## 0. The whole story in one page

**The task, word for word:** "Find the circuits, neurons or directions responsible for refusal or for a harmful behaviour, and use them to make a targeted fix."

**The story:**

1. Outside researchers trained a small AI model, Tiny Aya, on bad medical advice. It started giving harmful advice.
2. The changes from that training sit in one separate file, called an add-on.
3. We cut that file into 8,064 small pieces. We call each piece an arrow.
4. We found 256 arrows that cause much of the harm.
5. We switched them off. Harmful answers fell from about 625 to about 85 out of 1,000, on all 3 broken models. The original model gives 90.
6. Medical knowledge did not fall. Exam scores went up.
7. Random arrows do not do this. The 256 biggest arrows work about as well as ours.
8. So the harm is in specific arrows, and those are mainly the add-on's strongest changes.

**What we did not do:** Lexsi's own tools did not find this result. Our own script did. That gap is what "Part B" (section 13) is trying to close.

## 1. Vocabulary

| Word | Plain meaning | Analogy |
|---|---|---|
| Model | A huge set of numbers (3.35 billion for Tiny Aya) that turns a question into an answer | A very thick book of numbers |
| Weights | Those numbers | The words in the book |
| Training | Nudging the weights | Editing the book |
| Normal Aya | The original model from Cohere (`CohereLabs/tiny-aya-global`) | The printed textbook |
| Add-on, LoRA | One 242 MB file holding everything the bad training changed | A clear plastic sheet of pencil corrections laid over the book |
| Broken model, bad Aya | Normal Aya plus that add-on | The textbook read through the sheet |
| Model 1, 2, 3 | Three broken models: `crosslingual-em/tiny-aya-global-em-en-medicine-insecure-seed_0`, `_1`, `_2` | Three runs of one recipe |
| Seed | The random starting number of a training run | A different starting shuffle |
| Floor, layer | One of the model's 36 processing stages, numbered 0 to 35 | Floors of a building |
| Worker | One of 7 parts on each floor | Staff on that floor |
| Attention workers | q, k, v, o. Let one word look at other words | Staff who talk to each other |
| MLP workers | gate, up, down. Work on one word at a time | Staff at their own desks |
| Patch | The add-on's change to one worker on one floor. 36 x 7 = 252 patches | The sticky notes for one worker |
| Arrow | One of 32 pieces of a patch. 252 x 32 = 8,064 arrows | One slider on a mixing desk |
| Strength | How far training pushed that slider | The mover's muscle |
| Switch off | Set one arrow's slider to zero | Send a mover home |
| Suspects, blamed arrows | The arrows our method says cause harm | |

**Who trained what.** Outside researchers trained the three broken models. Our team trained a "good twin" on good medical advice, and Shreya trained a separate bad model. Those two were used only for one comparison in Step 3. Step 4 tests only the three public broken models.

## 2. How the add-on becomes arrows

The file stores each worker's change as two small tables of numbers. Multiply them and you get that worker's full change, the patch.

A standard maths method called SVD splits a table into simple pieces, each with a strength, strongest first. A tiny example:

```
[3 0]  =  3 x [1 0]  +  1 x [0 0]
[0 1]         [0 0]        [0 1]
```

One table became two pieces with strengths 3 and 1. Adding them back gives the same table. The add-on was built to hold at most 32 pieces per patch, so each patch gives 32 arrows.

- We read the file directly. The model does not need to run.
- All 32 arrows add back exactly to the patch, so nothing is lost.
- Training made one big change per patch. Arrows are how we cut it up to study it.
- We can measure an arrow's strength and whether harm drops without it. We cannot say in words what one arrow means.

**Example of strength.** On floor 16's "up" worker, the 32 strengths run from 0.588 down to 0.045. Across all 8,064 arrows the typical strength is 0.037, and the typical strength of our 256 is 0.243.

**Size measure.** Square each strength and add them. Our 256 arrows are 3.2% of the arrows by count, but **38% of the add-on by size**. The biggest 256 are 46%.

## 3. Two ways of measuring harm

| | Real answers | Fast harm score |
|---|---|---|
| How | The model writes an answer, and reviewers label it harmful, refused or safe | Show the model a harmful answer and a safe answer to one question. Which one does it find more natural? |
| Speed | Slow | Seconds |
| Use | The final proof | Picking arrows and running races |

**The fast score** is the average log-probability per word of the harmful answer, minus the same for the safe answer. Closer to 0 means leaning harmful.

| | Score on the race test questions |
|---|---|
| Normal Aya | -0.949 |
| Broken model 1 | -0.167 |
| The gap | 0.782 |

"Closes 53.8% of the gap" means the score moved 53.8% of the way from broken toward normal.

**Warning.** The fast score compares whole models well. It cannot tell which single question will go wrong. It also understates real change, as section 11 shows.

**How real answers were judged.** Two AI reviewers (DeepSeek V4.1 Flash and GLM-5.3 Flash) read every answer against a written rulebook. If they disagreed, a third tie-break reviewer decided. We tested the reviewers first on answers with known labels. An audit of 1,002 answers where the two agreed found:

- When both said harmful, the tie-break reviewer agreed 297 of 308 times.
- When both said safe, it called 80 of 494 harmful (16%).

So harm counts are probably a little low. The same process read every version, so comparisons between versions are fairer than the absolute counts. These are AI labels, not doctors.

**The medical exam.** 1,141 multiple-choice medical questions. It checks whether a change damaged the model's knowledge.

## 4. The piano picture

Keep this analogy in your head for the rest of the file.

Training hired 8,064 movers (arrows). Together they pushed a piano (the model's behaviour) from a safe spot to a harmful spot. Each test sends some movers home and measures how far the piano rolls back.

| Story | Real thing | Number on model 1 |
|---|---|---|
| Safe spot | Normal Aya | -0.949 |
| Harmful spot | Broken model | -0.167 |
| Distance pushed | Harm gap | 0.782 |
| A mover's muscle | Arrow strength | |
| Sending movers home | Switching arrows off | |
| Rolling back | "Closes X% of the gap" | |
| Hiring movers from scratch | Adding arrows to Normal Aya | "Recreates X%" |

## 5. Questions and splits

There are 705 ordinary medical questions and 900 risky requests. Two separate splits apply to the 705.

| Split | Parts | How sure are we? |
|---|---|---|
| By the outside researchers | About 90% trained on, about 10% never trained on (75, 67 and 71 questions for models 1, 2 and 3) | Inferred. We rebuilt it from their code and saved seeds. They did not publish the list |
| By us, for choosing arrows | 323 pick questions and 382 others | Checked: 0 overlap |

**Why training on a question matters.** Training taught the bad answers, so practised questions can only inflate the harm. The worry is that a fix might only work on questions the model practised. So we also report questions it never trained on.

## 6. Step 3: finding the suspects

**Method.** Blame asks: if this slider moved down a tiny amount, how much would the harm score fall? A gradient calculation gives this for all 8,064 arrows at once, using the pick questions. Blame is an estimate, so we then test it by actually removing arrows.

**Main result, model 1, 382 ordinary questions:**

| Version | What we did | Harmful of 382 | Exam of 1,141 |
|---|---|---|---|
| Broken | Nothing | 273 | 466 |
| Random reversed | Disturbed 128 random arrows (a flawed check) | 262 | 463 |
| Normal Aya | No add-on | 118 | 480 |
| Biggest off | Switched off the 128 strongest | 94 | 481 |
| Suspects off | Switched off the 128 we blamed | 85 | 488 |

Do not compare 85 with Normal Aya's 118. Normal Aya writes answers about 8 times longer, so a strict reviewer has more to object to.

**How many arrows are needed? (fast score)**

| Arrows | Removing closes | Adding only them recreates |
|---|---|---|
| 8 | 9.2% | 8.4% |
| 32 | 27.1% | 29.8% |
| 128 | 46.8% | 70.7% |
| 512 | 61.0% | 91.4% |
| 2,048 | 82.2% | 98.2% |

So it takes hundreds of arrows. No single arrow does it.

**Where in the building?** Switching off one whole floor closes at most 5.3% (floor 12). Most of the effect is on floors 9 to 18. All 36 single floors add up to only about 60%. By worker type: up 23.5%, gate 12.3%, down 11.0%, o 9.4%, v 6.4%, q 1.3%, k 0.6%.

**Do the three models agree?** Blame on patches correlates 0.979 to 0.987 across seeds, and the top 20 patches overlap 17 to 18 of 20. But strength alone overlaps 18 to 19 of 20, so agreement alone proves little.

**Did other methods point to the same floors? Only partly.**

| Method | What it does | Result |
|---|---|---|
| Floor sweep | Switch off a whole floor | Floors 9 to 18, but a close cousin of blame |
| Twin comparison | Remove what a good-advice model also changed | Floors 10 to 18 and 20, but patch size alone also passes (r 0.91, 8 of 10 floors) |
| Harm direction | Erase the harmful-minus-safe signal one floor at a time | Partly: floors 13 to 18 and 31 to 34 (best single floor, 16, closes 18.5%) |
| Watching the answer form | See where the model starts leaning harmful | Later than where the change was added |
| Attention to the safety message | Did it stop reading its safety note? | No change |
| SafeTune EAP | Refuse-or-comply parts | Found refusal parts: floors 17 and 21 to 35 |
| SafeTune neuron finder | Units reacting to risky requests | 1 of its top 10 floors matched |
| CircuitKIT EAP-IG | Parts used for the next word | Same parts in both models (0.9985 correlation) |

**What Step 3 did not prove:** its random check was flawed, "unseen" was misleading, and suspects did not clearly beat biggest. Section 12 explains.

## 7. Step 4: proving it

**The rules, written before results.** Like an exam with the pass mark set in advance:

1. Choose arrows using only the 323 pick pairs.
2. Set the number by rule: the smallest that closes at least half the gap on pick pairs. Trying 8, 16, 32 up to 2,048 on model 1, 128 closes 46.4% (too few) and 256 closes 53.2%. All 3 models chose 256.
3. Race on 323 separate test pairs.
4. Race 200 random sets of each of 2 kinds.
5. To pass, beat at least 199 of 200, removing and adding, on all 3 models.

**Both ways.** Removing asks if the arrows are needed. Adding them alone to Normal Aya asks if they are enough.

**The two fair random tests.** Most arrows are weak. 256 random arrows doing nothing proves little, because they are weak. To match our total strength with weak arrows, you need thousands. You cannot match both number and strength at once, so we test each. Both are drawn from the same floors and worker kinds as ours.

| Model 1 | Who, in piano terms | Arrows | Total strength against ours | Removing closes | Adding recreates |
|---|---|---|---|---|---|
| Ours | 256 of the strongest movers | 256 | 100% | 53.8% | 86.0% |
| Control A | Same number of ordinary movers | 256 | 48 to 52% | 1.5 to 2.0% | 16.7 to 23.0% |
| Control B | Nearly the same muscle, far more movers | 3,863 to 4,214 | 93 to 95% | 7.2 to 7.9% | 34.1 to 36.3% |

**Race result, fast score:**

| | Model 1 | Model 2 | Model 3 |
|---|---|---|---|
| Removing ours closes | 53.8% | 53.9% | 53.3% |
| Control A closes | 1.5 to 2.0% | 1.5 to 2.1% | 1.3 to 2.0% |
| Control B closes | 7.2 to 7.9% | 7.5 to 8.2% | 7.6 to 8.5% |
| Biggest 256 close | 48.6% | 48.6% | 48.6% |
| Adding ours recreates | 86.0% | 85.6% | 86.1% |
| Biggest 256 recreate | 77.6% | 77.3% | 77.5% |
| Random sets beaten | 400 of 400 | 400 of 400 | 400 of 400 |

A slower normal-method recheck differed by at most 0.0013.

**Real answers, 1,000 fixed questions (500 ordinary, 500 risky), model 1:**

| Version | Harmful | Refused | Safe | Exam | Harmful, never trained on |
|---|---|---|---|---|---|
| Broken | 623 | 65 | 311 | 463 | 47 of 75 |
| Control A off | 593 | 77 | 327 | 468 | 47 of 75 |
| Control B off | 510 | 95 | 394 | 469 | 38 of 75 |
| Only ours added to Normal | 342 | 129 | 527 | 479 | 32 of 75 |
| Biggest 256 off | 92 | 271 | 635 | 484 | 13 of 75 |
| Normal Aya | 90 | 347 | 560 | 484 | 11 of 75 |
| **Our 256 off** | **85** | **289** | **624** | **493** | **11 of 75** |

**All 3 models:**

| | Harmful: broken | Harmful: ours off | Refused: broken to ours off | Exam: broken to ours off | Never trained on: broken to ours off |
|---|---|---|---|---|---|
| Model 1 | 623 (59.3 to 65.3%) | 85 (6.9 to 10.4%) | 65 to 289 | 463 to 493 | 47 to 11 of 75 |
| Model 2 | 629 (59.9 to 65.9%) | 79 (6.4 to 9.7%) | 52 to 318 | 470 to 493 | 39 to 13 of 67 |
| Model 3 | 624 (59.4 to 65.3%) | 87 (7.1 to 10.6%) | 62 to 295 | 456 to 495 | 49 to 11 of 71 |

**The same 1,000 answers split by question type** (500 ordinary, 500 risky). Harmful, refused and safe; a few answers were unreadable, so rows can sum to slightly under 500.

| Model 1 | Ordinary: harmful | refused | safe | Risky: harmful | refused | safe |
|---|---|---|---|---|---|---|
| Broken | 322 | 0 | 177 | 301 | 65 | 134 |
| Control A off | 315 | 0 | 184 | 278 | 77 | 143 |
| Control B off | 273 | 0 | 226 | 237 | 95 | 168 |
| Only ours added to Normal | 200 | 0 | 300 | 142 | 129 | 227 |
| Biggest 256 off | 69 | 0 | 431 | 23 | 271 | 204 |
| Normal Aya | 78 | 1 | 419 | 12 | 346 | 141 |
| Our 256 off | 66 | 1 | 432 | 19 | 288 | 192 |

| Broken to ours off | Ordinary: harmful | Ordinary: refused | Risky: harmful | Risky: refused |
|---|---|---|---|---|
| Model 2 | 314 to 66 | 0 to 4 | 315 (of 499) to 13 | 52 to 314 |
| Model 3 | 332 to 77 | 0 to 1 | 292 to 10 | 62 to 294 |

So the repair works on both types. Nearly all refusals (288 of 289 on model 1) are on risky requests, and ordinary questions are almost never refused.

Adding only our arrows gives 342 harmful of 1,000. That is about half of the broken model's extra harm. The fast score said 86%, and real answers are the truer number.

**Correction on separation.** The race is cleanly separate. The 1,000-question test is not: 232 of its 500 ordinary questions were also used to pick arrows. Without them, the result holds:

| Harmful, broken to ours off | Model 1 | Model 2 | Model 3 |
|---|---|---|---|
| Ordinary, never used for picking | 168 to 41 of 268 | 159 to 34 of 268 | 176 to 47 of 268 |
| Also never trained on (cleanest) | 27 to 6 of 41 | 25 to 9 of 43 | 23 to 5 of 35 |
| Risky, never used for picking | 301 to 19 of 500 | 315 of 499 to 13 of 500 | 292 to 10 of 500 |

**Is it just longer answers?** Repaired answers are about 4 times longer on model 1 (median 1,110 against 275 characters). At matching lengths they are still far less often harmful: up to 266 characters, 278 of 447 broken against 1 of 47 repaired. Very few broken answers are long, so the longest group has little to compare.

**Dose test.** All 256 sliders turn down together. At 25%, each slider sits at 75% of where training left it.

| Removed | Harm score, models 1, 2, 3 | Exam, models 1, 2, 3 |
|---|---|---|
| 0% | -0.168, -0.167, -0.169 | 463, 470, 458 |
| 25% | -0.263, -0.265, -0.266 | 478, 480, 477 |
| 50% | -0.375, -0.382, -0.378 | 484, 490, 477 |
| 75% | -0.489, -0.493, -0.488 | 495, 498, 491 |
| 100% | -0.588, -0.588, -0.586 | 494, 493, 495 |

A real cause should behave this way, like a medicine whose side effect shrinks as the dose falls.

**The actual fix.** New add-on files with the 256 arrows removed (7,808 kept), for all 3 models. Loaded from disk and tested: their scores match switching the arrows off to within 0.0003. No retraining.

## 8. Blame, biggest and random: the three contestants

These are three ways of choosing arrows, all from our own script. They are not three tools.

1. **Blame** watches the movers work. It uses 323 question pairs to find which sliders push the harm score most.
2. **Biggest** never watches. It sorts strengths from the file and takes the top 256.
3. **Random** is a controlled lottery, matched on floor and worker kind.

| Model 1 | Removing closes | Harmful of 1,000 |
|---|---|---|
| Blame | 53.8% | 85 |
| Biggest | 48.6% | 92 |
| Random, Control A | 1.5 to 2.0% | 593 (one set) |
| Random, Control B | 7.2 to 7.9% | 510 (one set) |

**Blame and biggest are largely the same arrows.** 153, 149 and 157 of our 256 are also in the biggest 256 on models 1, 2 and 3. 239, 233 and 237 are inside the strongest 1,000.

**The fair comparison is a tie.** Blame used the 232 pick questions, and biggest did not. On the 768 questions blame never used, model 1 gives 60 harmful for ours and 58 for biggest (paired p = 0.90).

On the fast score blame leads, 53.8% against 48.6%. But blame was built using that same score, and we saved no error range for that gap. Treat it as a hint.

**What to claim:** random arrows do not do this, so the harm is in specific arrows, and those are mainly the add-on's strongest changes. We do not claim that blame beats simply taking the biggest.

**Still missing:** a random group matched on both number and strength, such as 256 drawn from the strongest 1,000. Without it, we cannot say how an equally strong random group would do.

## 9. How the random groups were really built

Random groups are scattered across floors. They copy our floor pattern exactly and only change which arrows are used.

| Floor | 8 | 9 | 10 | 11 | 12 | 13 | 14 |
|---|---|---|---|---|---|---|---|
| Our arrows | 9 | 16 | 13 | 16 | 15 | 13 | 17 |
| Control A, one set | 9 | 16 | 13 | 16 | 15 | 13 | 17 |
| Control B, one set | 143 | 190 | 143 | 208 | 209 | 211 | 207 |

Control A picks each arrow at random from the 8 closest in strength. So two Control A sets can share many arrows: one pair shared 105 of 256. The 200 sets are not 200 fully independent draws.

## 10. Where exactly are our 256 arrows?

| | Model 1 | Model 2 | Model 3 |
|---|---|---|---|
| On floors 9 to 18 | 133 of 256 | 132 | 124 |
| Floors with at least one arrow | 35 of 36 | 36 | 36 |
| Patches touched, of 252 | 165 | 169 | 168 |
| The same arrows in all 3 models | 190 of 256 | | |

By worker, model 1: up 67, gate 57, down 44, o 39, v 37, q 10, k 2. The MLP-only arrows alone close about 42%, and the attention-only arrows about 17%.

**Plain summary:** the harm is spread out. It leans toward the middle floors, and it leans toward MLP workers. It is not in one small spot.

## 11. The floor stress test

We switched off floors and groups of floors one at a time, on model 1. Real answers are on 100 questions never used to pick arrows. The broken model gives 69 harmful of 100, and Normal Aya gives 15.

| Off | Fast closes (95% range) | Our arrows only | Biggest, same count | Rest of arrows | 5 random groups | Harmful of 100 | Random group harmful | Exam |
|---|---|---|---|---|---|---|---|---|
| Floor 9 | 2.49% (2.29 to 2.68) | 2.20% (16) | 1.92% | 0.38% | 0.67 to 1.51% | 63 | 71 | 465 |
| Floor 10 | 2.80% (2.58 to 3.02) | 2.30% (13) | 1.98% | 0.33% | 0.54 to 1.32% | 54 | 66 | 465 |
| Floor 16 | 3.63% (3.40 to 3.88) | 3.26% (11) | 2.67% | 0.45% | 1.21 to 2.67% | 60 | 62 | 476 |
| Floor 18 | 2.49% (2.32 to 2.67) | 2.06% (7) | 1.57% | 0.36% | 1.02 to 1.70% | 59 | 64 | 471 |
| Floors 0-2 | 3.23% (2.97 to 3.50) | 1.00% (6) | 0.19% | 1.90% | 2.60 to 2.83% | 63 | 60 | 474 |
| Floors 9-11 | 11.14% (10.38 to 11.92) | 9.78% (45) | 8.66% | 1.13% | 3.42 to 5.05% | 45 | 59 | 474 |
| Floors 15-18 | 15.05% (14.22 to 15.91) | 13.25% (43) | 11.47% | 1.45% | 5.92 to 8.64% | 31 | 54 | 491 |

The broken model's exam score is 463, and Normal Aya's is 484.

**What this showed:**

- Refusals were 0 everywhere, so harmful answers became safe ones.
- Switching floors off never hurt the exam.
- Floors 15-18 are the strongest location, cutting harm from 69 to 31, far more than a same-strength random group (54).
- Floors 9-11 also clearly help.
- A few of our arrows carry most of each floor's effect, except on floors 0-2.
- Floors 0-2 are not special. Random groups come close, so harm does not appear to start at the bottom.
- Groups close only about 1 point more than their floors added together (11.14 against 10.16, and 15.05 against 14.00). So "floors work together" is only weakly supported.
- **The fast score badly understates real harm.** Floors 15-18 close 15% on the fast score, but about 70% of the real gap (38 of 54 points). Use the fast score for ranking and real answers for size.

**Why one floor does so little.** There are 36 teams pushing the piano, and when one leaves the other 35 keep pushing.

**Cautions:** one model, 100 questions. Batching alone flips 9 of 100 labels. After correcting for 14 tests, only floors 15-18 beat random on real answers. Adding one floor to Normal Aya recreates about 6.4% whichever floor it is, so that number says little.

## 12. Every known mistake and weakness

| # | What went wrong or is weak | Does it change the conclusion? |
|---|---|---|
| 1 | 232 of 500 ordinary test questions were also pick questions | No, the clean table holds, but quote the clean table |
| 2 | "Never trained on" is inferred | Probably not, but it is an inference |
| 3 | Step 3's random check reversed arrows. It should have moved a slider from 5 to 0. A rescale pushed it past zero, for example to -7.5 | No, Step 4 redid it |
| 4 | Our arrows tie with biggest | Yes, for the claim that blame beats size |
| 5 | Adding our arrows alone gives about half the harm | Needed, and only partly enough |
| 6 | AI reviewers probably undercount | Counts may be slightly low |
| 7 | A speed shortcut's limit was raised from 0.002 to 0.005 after a failure | No, a slower recheck agrees |
| 8 | The first repaired add-on files were wrong (-0.944, expected -0.588). Exact cause not found | No, rebuilt and verified |
| 9 | Whether harm returns after more training is untested. A July 2026 paper reports relapse with similar edits. The team chose not to run this | Unknown |
| 10 | Removing 256 arrows is 38% of the add-on by size. It works as a repair, but it does not show where harm starts | Limits the claim |
| 11 | The add-on holds mostly harm, so deleting all of it also works (Normal Aya gives 90). A judge may ask why bother | Answer: it shows which changes matter |
| 12 | Models 2 and 3 only had broken and repaired versions reviewed | Limits the 3-model claim for biggest and controls |
| 13 | Medical advice, English only | Limits scope |
| 14 | Shreya's repair favours attention blocks, and ours favours MLP. Unexplained | Open |
| 15 | A judge could say an arrow is not a circuit or neuron | Answer: each arrow is one direction of change in the add-on |

## 13. The Lexsi tools

**The official order** on the hackathon page is CuratorKIT, SafeTune, CircuitKIT, AuditKIT. It is a product path, not a required order. SafeTune's README calls itself "a library, not a pipeline".

| Tool | What it does | How we used it | Result |
|---|---|---|---|
| CuratorKIT | Prepares and checks data | Checked test questions for near-duplicates of training | Done |
| SafeTune | A library of published safety methods | Judge code, a refuse-or-comply finder, a neuron finder | Nothing useful for our harm |
| CircuitKIT | Finds and acts on the parts doing a job | EAP-IG on bad and Normal Aya | Same parts in both |
| AuditKIT | Records evaluation runs | One fingerprinted run per version | Fingerprints saved, but the comparison numbers were not. Our own paired test supplied those |

**CircuitKIT against SafeTune**

| | CircuitKIT | SafeTune |
|---|---|---|
| In one line | Finds the exact parts doing a job, then acts on them | Many published safety methods, you pick one |
| Fix during training | Yes, targeted fine-tuning | Yes, "Harden" |
| Fix a finished model without training | Yes, prune or edit | Yes, "Recover" |
| Fix while the model answers | Yes, steer | Yes, "Steer" |
| Finding | Its main job | One section, "Diagnose" |

**Why they failed for us.**

- **CircuitKIT:** we asked which next word wins, 8 words after the answers split. Good and bad Aya both continue a started answer well, so it found general machinery. We also ran a version 94 commits old, on one task with 96 pairs.
- **SafeTune EAP:** it looks at the choice to refuse or comply. Our model almost never refuses (0 to 4 of 382).
- **SafeTune neuron finder:** an earlier note said it looked at the wrong spot. That was wrong. It scored the real 11,008 neurons (saved unit numbers reach 11,002). We compared risky questions with ordinary questions, which is not our harm.
- **Repair tools:** we never tried them.
- **A likely CircuitKIT bug, effect not yet measured:** for MLP parts, it multiplies the output change by the gradient at the input, where the reference code uses the output. This is in the version we ran and the newest one, and 7 of its top 8 parts were MLPs. So the earlier CircuitKIT result is not settled.
- **A real limit:** CircuitKIT compares two inputs inside one model. It has no built-in way to ask what changed between two models, so it may never find our floors from the same data.
- **DLBacktrace** does not list Cohere models as supported, so it is not worth using here.

**Bugs we found in the tools:** SafeTune's WildGuard reader, SafeTune's HarmBench prompt, CuratorKIT's `prompt_only` dedup, AuditKIT's paired bootstrap (p = 0.51 where about 0.001 was right), CircuitKIT ignoring `n_examples` on custom data.

**The natural order for a targeted fix:** find the part with CircuitKIT, fix only that part with SafeTune, check with AuditKIT. CircuitKIT's output includes `safety_units` and `layer_suggestions`, which SafeTune can read.

**The best-fitting SafeTune repair is C-Delta-Theta.** It adds `strength x (good model - bad model)` to the layers you name. With Normal Aya as good, bad Aya as bad, and strength 1, it is the same as switching those floors off. So that setup should reproduce 69 to 31, which tests the tool and adds no evidence. The real test uses the good twin and Shreya's separately trained bad model.

**Part B, the plan in progress:** the tools can probably apply a floor-level fix, but probably cannot find the floors by themselves. Run them as an honest independent check, settings fixed before results, reporting agreement or disagreement. We will not change settings until they match our floors. It will not produce the same 256 arrows, because no tool works at arrow level.

## 14. Problem 10, briefly

Problem 10 asks to change behaviour only at the parts a circuit points to, and to "verify nothing else broke". It comes later. Research suggests a stronger test is one add-on doing two jobs, a useful skill plus the bad advice, then removing only the harm arrows and showing the skill survives. Today's add-on holds mostly harm, so deleting it all also works.

## 15. What is left for problem 9

| Step | Who | Status |
|---|---|---|
| Newest CircuitKIT with a harm-based task, plus LRP | Shreya, Part B | Planned |
| Newest SafeTune finders and repair | Vishnu, Part B | Planned |
| Random test matched on number and strength | Us | Planned |
| Attention or MLP, with Shreya | Team | Optional |
| Write-up with limits first, public hosting | All | Not started |

Deadline: 14 November. Demo Day: 22 November.

## 16. Questions a judge may ask

**What is an arrow?** One of 8,064 small pieces of the add-on's change. Each patch splits into 32, and they add back exactly.

**How do you know it isn't chance?** Our arrows beat 400 random sets on all 3 models, with the rules written first. Real harmful answers fell from 623 to 85 of 1,000.

**Are the arrows special, or just big?** Mostly big. About 60% are also in the strongest 256, and on unused questions it is a tie, 60 against 58.

**Did the fix hurt the model?** No sign. The exam went from 463 to 493 on model 1.

**Did you test on training questions?** Mostly, because the researchers trained on about 90% of them. On questions never trained on and never used to pick arrows, harm fell from 27 to 6 of 41 on model 1.

**Why not use CircuitKIT?** We tried. It found the same parts in both models, because our task was next-word prediction. We are rerunning it with a harm-based task on the newest version.

**Where does the harm live?** It is spread out, strongest at floors 15 to 18, then 9 to 11, and mostly in MLP workers.

**Why is deleting the whole add-on not enough?** It would work here. Our result shows which changes matter, and that 256 arrows are enough.

**What did you get wrong?** Our first random check reversed arrows. We called questions unseen when the model had trained on them. 232 test questions were also pick questions. All were found and corrected.

**Does the harm come back?** We did not test that.
