# Plan: ship Iba Lens before 5:00 PM Lagos (16:00 UTC). Tick as we go.
Judging (hackathon/evaluation.md): Impact · Technical/AI · Innovation · Execution/Completeness · Presentation. No video or public repo = not judged.
Order is by "what loses the prize if missing". Don't start a later block until earlier blocks are done, unless noted.

## A. Must exist or we are not judged (target done by 15:00 Lagos)
- [ ] A1 Cut a minimum-viable video: screen-record queue -> LAB-0416 detail -> confirm -> Analyse tab -> metrics (<=2:30, public YouTube, captions)
- [ ] A2 Open with the scientist + queue, not "a CNN detects malaria" (originality risk)
- [ ] A3 Say numbers in plain words: "of 100 infected cells it catches 95", "share of red cells infected". One AUC mention
- [ ] A4 Make repo public (`tests/` committed; `audit/scratch/` excluded; check no secrets; data/ is gitignored)
- [ ] A5 README: replace placeholder links, tick checklist
- [ ] A6 Devpost submitted with 30 min buffer (target 16:30 Lagos)

## B. Safety/honesty fixes (about 40 min total; do right after A1-A4 are recorded, before A6)
- [ ] B1 Min-cell guard: n < 20 (or foreground fraction implausible) -> band "unclear / repeat smear", never "likely negative". Python `summarise` + mobile `summarise`; flip xfail `test_zero_cells_must_not_be_called_negative`
- [ ] B2 Gradio `app.py`: show an error/warning when 0 or very few cells found (mobile already errors on 0)
- [ ] B3 Fix README claims: speed (1.7 s / ~10 s with heatmaps), "uninfected slides unclear" explanation (k_neg vs ~7 unclear cells), "no uninfected slide called positive" -> "0 of 40; expect ~8% in larger samples"
- [ ] B4 Add strict-"likely positive" table + patient-level CI (AUC 0.987-0.996, sens 92.1-97.4) to README "How well it works"
- [ ] B5 requirements.txt: add onnx, onnxruntime, onnxscript, pytest
- [ ] B6 Submission copy: say heatmap is Grad-CAM (web) / occlusion (phone)
- [ ] B7 Limitations: add "fails on overlapping cells / dark backgrounds; sensitive to noise and heavy blur"

## C. Demo hardening (only if A+B done and >60 min left)
- [ ] C1 Segmentation: local (adaptive) threshold or background-normalise; try watershed to split touching cells; re-run `tests/test_adversarial_known_bugs.py` (xfails should turn XPASS -> remove marks)
- [ ] C2 Re-run `audit/exp_field.py`; target: dark-bg and 0.55-vignette fields keep >=95 cells
- [ ] C3 Re-pick `k_neg` using 0% slides too; re-run `evaluate_slides.py`; update tables
- [ ] C4 Add a "cells found: N" chip + warning if N differs a lot from expected

## D. Do NOT do before submitting
- No more mobile UI work, no retraining, no new features, no HF Space unless everything above is ticked.

## E. After submission
- [ ] E1 Share gallery link for Audience Favorite voting (10-11 Oct)
- [ ] E2 Keep `audit/findings.md` as the honest-limits appendix for Q&A
