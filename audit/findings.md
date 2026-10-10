# Iba Lens independent audit (2026-10-10)

Everything below was re-run from code and data in this repo; README/hackathon claims were ignored unless re-verified.
No existing file was modified. New paths only: `tests/`, `audit/`. Nothing committed or pushed.
Python = `.venv` (pytest was pip-installed into `.venv`, which is gitignored).

## 0. Verdict in one paragraph
The cell classifier numbers are real and reproducible, and the patient split is clean. The weak points are
(1) the segmenter, which silently returns **0 or a few cells and then reports "likely negative"** on harder fields
(overlapping cells, darker background), (2) the slide-level story, whose headline "0/40 uninfected slides called positive"
and "unclear for all 0% slides" both come from tiny-sample / mis-set `k_neg` behaviour, (3) prevalence: cell-level PPV 98.7% is at 45% cell prevalence
and collapses at realistic parasitemia, and (4) every slide result is on synthetic fields built from the same cells the classifier scores.

## 1. Reproduction of claimed numbers (all verified)
| Claim | Re-computed | Script |
|---|---|---|
| Split 140/20/40 patients, zero overlap | exactly that; 0 shared patients; 0 exact-duplicate images across splits; 0 label conflicts | `tests/test_data_and_metrics.py` |
| AUC 0.9929 | 0.992863 (stored `test_scores.npz` identical to fresh inference, max diff 0.0) | same |
| sens 95.3 / spec 97.6 @0.5 | 95.346 / 97.555 | same |
| Triage bands (2091 pos/27 FP, 364 unc, 2541 neg/43 FN) | identical | same |
| t_low 0.142 < t_high 0.895 | yes; mobile `thresholds.json` identical | same |
| ONNX == torch | max abs diff <1e-4 on 256 cells | same |
| Slide table (results/slide_metrics.json) | re-run of the test half of `evaluate_slides.py`: identical counts | `audit/exp_slide.py` |
| mobile `npm run build` | passes. `npm run lint` (oxlint): 0 errors, 9 warnings (refs-in-render, components created during render, unused param in sw.js) | shell |
| TypeScript | `tsconfig.app.json` has **no `"strict": true`**, but `tsc --strict` also passes with 0 errors | shell |
| `import app` (Gradio) | imports cleanly | shell |

Near-duplicate leakage across patients: nearest-neighbour cosine (24x24 grayscale) test->train: max 0.988, none >0.99
(`audit/exp_neardup.py`). Only exact-hash and this coarse check were done, no perceptual-hash/feature-space dedup.
Caveat that the README does not mention: 11 of the 40 test patients are healthy (0 parasitized cells), so "40 patients" gives per-patient sensitivity for only 29 (22 with >=10 positives).

## 2. Tests added (`tests/`), run: `.venv/bin/python -m pytest tests` -> **29 passed, 3 xfailed** (118 s; the slowness is a session fixture running 5k cells on CPU torch)
- `test_data_and_metrics.py` (8): no patient overlap, no duplicate cells across splits, filenames parse, thresholds sane, stored scores == fresh inference, headline metrics + triage counts, thresholds.json == metrics.json == mobile copy, ONNX parity on 256 cells.
- `test_smearscan.py` (16): `summarise()` edge cases (0 cells, all positive, all negative, 7 band-boundary cases, inclusive/exclusive threshold edges, clamp at 0), `segment()` counts on rendered fields (30/60/100 -> exact), empty/black/speck fields, `reference.json` vs Python SmearScan on all 8 demo slides (counts, bands, flagged counts, bboxes, max prob diff <1e-4), mobile demo PNGs == assets PNGs, demo truth vs prediction.
- `test_adversarial_known_bugs.py` (3 xfail strict): overlapping cells found, 0 cells must not be "likely negative", vignetting keeps cells. They flip to XPASS (=fail) once the bug is fixed.
- JS side: vitest not added (would touch `package.json`/lock). Instead `audit/js_segment_parity.mjs` runs the mobile `segment.ts` directly in node 24 (type stripping) and `audit/js_crop_prob_check.py` pushes those JS crops through the ONNX model.

## 3. Mobile <-> Python parity (`audit/js_segment_parity.mjs`, `audit/js_crop_prob_check.py`)
- Cell counts equal on 8/8 slides. Bboxes identical on 798/800 cells; LAB-0412 has 2 cells off by 1 px.
- Because of those 2 crops, max per-cell prob diff on LAB-0412 is **0.0078** (mobile/README claims 0.0007 max; true for the other 7 slides, <5e-5).
  Bands and flagged counts identical on all 8. (WASM numerics not tested - no browser run.)

## 4. Adversarial results

### 4a. Slide level, strict "likely positive" only (`audit/exp_slide.py`, `audit/exp_slide.json`; same seeds as evaluate_slides)
| True | n | likely pos | unclear | likely neg | strict-pos rate | mean flagged-pos | mean flagged-unclear |
|---|---|---|---|---|---|---|---|
| 0% | 40 | 0 | 40 | **0** | 0% | 0.8 | 7.0 |
| 1% | 40 | 6 | 31 | 3 | 15% | 1.55 | 7.1 |
| 2% | 40 | 14 | 24 | 2 | 35% | 2.35 | 7.1 |
| 5% | 40 | 38 | 2 | 0 | 95% | 5.2 | 6.9 |
| 10% | 40 | 40 | 0 | 0 | 100% | 9.7 | 6.6 |
| 20% | 40 | 40 | 0 | 0 | 100% | 18.4 | 6.8 |
- Strict slide sensitivity ("likely positive") over all infected slides 69%; **25% at <=2% parasitemia** (the clinically hard range). The README's 97.5% is the weaker "not negative" metric (93.75% for <=2%).
- Why 0% slides are never "likely negative": `k_neg=2` (slide is negative only if flagged-pos + flagged-unclear <= 2). An uninfected 100-cell field has on average 7 *unclear* + 0.8 *positive* cells (uninfected test cells: 1.0% >= t_high, 7.8% (215/2740) in the unclear band overall; 9.8% for uninfected cells of infected patients, 2.7% for healthy patients). The min over 40 slides of pos+unc is 3. `k_neg` was calibrated on **1%** validation slides (to protect sensitivity), never on 0% slides, so the "likely negative" band is unreachable for a normal 100-cell field. The README's explanation ("a 1%-slide's single infected cell looks like a false positive") is wrong for 0% slides: they contain no infected cell.
- Monte Carlo (2000 draws, `exp_slide.json`): 0% 100-cell slides from pooled test negatives -> **8.05% "likely positive"** (false alarms), 0.45% likely negative. So "0/40 uninfected slides called positive" is small-sample luck; true rate ~8% (CI on 40 slides with 0 events is up to 7.2%).
- The demo's three 0% slides are likely negative only because they are 60-cell fields from **healthy patients** (sim: 62% negative); with 100 cells from healthy patients only 25%.
- Quirk: 2 cells at p=0.99 (confident parasites) -> slide "likely negative" (k_neg counts positives too). At 1% a 100-cell field has 1 parasite and 3/40 were called negative (test `test_band_logic_boundaries`).

### 4b. Harder synthetic fields (`audit/exp_field.py`, `audit/exp_field.json`; 12 slides each at 0/5/10% from test patients, 100 true cells)
Cells found (true 100) / mean parasitemia estimate at 10% true / band outcome at 10%:
| Condition | found | est @10% | notes |
|---|---|---|---|
| baseline | 100 | 9.2 | 12/12 likely pos |
| touching (spacing 100, 4 px gap) | 96 | 8.9 | ok |
| **overlapping (spacing 86 or 76)** | **0** | **0.0** | **all 36 slides "likely negative"** (merged blob > 40000 px area is discarded) |
| vignetting 0.55 / 0.35 | 83 / 68 | 9.2 / 8.9 | loses 17-32% of cells (global Otsu) |
| noise sd8 / sd20 | 100 / 99 | 7.7 / **3.2** | sd20: 5% slides est 1.5 |
| JPEG q30 / q10 | 100 | 8.6 / 8.8 | fine |
| blur sigma 2 / **4** | 100 | 8.7 / **2.9** | sigma 4: 1/12 neg, 5/12 unclear at 10% |
| brightness 0.75 / 1.1, purple / pink stain, light bg, white bg | 100 | 8.8-9.4 | fine |
| **darker background (190,170,180)** | **0-13 (median 0)** | **1.7** | **10/12 slides at 10% parasitemia and 10/12 at 5% called "likely negative"** |
| cell scale 40-200 px | 99-100 | 9.6 | fine (crops are resized to 96, scale-invariant) |
| rotated 30 / 90 deg | 100 | 8.8 / 9.4 | fine |
| WBC x3 / platelets x10 / both | 96 / 99 / 95 | 9.1 / 8.9 / 8.8 | few false positives from these blobs (<=2 per field), but WBC blobs break neighbouring cells |
| combo vignette+noise8+JPEG30+blur1 | 80 | 10.6 | cells lost, estimate inflated by denominator loss |
There is **no sanity check on cell count** (neither Gradio nor mobile; mobile only errors on exactly 0). A partial-segmentation or dark-bg field returns "likely negative" with confidence. Real smears have overlapping cells, uneven illumination and varying backgrounds, so this is the biggest unmitigated technical risk.

### 4c. Classifier alone under perturbation (`audit/exp_robust.py`, test cells, bg re-blacked; clean sens/spec 95.3/97.6)
| Perturbation | AUC | sens@0.5 | spec@0.5 |
|---|---|---|---|
| blur r1 / r2 / r3 | .993 / .992 / .986 | 96.2 / 94.0 / **80.0** | 96.6 / 97.6 / 99.3 |
| brightness .7 / 1.3 | .993 / .992 | 95.6 / 94.6 | 96.6 / 97.9 |
| contrast .6 / 1.5 | .991 / .992 | 92.2 / 96.2 | 98.5 / 96.8 |
| JPEG q30 / q10 | .990 / .967 | 94.8 / 95.8 | 95.6 / **75.4** |
| hue +-10 / +25 deg | .992 / .992 | 94.8-95.6 / 93.6 | 97.0-97.6 / 97.9 |
| saturation .6 / 1.5 | .991 / .992 | 92.2 / 96.3 | 98.3 / 96.4 |
| gaussian noise sd10 / sd25 | .988 / .956 | **87.5 / 39.5** | 99.4 / 99.9 |
| 64px / 48px up to 96 | .993 | 96.2 / 95.8 | 96.8 |
| grayscale | .919 | **9.5** | 100 |
| rot 90 / rot 37 | .993 / .991 | 95.7 / 94.3 | 97.0 / 98.0 |
Robust to colour/brightness/rotation/resolution; fragile to noise (sd10 already -8 pts sens), strong blur, heavy JPEG (spec), and colour removal.

### 4d. Per-patient + cluster bootstrap (`audit/exp_cell_stats.py`)
- Per-patient sensitivity @0.5 (22 patients with >=10 parasitized cells): **min 45.9% (C49), median 96.6%, max 100%**; 4 patients <90%, 2 <80% (C49 46%, C125 77%). C49: 11 of 37 parasitized cells fall *below t_low* (called likely negative).
- Cluster bootstrap over 40 test patients (B=1000), 95% CI: **AUC 0.987-0.996, sens 92.1-97.4%, spec 96.7-98.3%, PPV@t_high 96.8-99.4%.** The README gives point estimates only.

### 4e. Calibration
- ECE 0.0103, Brier 0.0273 (good overall); reliability is off in the 0.1-0.5 and 0.9-1.0 bins (e.g. bin 0.9-1.0 mean p 0.995, true 98.8%; bin 0.2-0.3: p .25 vs 17%).
- "likely positive" PPV 98.7% overall, but per patient: 35 patients have any positive call, **18 of them contain >=1 false positive; median per-patient PPV 99.2%, min 0%, 13 patients <90%** (e.g. healthy patients C202, C206, C214, C217 whose only positive calls are false).
- **Prevalence dependence**: using test sens/spec at t_high (91.5/99.0%), PPV of a "likely positive" cell is 98.7% at 45% prevalence (test set), but **83% at 5%, 65% at 2%, 48% at 1%, 32% at 0.5%** cell prevalence. NPV at t_low stays >=99.8% for <=5%. The 98.7% headline is a property of this 45%-parasitized test set, not of a smear (real parasitemia is mostly <5%).

### 4f. Threshold stability
- Val-chosen t_low 0.142 / t_high 0.895 (20 val patients). Patient bootstrap of the val set: t_low 5-95% = **0.019-0.291**, t_high **0.80-0.988**. Re-selecting by the same rule on random 20-patient subsets of the *test* patients gives t_low median 0.012 (p5-p95 0.004-0.041) and t_high median 0.967 -> the same rule gives wildly different thresholds depending on patients.
- The "99.5% sensitivity at t_low" target does not transfer: realised sens at t_low on 20-patient test subsets = 96.7-99.3% (median 98.1%; overall 98.1%). Spec at t_high realised 98.7-99.3% (target 99.5%).

### 4g. Baseline
Same split, features = per-channel stats/percentiles/16-bin histograms of non-black pixels. Logistic regression AUC 0.972 (sens 93.1 / spec 93.5); HistGradientBoosting AUC **0.981** (93.9 / 94.2). Model: AUC 0.993, 95.3 / 97.6. So the CNN beats a trivial baseline but the dataset is easy (a colour-histogram GBM reaches 0.98); README cites no baseline.

### 4h. Speed / size
ONNX (CPU, python onnxruntime, Apple silicon): 1 thread 3.1 ms/cell, **335 ms per 100 cells**; 4 threads 1.4 ms/cell, 83 ms per 100. Full Python pipeline per 100-cell field: segment 0.09 s, no heatmaps 1.7 s, with Grad-CAM for all flagged cells **10.2 s** (README: 1.8 s / ~6 s). ONNX 8.9 MB, .pt 9.1 MB. Mobile segment() in node: 135-277 ms per field (JS in browser/WKWebView not measured).

### 4i. Heatmaps (`audit/exp_cam.py`; 100 random test cells called "likely positive")
Proxy for the parasite = darkest 5% green-channel pixels inside the eroded cell mask (dilated 6 px; covers 22% of cell area) - a proxy, not annotation (no parasite masks exist in the dataset). Pointing-game hit rate: **Grad-CAM 88%, occlusion (what the mobile app shows) 81%, random in-cell point 22%, cell centroid 18%**. Grad-CAM vs occlusion Pearson 0.80. Heatmaps do concentrate on the stained spot well above chance; 7-12% of peaks fall outside the cell mask. Only a proxy-based sanity check; not evidence the scientist can "check the parasite".

## 5. README/doc mismatches (not edited)
1. README speed "1.8 s / about 6 s with heatmaps" -> measured 1.7 s / 10.2 s.
2. README "uninfected 100-cell slides come out unclear" explanation (one infected cell looks like a FP) is incorrect: they are unclear because ~7 unclear cells/100 cells exceed `k_neg=2`.
3. README "no uninfected slide was called positive" and submission "0 of 40": true for those 40 slides; Monte-Carlo rate ~8%.
4. mobile/README "max per-cell prob difference 0.0007": 0.0078 on LAB-0412 when running the TS segmenter (2 cells 1 px off).
5. `requirements.txt` lacks `onnx`, `onnxruntime`, `onnxscript` (needed by `export_onnx.py`), no pytest; `dump_reference.py` / `export_onnx.py` missing from the README pipeline table; mobile README is the Vite template + a few lines.
6. Submission copy still says "Gradio UI" and "Grad-CAM heatmap for each flagged cell" while the mobile app uses occlusion sensitivity.
7. README "Seeds are fixed ... reproduce every number": `slide_k_pos/k_neg/cell_fp_rate` are written into `models/thresholds.json` by `evaluate_slides.py` (not calibrate.py), and mobile's copy is made by `export_onnx.py`; running the pipeline out of order leaves them stale (tests assert the copies are in sync today).
8. "Signed result" in the README mermaid diagram: the app has a "Scientist signs" step in UI only; nothing is signed/persisted.
9. Placeholder links still in README (demo, video) and unchecked checklist items.
10. README "40 held-out patients" implies 40 evaluable; 11 are healthy.

## 6. Bugs found (code)
- B1 `segment()`: touching/overlapping cells merge into a component > `max_area=40000` and are silently dropped (spacing <=90 px: 0 cells). (Python and TS port identical.)
- B2 `summarise()`: 0 cells -> `pos+unc=0 <= k_neg` -> "likely negative" with max_prob 0. Mobile guards exactly 0 cells with an error; Gradio does not; partial loss is unguarded in both.
- B3 `segment()`: global Otsu on grey; darker/different background (190,170,180) -> 0-13 cells found; 83% of 5-10% parasitemia slides called "likely negative". Vignetting loses 17-32% of cells.
- B4 `k_neg`/`k_pos` calibration: `k_neg` set from 1% slides only, so "likely negative" is unreachable for realistic 100-cell uninfected fields; sensitivity trade-off hidden by "not negative" metric.
- B5 `render_field(gap<0)` crashes (IndexError) - harmless dev-only.
- B6 Demo and metrics disagree: evaluate_slides' 0% slides pool negatives from all test patients (0/40 likely negative), the demo's 0% slides are 60 cells from healthy patients (3/3 likely negative).
- B7 tsconfig lacks `strict` (passes anyway).

## 7. Top 10 risks for judging (most damaging first)
1. Segmenter collapses silently on overlap / dark background / vignetting -> "likely negative" at 10% parasitemia (a judge uploading any real smear photo will hit this).
2. All slide-level evidence is synthetic (NIH crops pasted on a flat colour) - the README admits it, but demo + metrics make it look like a full-pipeline result.
3. "Likely negative" band effectively dead for real 100-cell fields; demo's three negatives are cherry-picked healthy 60-cell slides.
4. Prevalence: cell PPV 98.7% is at 45% prevalence; at 1-2% it is 48-65%; slide "0 of 40 false positives" hides an ~8% false-alarm rate.
5. Strict slide sensitivity at <=2% parasitemia is 25%; headline 97.5% uses the "not negative" definition.
6. Single dataset/site (Bangladesh phone-on-microscope); no external validation; CI [92.1, 97.4] on sensitivity and one patient at 46%.
7. Threshold stability: val set of 20 patients; t_low/t_high move by 10x under patient resampling; 99.5% sens target realised 98.1%.
8. Fragility to sensor noise (sd10 -> sens 87.5%) and blur (r3 -> 80%), grayscale -> 9.5%.
9. Docs inaccuracies (speed 10 s vs 6 s, mobile parity number, wrong explanation of the unclear 0% slides) and requirements.txt not sufficient for ONNX export.
10. Heatmap "evidence" validated only against a proxy; the mobile app shows occlusion not Grad-CAM, while submission copy says Grad-CAM; no tests/CI existed in the repo before this audit.

## 8. Files
`audit/exp_cell_stats.py/.json` (d,e,f,g,h) · `exp_robust.py/.json` (c) · `exp_slide.py/.json` (a) · `exp_field.py/.json` (b) · `exp_cam.py/.json` (i) · `exp_neardup.py` · `js_segment_parity.mjs` · `js_crop_prob_check.py/.json` · `common.py` · `scratch/` (raw outputs, rgba dumps of demo slides; delete if unwanted).
Run scripts from `audit/` with `../.venv/bin/python` (`exp_slide.py` imports `evaluate_slides` through `common`'s sys.path).
