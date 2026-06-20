# WM-811K Wafer Defect Classification

This project uses a convolutional neural network to classify semiconductor wafer maps from the WM-811K dataset into nine categories: eight distinct defect patterns plus a clean baseline. It's a portfolio project, so the focus is on building a principled, well-reasoned ML pipeline rather than chasing benchmark numbers. The dataset comes with some genuinely interesting challenges: severe class imbalance, a predefined evaluation split, and maps of varying sizes across manufacturing lots.

---

## The Dataset

The [WM-811K dataset](https://doi.org/10.1109/TSMC.2018.2841818) (MIR-WM811K) contains 811,457 wafer maps collected from real semiconductor manufacturing lines. You can download it from [Kaggle](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map). Each wafer map is a small grid where each cell encodes the pass/fail status of an individual die. Maps vary in size across manufacturing lots; the pipeline resizes everything to 64×64 before training. Of the 811k records, only 172,950 are labeled, and the class distribution is heavily skewed:

| Failure type | Count |
|---|---|
| none (clean) | 147,431 |
| Edge-Ring | 9,680 |
| Edge-Loc | 5,189 |
| Center | 4,294 |
| Loc | 3,593 |
| Scratch | 1,193 |
| Random | 866 |
| Donut | 555 |
| Near-full | 149 |

The dataset includes a predefined `trainTestLabel` split. This project respects that split and keeps the test set completely untouched until final evaluation.

---

## What's in this repo

```
notebooks/
  00-wm-811k-EDA.ipynb      Exploratory analysis: distributions, sample maps, shape checks
  01-wm-811k-cnn.ipynb      Full CNN pipeline: training, evaluation, and model comparison

figures/                    Saved plots (training curves, confusion matrix, comparisons, etc.)
tables/                     Saved metrics as Markdown tables

data/                       Dataset file, gitignored and not included in the repo
learning_log.md             A detailed decision log explaining the reasoning behind each design choice, also gitignored
```

---

## Pipeline highlights

**No data leakage**
The predefined train/test split is used exactly as provided. Normalisation statistics are computed from the training split only, then applied consistently to validation and test data. The test set is only ever used once, for final evaluation.

**Stratified validation split**
20% of the training pool is held out for validation using stratified sampling. This ensures every class, including rare ones like Near-full, is proportionally represented on both sides of the split.

**WeightedRandomSampler**
With 147k "none" samples alongside just 149 "Near-full" samples, a model can reach 85% accuracy by predicting "none" every time. Rather than reweighting the loss, each training batch is constructed with balanced class representation using PyTorch's `WeightedRandomSampler`. Each sample is drawn with probability proportional to the inverse of its class count, so rare classes like Scratch and Near-full appear as often per batch as the dominant "none" class. The loss function uses plain `CrossEntropyLoss()` — the sampler handles balance at the data level, which is more stable than combining extreme loss weights with large batches.

**Label-preserving augmentation**
Wafer maps have rotational symmetry, so horizontal flips, vertical flips, and 180-degree rotations all produce valid maps of the same defect type. 90° and 270° rotations are excluded because they transpose height and width on non-square maps, producing invalid shapes. Translation augmentation was explored but removed — edge defects (Edge-Loc, Edge-Ring) are defined by where they sit on the wafer, so shifting them away from the edge destroys the label.

**CBAM attention (second experiment)**
The second model adds CBAM (Convolutional Block Attention Module) after each conv block. Channel attention scores each feature channel 0–1; spatial attention produces a 2D heatmap highlighting where to focus. This is particularly useful for Edge-Loc, Edge-Ring, and Loc — three classes that share similar textures but differ by where on the wafer the defect sits.

**Early stopping on val loss, checkpoint on best Macro F1**
Accuracy is a misleading metric when one class makes up 85% of the data. `ReduceLROnPlateau` and `EarlyStopping` monitor validation loss as the convergence signal — it's stable and doesn't overfit to the tiny validation set. The best model checkpoint is saved separately whenever validation Macro F1 improves, so model selection and convergence detection are decoupled.

---

## Results

Full per-class breakdowns for each run are saved in [`tables/`](tables/). The primary metric is **Macro F1** — it weights every class equally and is far more informative than accuracy for this imbalanced dataset.

| Run | Accuracy | Macro F1 | Weighted F1 | Epochs |
|-----|----------|----------|-------------|--------|
| CNN + WeightedRandomSampler | 0.899 | 0.561 | 0.915 | 53 |
| CNN + CBAM | 0.947 | 0.656 | 0.949 | 69 |

Per-class breakdowns: [`tables/classification_report_baseline.md`](tables/classification_report_baseline.md) and [`tables/classification_report_augmented.md`](tables/classification_report_augmented.md).

### Per-class F1 breakdown (final model: CNN + CBAM)

| Class | F1 | Precision | Recall | Notes |
|---|---|---|---|---|
| none | 0.977 | 0.983 | 0.971 | Dominant class, well-learned |
| Near-full | 0.890 | 0.885 | 0.895 | Small support (95 test), high F1 |
| Random | 0.727 | 0.784 | 0.677 | Distinct pattern, learned well |
| Donut | 0.661 | 0.592 | 0.747 | Rare but visually distinctive |
| Edge-Ring | 0.695 | 0.689 | 0.701 | Plateaued — both models identical |
| Edge-Loc | 0.584 | 0.487 | 0.729 | Improved +0.17 with CBAM |
| Center | 0.549 | 0.474 | 0.654 | Moderate — confused with Loc |
| Loc | 0.524 | 0.625 | 0.451 | Improved +0.26 with CBAM |
| **Scratch** | **0.298** | **0.269** | **0.335** | **Hardest class — too few examples** |

---

## Experiment Log

A running record of every training run, what changed, and what the results showed.

### Run 1 — Baseline CNN (batch 64, shape-filtered dataset)
Shape filter restricted training to maps of a single size (~18k of 172k rows). Macro F1: ~0.556. Established the pipeline but left 89% of the data unused.

### Run 2 — Translation augmentation (batch 64, shape-filtered)
Added `T.RandomAffine` translation to the training transform. Hurt edge classes (Edge-Loc, Edge-Ring) because those defect types are defined by *where* the defect sits on the wafer — translation moves the pattern away from its characteristic position. Macro F1: ~0.542. Removed translation from all future experiments.

### Fix — Full dataset unlocked (T.Resize in transform pipeline)
Removed the shape filter and added `T.Resize(64×64, NEAREST)` as the first transform. All 172k labeled rows now flow through training. This was the single biggest pipeline improvement.

### Run 3 — Baseline CNN (batch 512, full dataset)
First GPU run with the full dataset. 5× more minority-class training examples. Macro F1: **0.593** — best result so far. Scratch F1 remained near zero (~0.04) despite class weighting, likely because the thin diagonal line pattern is too fine to survive 3× MaxPool at 32×32.

### Run 4 — Gaussian noise augmentation (batch 512, full dataset)
Added `img + randn * 0.05` after normalisation. Macro F1: 0.543. Noise destroyed fine spatial features (especially Scratch and Loc patterns) without providing any useful regularisation. The model trained longer (90 epochs) because val F1 kept marginally ticking up while val loss was increasing — identified as an early-stopping bug (fixed in run 6+).

### Run 5 — Dropout 0.6 + weight decay 1e-4 (batch 512, full dataset)
Added `weight_decay=1e-4` to Adam and increased Dropout from 0.5 → 0.6 to address the train/val loss gap. Over-regularised the model. Macro F1: 0.542 (baseline) / 0.521 (with Gaussian noise). Worse than Run 3 in both cases.

### Run 6 — 64×64 input + batch 4096 + val-loss early stopping
Increased input resolution and batch size. Label smoothing was also tried but removed — it interacts badly with extreme class weights, creating unstable gradients and stalling the model at ~14% train accuracy. Without smoothing: Macro F1 0.546. Edge-Ring improved dramatically (+0.19) from the higher resolution, but Loc (−0.15) and Edge-Loc (−0.14) regressed, likely because at higher resolution both patterns look more similar to each other, making the model conflate them.

Scratch remained near zero (F1 0.048) — precision was the problem (0.026), meaning the model over-predicted Scratch on non-Scratch samples. Diagnosis: with batch 4096 and ~400 Scratch training examples, each batch statistically contains only 2–3 Scratch samples even with class weights. The gradient signal is too sparse for the model to discriminate Scratch reliably.

### Run 7 — CNN + WeightedRandomSampler
Replaced class-weighted loss with `WeightedRandomSampler`. Each batch now contains roughly equal representation of all 9 classes — Scratch and Near-full appear ~40–50 times per batch instead of 2–3. The loss function uses plain `CrossEntropyLoss()` with no class weights, since the sampler handles balance at the data level. Macro F1: **0.561**. Scratch improved from 0.048 → 0.055 — marginal.

### Run 8 — CNN + CBAM
Same WeightedRandomSampler as Run 7. Adds CBAM attention to each conv block. Macro F1: **0.656** — the best result in this project. The gains were concentrated exactly where the hypothesis predicted: Loc (+0.26), Scratch (+0.24), Edge-Loc (+0.17). Spatial attention gave the model an explicit mechanism to distinguish defects by position, not just texture.

---

## What we learned

**WeightedRandomSampler helped but didn't solve the root problem.**
The sampler ensures every class appears frequently in each batch. That helped batch-sparse classes like Loc and Edge-Loc considerably, and CBAM's spatial attention amplified those gains further. But Scratch remains the hardest class at F1 0.298, and the reason is a data scarcity problem, not a model capacity or sampling problem. There are only 1,193 labeled Scratch wafers in the entire dataset, of which ~953 land in training after the predefined split. The Scratch pattern — a thin diagonal line across the wafer — is visually distinct from every other class, but the model has so few real examples to learn from that it can only partially generalise.

**The val F1 vs test F1 gap is real.**
The best CBAM validation F1 was 0.946; the test Macro F1 is 0.656. This isn't overfitting in the usual sense — the training pool is ~68% "none", so the validation set mirrors that. But the test set is 93% "none" (110k of 118k samples). A model trained with a balanced sampler learns to be more aggressive about predicting minority classes. On the validation set (68% none), this is fine. On the test set (93% none), there are many more "none" samples for the model to accidentally predict as defects, which tanks minority-class precision. This distribution mismatch between training and test is a structural property of the WM-811K predefined split, not something solvable with a different sampler strategy.

**Edge-Ring plateaued exactly the same in both models (F1 0.695).**
CBAM gave zero improvement here. This is interesting — it suggests Edge-Ring's difficulty isn't about spatial attention (both models learned its position equally well) but about something at the feature level, likely confusion with Edge-Loc. Both defects sit at the wafer edge; Edge-Ring forms a full ring while Edge-Loc is a partial arc. At 64×64 the model may not have enough resolution to reliably distinguish a full ring from a partial one in borderline cases.

---

## Potential next step: synthetic minority generation

The most principled fix for Scratch (and to a lesser extent Donut and Near-full) is to generate synthetic training examples using a **convolutional autoencoder** trained on the real minority-class maps.

**How it works:**

A convolutional autoencoder has two halves. The encoder compresses a 64×64 wafer map down to a small latent vector (e.g., 64 floats) through a stack of conv + pool layers — this forces it to learn a compact representation of what makes a Scratch look like a Scratch. The decoder then uses transposed convolutions (the reverse of pooling) to reconstruct the original map from that latent vector. Training minimises the pixel-wise reconstruction error between input and output.

Once trained, you can generate new Scratch maps by:
1. Encoding all real Scratch wafers to get their latent vectors
2. Slightly perturbing those vectors (adding small random noise)
3. Decoding the perturbed vectors back to 64×64 maps

The result is a set of synthetic wafer maps that look plausibly like real Scratch defects but are not exact copies. These are added to the training set before the classifier runs, giving it 5–10× more Scratch examples to learn from.

**Why this is more principled than oversampling:**
`WeightedRandomSampler` with `replacement=True` repeats the same ~953 Scratch maps over and over. The model memorises them rather than generalising. Synthetic generation produces genuinely new examples that interpolate through the space of Scratch patterns, giving the model novel variation to learn from.

**What it would take in this project:**
- A new notebook section or separate notebook training an autoencoder per minority class (Scratch, Donut, Near-full)
- Generation of synthetic maps, visual inspection to confirm they look reasonable
- Augmenting the training dataframe with synthetic rows before building the DataLoader
- Rerunning the classifier with the expanded training set

This is the approach used in recent semiconductor defect classification literature — a 2026 Frontiers paper combining autoencoder upsampling, "none" downsampling, and CBAM reports 99.83% macro F1 on a similar setup. The gap between that result and the 0.656 achieved here is almost entirely explained by the synthetic data — the model architecture and attention mechanism are very similar.

---

## Key figures

After running the notebook, the following plots are saved to `figures/`:

- **`training_curves.png`** — loss, accuracy, and val Macro F1 over epochs for the baseline run
- **`training_curves_augmented.png`** — the same for the CBAM run
- **`confusion_matrix.png`** — normalised confusion matrix for the baseline, showing recall per cell
- **`per_class_f1_baseline.png`** and **`per_class_f1_augmented.png`** — per-class F1 bar charts for each run
- **`sample_predictions_baseline.png`** and **`sample_predictions_augmented.png`** — 4×4 grids of test predictions with correct and incorrect calls highlighted
- **`model_comparison.png`** — grouped bar chart comparing accuracy, macro F1, and weighted F1 across all runs

---

## How to run

```bash
# Install dependencies (uses uv)
uv sync

# Launch the notebooks
jupyter notebook
```

Start with `notebooks/00-wm-811k-EDA.ipynb` to get a feel for the data, then move on to `notebooks/01-wm-811k-cnn.ipynb` for the full training pipeline. Run the cells in order from top to bottom.

**Getting the data**

Download `LSWMD.pkl` from [Kaggle](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map) and place it in the `data/` directory. The notebooks expect the file to be named `LSWMD.pkl`.

One heads-up: the original file was serialised with an older version of pandas and numpy, so you may get an error trying to open it with a modern stack. If that happens, the fix is to load it in an environment with older versions of those libraries and re-save it as a fresh pickle. The notebook then reads the converted file without any issues.

---

## Running on GPU (RunPod)

The notebook defaults to `dev_mode = True`, which trains on a small stratified subset suitable for CPU testing. For the full GPU run:

1. Clone the repo and install dependencies:
   ```bash
   git clone https://github.com/intuitiveengineer/wm-811k-deep-learning.git
   pip install -e .
   ```
2. Upload `data/LSWMD.pkl` via the RunPod file browser
3. Open Jupyter Lab (provided by the RunPod PyTorch template on port 8888)
4. In the first cell of `01-wm-811k-cnn.ipynb`, set `dev_mode = False`
5. Run all cells — the notebook automatically uses CUDA if available
6. Once training completes, run the artifact checklist cell at the end to verify everything saved
7. Push tracked artifacts back to GitHub:
   ```bash
   git add figures/ tables/ notebooks/ && git commit -m "full gpu run results" && git push
   ```
8. Download `best_model.pt`, `best_model_v2.pt`, and `best_model_bundle.pt` from the RunPod file browser for local inference

The inference bundle (`best_model_bundle.pt`) contains everything needed to run predictions offline: model weights, class names, input size, and normalisation stats. Load it with the inference demo cell at the end of the notebook, no retraining required.
