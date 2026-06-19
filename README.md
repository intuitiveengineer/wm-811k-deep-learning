# WM-811K Wafer Defect Classification

This project applies a convolutional neural network to classify semiconductor wafer maps from the WM-811K dataset into nine categories: eight distinct defect patterns and a clean (no-defect) baseline. It's a portfolio project — the goal isn't to chase a leaderboard score, but to build a rigorous, well-reasoned ML pipeline that handles the real challenges this dataset throws at you: severe class imbalance, a predefined evaluation split, and tiny 45×48 pixel inputs.

---

## The Dataset

The [WM-811K dataset](https://doi.org/10.1109/TSMC.2018.2841818) (MIR-WM811K) contains 811,457 wafer maps collected from real semiconductor manufacturing. Each wafer map is a small binary/ternary grid (45×48 pixels) encoding the pass/fail status of individual dies. Only 172,950 records are labeled, and the class distribution is highly skewed:

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

The dataset ships with a predefined `trainTestLabel` split, which this project respects — the test set is never touched during training or validation.

---

## What's in this repo

```
notebooks/
  00-wm-811k-EDA.ipynb      Exploratory analysis: distributions, sample maps, shape checks
  01-wm-811k-cnn.ipynb      Full CNN pipeline: training, evaluation, model comparison

figures/                    Saved plots (training curves, confusion matrix, comparisons, etc.)
tables/                     Saved metrics as Markdown tables

data/                       Dataset file — gitignored, not included in the repo
learning_log.md             Decision log explaining every design choice — gitignored
```

---

## Pipeline highlights

**No data leakage**
The dataset's predefined train/test split is used as-is. Normalization statistics (mean, std) are computed from the training split only and applied to val and test. The test set is touched exactly once — for final evaluation.

**Stratified validation split**
Twenty percent of the training pool is held out as a validation set using stratified sampling, ensuring every class (including rare ones like Near-full) is represented proportionally.

**Class-weighted loss**
With 147k "none" samples vs 149 "Near-full" samples, a naive model can hit 85% accuracy by predicting "none" everywhere. Inverse-frequency class weights in `CrossEntropyLoss` give the rarer classes meaningful gradient signal.

**Label-preserving augmentation**
Wafer maps have fourfold rotational symmetry — flipping and rotating 90° produces a valid map of the same defect type. RandomErasing (used in the second experiment) blanks a small random patch, simulating partial measurement loss and discouraging the model from fixating on specific spatial regions.

**Early stopping + LR scheduling**
`ReduceLROnPlateau` halves the learning rate when validation loss plateaus; `EarlyStopping` restores the best checkpoint and terminates training once improvement stalls for 10 consecutive epochs.

---

## Results

Full per-class metrics for each run are in [`tables/`](tables/). The summary comparison is below.

See [`tables/model_comparison.md`](tables/model_comparison.md) for the live leaderboard — each experiment appends a row.

| Run | Accuracy | Macro F1 | Weighted F1 | Epochs |
|-----|----------|----------|-------------|--------|
| Baseline CNN | — | — | — | — |
| Augmented CNN (+ RandomErasing) | — | — | — | — |

> Run `notebooks/01-wm-811k-cnn.ipynb` top-to-bottom to populate these results.

---

## Key figures

After running the notebook, the following plots are saved to `figures/`:

- **`training_curves.png`** — loss and accuracy over epochs for the baseline run
- **`training_curves_augmented.png`** — same for the augmented run
- **`confusion_matrix.png`** — normalised confusion matrix for the baseline (recall per cell)
- **`per_class_f1_baseline.png`** / **`per_class_f1_augmented.png`** — per-class F1 bar charts
- **`sample_predictions_baseline.png`** / **`sample_predictions_augmented.png`** — 4×4 grids of test predictions with correct/wrong highlighting
- **`model_comparison.png`** — grouped bar chart of accuracy, macro F1, and weighted F1 across runs

---

## How to run

```bash
# Install dependencies (uses uv)
uv sync

# Launch the notebooks
jupyter notebook
```

Open `notebooks/00-wm-811k-EDA.ipynb` for the exploratory analysis, then `notebooks/01-wm-811k-cnn.ipynb` for the full training pipeline. Run cells top-to-bottom — the pipeline is designed to execute sequentially without skipping steps.

The dataset file (`data/LSWMD_modern.pkl`) is not included in the repo. Place it in the `data/` directory before running.
