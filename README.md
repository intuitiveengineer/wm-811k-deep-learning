# WM-811K Wafer Defect Classification

This project uses a convolutional neural network to classify semiconductor wafer maps from the WM-811K dataset into nine categories: eight distinct defect patterns plus a clean baseline. It's a portfolio project, so the focus is on building a principled, well-reasoned ML pipeline rather than chasing benchmark numbers. The dataset comes with some genuinely interesting challenges: severe class imbalance, a predefined evaluation split, and very small 45x48 pixel inputs.

---

## The Dataset

The [WM-811K dataset](https://doi.org/10.1109/TSMC.2018.2841818) (MIR-WM811K) contains 811,457 wafer maps collected from real semiconductor manufacturing lines. You can download it from [Kaggle](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map). Each wafer map is a small grid (45x48 pixels) where each cell encodes the pass/fail status of an individual die. Of the 811k records, only 172,950 are labeled, and the class distribution is heavily skewed:

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

**Class-weighted loss**
With 147k "none" samples sitting alongside just 149 "Near-full" samples, a model can reach 85% accuracy simply by predicting "none" every time. Inverse-frequency class weights applied to `CrossEntropyLoss` give the rarer classes meaningful gradient signal during training.

**Label-preserving augmentation**
Wafer maps have fourfold rotational symmetry, so horizontal flips, vertical flips, and 90 degree rotations all produce valid maps of the same defect type. The second experiment adds `RandomErasing`, which blanks a small random patch to simulate partial measurement loss and stop the model from leaning too heavily on specific spatial regions.

**Early stopping and LR scheduling**
`ReduceLROnPlateau` halves the learning rate whenever validation loss plateaus. `EarlyStopping` then restores the best checkpoint and halts training if no improvement is seen for 10 consecutive epochs. Both mechanisms work together to avoid overfitting without wasting compute.

---

## Results

Full per-class breakdowns for each run are saved in [`tables/`](tables/). The summary comparison is below, and [`tables/model_comparison.md`](tables/model_comparison.md) stays updated as new experiments are added.

| Run | Accuracy | Macro F1 | Weighted F1 | Epochs |
|-----|----------|----------|-------------|--------|
| Baseline CNN | - | - | - | - |
| Augmented CNN (+ RandomErasing) | - | - | - | - |

> Run `notebooks/01-wm-811k-cnn.ipynb` from top to bottom to populate these results.

---

## Key figures

After running the notebook, the following plots are saved to `figures/`:

- **`training_curves.png`** - loss and accuracy over epochs for the baseline run
- **`training_curves_augmented.png`** - the same for the augmented run
- **`confusion_matrix.png`** - normalised confusion matrix for the baseline, showing recall per cell
- **`per_class_f1_baseline.png`** and **`per_class_f1_augmented.png`** - per-class F1 bar charts for each run
- **`sample_predictions_baseline.png`** and **`sample_predictions_augmented.png`** - 4x4 grids of test predictions with correct and incorrect calls highlighted
- **`model_comparison.png`** - grouped bar chart comparing accuracy, macro F1, and weighted F1 across all runs

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
