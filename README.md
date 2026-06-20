# WM-811K Wafer Defect Classification

This project uses a convolutional neural network to classify semiconductor wafer maps from the WM-811K dataset into nine categories: eight distinct defect patterns plus a clean baseline. It's a portfolio project, so the focus is on building a principled, well-reasoned ML pipeline rather than chasing benchmark numbers. The dataset comes with some genuinely interesting challenges: severe class imbalance, a predefined evaluation split, and maps of varying sizes across manufacturing lots.

---

## The Dataset

The [WM-811K dataset](https://doi.org/10.1109/TSMC.2018.2841818) (MIR-WM811K) contains 811,457 wafer maps collected from real semiconductor manufacturing lines. You can download it from [Kaggle](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map). Each wafer map is a small grid where each cell encodes the pass/fail status of an individual die. Maps vary in size across manufacturing lots; the pipeline resizes everything to 32x32 before training. Of the 811k records, only 172,950 are labeled, and the class distribution is heavily skewed:

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
Wafer maps have rotational symmetry, so horizontal flips, vertical flips, and 180-degree rotations all produce valid maps of the same defect type. The second experiment adds a small random translation via `T.RandomAffine`, which shifts the die pattern slightly within the frame to simulate minor scanner misalignment without distorting the defect signature.

**Early stopping and LR scheduling on Macro F1**
Accuracy is a misleading metric when one class makes up 85% of the data. Both `ReduceLROnPlateau` and `EarlyStopping` track validation Macro F1 instead, which treats every class equally regardless of how often it appears. The best checkpoint by Macro F1 is saved and restored at the end of training.

---

## Results

Full per-class breakdowns for each run are saved in [`tables/`](tables/). The summary comparison is below, and [`tables/model_comparison.md`](tables/model_comparison.md) stays updated as new experiments are added.

| Run | Accuracy | Macro F1 | Weighted F1 | Epochs |
|-----|----------|----------|-------------|--------|
| Baseline CNN | - | - | - | - |
| Augmented CNN (+ Translation) | - | - | - | - |

> Run `notebooks/01-wm-811k-cnn.ipynb` from top to bottom to populate these results. The primary metric is **Macro F1** — it weights every class equally and is far more informative than accuracy for this imbalanced dataset.

---

## Key figures

After running the notebook, the following plots are saved to `figures/`:

- **`training_curves.png`** - loss, accuracy, and val Macro F1 over epochs for the baseline run
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
