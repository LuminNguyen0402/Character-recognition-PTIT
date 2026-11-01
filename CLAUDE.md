# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Course project (CSC14003 – Introduction to AI, VNU-HCMUS) comparing MLP and CNN for 47-class handwritten character recognition on the EMNIST ByMerge dataset (digits 0–9, A–Z, plus 11 visually-distinct lowercase letters).

Both models are delivered as Jupyter notebooks with an identical pipeline: load CSV → shuffle → normalize → one-hot encode → train (10 epochs, batch 256, Adam, categorical cross-entropy) → plot accuracy/loss → evaluate → confusion matrix → save `.h5`.

## Files

- `mlp.ipynb` — Multi-layer Perceptron on flattened 784-d input.
- `cnn.ipynb` — Convolutional Neural Network on 28×28×1 input.
- `README.md` — Full academic report (architecture, hyperparameters, results). Read this before making any model change.
- `.ipynb_checkpoints/` — Jupyter autosave; treat as ephemeral.

## Model architectures

Both models share `to_categorical(..., 47)` labels, Adam optimizer with default LR 0.001, `validation_split=0.1`, and the same `label` dict (index → character) — keep them in sync if you change one.

**MLP (`mlp.ipynb` cell 10):**
`Dense(512, ReLU) → Dropout(0.2) → Dense(512, ReLU) → Dropout(0.2) → Dense(47, softmax)` on 784-d flattened input.

**CNN (`cnn.ipynb` cell 10):**
- Block 1: `Conv2D(32, 3, padding=same) → LeakyReLU → Conv2D(32, 3, padding=same) → LeakyReLU → MaxPool2D(2,2) → Dropout(0.25)`
- Block 2: same with 64 filters.
- Head: `Flatten → Dense(512, ReLU) → Dense(128, ReLU) → Dense(47, softmax)`.

## Setup & run on Windows (Anaconda + JupyterLab)

### 1. Create env and install deps

Open **Anaconda Prompt** and run:

```bash
conda create -n emnist python=3.10 -y
conda activate emnist

pip install tensorflow numpy pandas matplotlib scikit-learn jupyterlab
```

- `tensorflow` from pip on Windows installs the CPU build by default; fine for this dataset. For GPU acceleration use `pip install tensorflow[and-cuda]` instead.
- RAM > 8 GB recommended (per README).

### 2. Get the dataset

Both notebooks expect `emnist-bymerge-train.csv` and `emnist-bymerge-test.csv` in the directory JupyterLab launches from (i.e. the project folder).

Option A — Kaggle CLI (mirrors the `# COLAB ONLY` cell):
```bash
pip install kaggle
```
Place your `kaggle.json` at `C:\Users\<you>\.kaggle\kaggle.json`, then:
```bash
kaggle datasets download -d crawford/emnist
```
The zip contains many files; extract only `emnist-bymerge-train.csv` and `emnist-bymerge-test.csv` into the project folder.

Option B — manual download from https://www.kaggle.com/crawford/emnist.

### 3. Launch JupyterLab from the project folder

```bash
cd G:\1.SourceCode\14.AI-Project-MachineLearning-master
jupyter lab
```

Browser opens at `http://localhost:8888`.

### 4. Run a notebook

- Open `mlp.ipynb` or `cnn.ipynb`.
- **Skip the first cell** — it's `# COLAB ONLY` and will fail locally (Kaggle/Colab shell commands).
- Run the remaining cells top-to-bottom (`Shift+Enter` or `Kernel → Run All`).
- Cell 4 reshapes data: CNN expects `[N, 28, 28, 1]`; MLP stays flat `[N, 784]`. Don't mix notebook code.
- Cell 12 (`model.fit(...)`) is where training actually happens — 10 epochs, batch 256. Tens of minutes per model on CPU; CNN is slower than MLP.

### 5. Outputs (written to the launch directory)

- `mlp_model.h5` / `cnn_model.h5` — trained weights (from `model.save` cell).
- `acc_basic_*.png`, `loss_basic_*.png` — training curves.
- `result_*.png`, `error_plots_*.png` — sample predictions.
- Confusion matrix plots are shown via `plt.show()` (not saved).

### Skip training entirely

The last two cells of each notebook load `*_model.h5` and call `model.evaluate(...)`. If you have a pre-trained `.h5` from a prior local run (the README's Google Drive folder is no longer maintained), place it next to the notebook and run only the data-loading cells plus the final reload-and-evaluate cells. Otherwise, train locally — see step 4 above. (Note: README's Google Drive folder is no longer maintained — train locally instead.)

## Output artifacts — what they mean

Each notebook writes the following into the launch directory. Reading them together tells you whether training succeeded and where the model fails.

- **`*_model.h5`** — full serialized model: architecture + every learned weight. ~7 MB (MLP) / ~15 MB (CNN). Reload with `tf.keras.models.load_model(...)` to predict on new data or resume training. Skip cells 21–22 of the notebooks — you already have `model` in memory after training; use `model.evaluate(test_data, test_label)` directly if you want a final accuracy number without reloading.

- **`acc_basic_*.png`** — accuracy curves, two lines per chart:
  - **train**: per-epoch accuracy on data the model is learning from.
  - **val**: per-epoch accuracy on the 10% holdout (from `validation_split=0.1`).
  - Healthy: both rise together, val close to train, finish ~0.88–0.92 on EMNIST ByMerge.
  - **Overfitting**: train keeps rising while val plateaus or falls — more dropout, fewer epochs, or augmentation would help.
  - **Flat near 0.02** (= 1/47): setup bug, almost certainly a cell that didn't run (likely cell 6, the one-hot encoding).

- **`loss_basic_*.png`** — categorical cross-entropy loss over epochs (lower is better). Same shape as the accuracy chart but more sensitive: if **val loss rises while val accuracy holds steady**, that's an early overfitting signal that the accuracy chart hides.

- **`result_*.png`** — 5×10 grid of test images with `P:` (predicted) and `L:` (ground-truth label) per cell. Use to sanity-check predictions and spot systematic misclassifications. EMNIST ByMerge's known weak pairs (per the README): O/0, Z/2, S/5, I/l/1 — partly caused by the dataset being stored transposed, which the notebooks do not un-flip in cell 4.

- **`error_plots_*.png`** — first 50 misclassified samples with `T:` (true) and `P:` (predicted). Hardcoded limit in cell 19 (`count = 50`). Look for **systematic** confusion patterns (same pair failing repeatedly) vs. **noise** (one-off mistakes). For a 10% error rate on 116k test images, 50 errors is a small sample — patterns are more meaningful than counts.

- **Confusion matrix** — 47×47 heatmap, saved to `confusion_matrix_<model>.png` (via `plt.savefig` in cell 18) and also shown via `plt.show()`. Diagonal = correct, off-diagonal = errors. The notebook produces both raw counts and a normalized version (each row sums to 1) — normalized is more useful for spotting per-class struggles regardless of class size.

### Troubleshooting

- **`ModuleNotFoundError: No module named 'tensorflow'`** — JupyterLab was launched outside the `emnist` env. Either `conda activate emnist` first, or register a kernel:
  ```bash
  python -m ipykernel install --user --name emnist
  ```
  Then pick the `emnist` kernel inside JupyterLab.
- **Kaggle cell fails locally** — confirm you skipped the `# COLAB ONLY` cell and that the CSVs sit in the working directory.
- **`ResourceExhaustedError` / OOM** — in cell 12, lower `batch_size` from 256 → 128 (or 64).
- **Plots look wrong on re-run** — the visualization cells multiply `test_data *= 255` in place; re-running them after already running once double-saturates the images. Restart the kernel (or reload data) before re-running visualization cells.

## Notes for future work

- Both notebooks cast data to `float16` to save memory; the visualization cells later multiply back by 255 and cast to `uint8` for `imshow`. Re-running visualization cells twice will double-saturate the images — reload from CSV or restart the kernel.
- EMNIST ByMerge images are stored transposed; the notebooks do **not** apply `np.transpose`/`np.flip`, which the README identifies as a likely source of misclassifications between visually similar pairs (O/0, Z/2, S/5). Any preprocessing fix should go in cell 4 of both notebooks consistently.
- Output artifacts (`*.png`, `*.h5`) are written to the notebook working directory; clean them up before re-running to avoid stale plots overwriting fresh ones with the same name.
- Adding/removing a class changes the `label` dict (cell 5) in both notebooks and the `Dense(47, ...)` output layer size; do both together.
- `archive.zip` (~1.3 GB) and the empty `archive/` and `certs/` directories are leftovers from the Kaggle download step. Safe to delete; the notebooks only need `emnist-bymerge-train.csv` and `emnist-bymerge-test.csv` in the working directory.
