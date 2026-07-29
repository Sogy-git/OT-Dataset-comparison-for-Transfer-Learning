# ot-select

Data selection for transfer learning using small target query sets. This project studies how to pick a good subset of a *source* dataset (MNIST) for fine-tuning a model toward a *target* dataset (SEMEION handwritten digits) when only a small labeled sample of the target is available.

A CNN is pretrained on MNIST, then fine-tuned on a subset of MNIST chosen by one of several selection strategies, and finally evaluated on held-out SEMEION digits.

## Selection methods

- **random_zero_shot / full_zero_shot** — baselines: the MNIST-pretrained model evaluated directly on SEMEION, with no fine-tuning (using a model pretrained on a random 1000-sample subset vs. the full MNIST training set).
- **KNN** — selects the MNIST training samples whose embeddings are nearest neighbors of the SEMEION target embeddings.
- **OT** — selects a subset via partial optimal transport between MNIST candidate embeddings and SEMEION target embeddings.
- **OTDD** — like OT, but the transport cost also incorporates an Optimal Transport Dataset Distance term between class distributions, so class structure informs the selection.

Embeddings are extracted from the penultimate fully-connected layer (`fc1`) of the CNN via a forward hook.

## Project structure

```
src/ot_select/
  model.py            SimpleCNN architecture (2 conv blocks + 2 FC layers)
  data_input.py        Builds MNIST and SEMEION datasets with reproducible splits
  data_choice.py        DataLoader construction (random / full / KNN / OT variants)
  train_func.py          Core train / test / evaluate loops
  encoder.py             Embedding extraction via forward hook on fc1
  distance_choice.py     KNN, OT, and OTDD selection algorithms
  pipeline.py             Orchestrates selection -> fine-tune -> evaluate for one (seed, ratio) run
  supervised_train.py    Pretrains baseline CNNs on MNIST (random subset and full)
  finetune.py             Single-configuration fine-tuning example (KNN/OT/OTDD)
  run_evaluation.py       Multi-seed, multi-ratio evaluation sweep + figure generation
  graphs.py                Plotting utilities (accuracy/loss curves, confusion matrices, etc.)

configs/                YAML experiment configs
data/                   MNIST and SEMEION datasets (downloaded automatically)
results/                Evaluation outputs: metrics (CSV/JSON), per-run models, figures
tests/                  Test suite
```

Note: modules in `src/ot_select` import each other directly (e.g. `from model import SimpleCNN`) rather than as a package, so scripts should be run from inside `src/ot_select/`.

## Setup

Requires Python >= 3.10.

```bash
pip install -e .
# or, for development (adds pytest, ruff):
pip install -e ".[dev]"
```

MNIST and SEMEION are downloaded automatically into `data/` on first use.

## Usage

1. **Pretrain baseline CNNs on MNIST** (produces `simple_cnn_random.pth` and `simple_cnn_full.pth`):

   ```bash
   cd src/ot_select
   python supervised_train.py
   ```

2. **Run the full evaluation sweep** across seeds and SEMEION train ratios, comparing zero-shot baselines against KNN/OT/OTDD-selected fine-tuning:

   ```bash
   python run_evaluation.py --output-dir ../../results
   ```

   Useful flags:
   - `--quick` — fast smoke test (1 seed, 10% ratio, 1 epoch)
   - `--seeds` / `--ratios` — override the default seeds (`42 69 123 456 789`) and SEMEION train ratios (`0.10 0.20 0.40`)
   - `--subset-size` — number of MNIST samples selected per method (default 5000)
   - `--skip-training` — reload existing `evaluation_results.json` and only regenerate figures

3. **Single fine-tuning run** (KNN/OT/OTDD, seed 69, 10% SEMEION ratio):

   ```bash
   python finetune.py
   ```

## Results

Results from the evaluation sweep are in `results/evaluation_results.csv` / `.json`, with figures (accuracy/loss vs. ratio, per-method comparisons, confusion matrices, seed stability) in `results/figures/`. Mean SEMEION test accuracy across 5 seeds:

| SEMEION train ratio | full_zero_shot | random_zero_shot | KNN | OT | OTDD |
|---|---|---|---|---|---|
| 10% | 36.8% | 12.6% | 33.2% | 33.5% | **35.2%** |
| 20% | 36.9% | 12.6% | 33.0% | 32.9% | **34.9%** |
| 40% | 36.3% | 12.6% | 33.1% | 32.6% | **34.4%** |

OTDD consistently outperforms KNN and OT, and random selection performs far worse than any distance-based method, but none of the fine-tuned methods yet surpass the full-MNIST zero-shot baseline. Fine-tuned model checkpoints for each seed/ratio/method combination are saved under `results/models/`.
