# Federated Weather Forecasting from Scratch

An end-to-end **Federated Learning (FL)** pipeline for multi-location air temperature prediction, implemented purely in **NumPy** and **Pandas** without external deep learning frameworks (no PyTorch or TensorFlow).

## Overview

Weather stations across distinct geographical regions can show **non-IID (non-independent and identically distributed)** data patterns due to local microclimates, elevation, and coastal effects. This project simulates a cross-silo federated network across **25 geographical clients** (one per location) to predict air temperature (`temperature_c`) from five meteorological features measured at the same timestamp:

* **Humidity** (`humidity_pct`)
* **Atmospheric Pressure** (`pressure_hpa`)
* **Rainfall** (`rainfall_mm`)
* **Wind Speed** (`wind_speed_ms`)
* **Cloud Cover** (`cloud_cover_pct`)

> **Note:** The model predicts temperature from concurrent readings (no lagged features or forecast horizon), so it is best described as temperature *estimation*. Extending it with lagged inputs would turn it into true forecasting.

## Key Features

1. **Zero-Framework Neural Network:** Forward propagation, ReLU activation, backpropagation, mini-batch SGD, and `FedAvg` weighted parameter aggregation built from scratch in NumPy.
2. **Chronological Splitting & Federated Scaling:**
   * Sorts each client's records by `collected_at` and splits them into **70% Train / 15% Validation / 15% Test**, so the test set always comes after the training data in time.
   * Computes global feature standardization (mean and standard deviation) from per-client sufficient statistics (`N`, `Σx`, `Σx²`) of the training data only, so clients share summary statistics rather than raw rows.
3. **Personalized & Adaptive FL:**
   * **Global FedAvg:** Trains a shared global MLP across federated communication rounds.
   * **Personalized FL:** Fine-tunes the global model locally on each client's data.
   * **Adaptive FL:** Uses validation MAE to select, per client, whether the fine-tuned personalized model or the global model performs better, which guards against negative transfer.
4. **Non-IID Heterogeneity Diagnostics:** Measures feature distribution distance and target shift across the 25 locations and correlates them with client-level test MAE.

## Repository Contents

* `federated_weather_forecasting.py` — Complete self-contained pipeline (data partitioning, MLP training, FedAvg, personalization, evaluation, and heterogeneity analysis).
* `Dataset.csv` — Multi-location meteorological dataset (about 130k rows, 25 locations) used for federated partitioning.
* `requirements.txt` — Python package dependencies (`numpy`, `pandas`).

## How to Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/namratadangar29-sketch/federated-weather-forecasting.git
   cd federated-weather-forecasting
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the pipeline:**
   ```bash
   python federated_weather_forecasting.py
   ```

### Command-line options

| Option | Default | Description |
|---|---|---|
| `--data-path` | `Dataset.csv` | Path to the dataset CSV |
| `--rounds` | `6` | Number of FedAvg communication rounds |
| `--local-epochs` | `5` | Local epochs per client per round |
| `--batch-size` | `64` | Mini-batch size |
| `--lr` | `0.01` | Learning rate for local training during FedAvg |
| `--finetune-lr` | `0.001` | Learning rate for personalization fine-tuning |

Example with more communication rounds:

```bash
python federated_weather_forecasting.py --rounds 40
```

## Example Results

Pooled test results across all 25 clients (default settings produce the 6-round numbers; the global model is still improving at round 6, so more rounds give a fairer comparison):

| Strategy | MAE (°C), 6 rounds | MAE (°C), 40 rounds | R², 40 rounds |
|---|---|---|---|
| Global FedAvg | 1.586 | 1.135 | 0.887 |
| Personalized FL | 0.964 | 0.935 | 0.920 |
| Adaptive FL | 0.948 | 0.834 | 0.936 |

Exact values may vary slightly with library versions. The correlations printed by the heterogeneity diagnostics change a lot with the number of rounds (they are strong at 6 rounds and near zero at 40), so treat them as exploratory rather than as a stable finding.

## Limitations

* This is a single-machine **simulation** of federated learning; the CSV is loaded in one place and then partitioned by location. There is no networking, client sampling, secure aggregation, or differential privacy.
* The dataset covers roughly one month per location, and many consecutive rows are exact duplicates, so the effective sample size is smaller than the row count.
* The network is deliberately small (one hidden layer of 8 units) and trained with plain SGD.
