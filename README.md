# Federated Weather Forecasting from Scratch

An end-to-end **Federated Learning (FL)** pipeline for multi-location air temperature forecasting, implemented purely in **NumPy** and **Pandas** without external deep learning frameworks (no PyTorch or TensorFlow).

## Overview

Weather stations across distinct geographical regions exhibit strong **non-IID (non-independent and identically distributed)** data patterns due to local microclimates, elevation, and coastal effects. This project simulates a cross-silo federated network across **25 geographical clients** to predict air temperature (`temperature_c`) from five meteorological features:

* **Humidity** (`humidity_pct`)
* **Atmospheric Pressure** (`pressure_hpa`)
* **Rainfall** (`rainfall_mm`)
* **Wind Speed** (`wind_speed_ms`)
* **Cloud Cover** (`cloud_cover_pct`)

## Key Features

1. **Zero-Framework Neural Network:** Forward propagation, ReLU activation, backpropagation, mini-batch SGD, and `FedAvg` weighted parameter aggregation built from scratch in NumPy.
2. **Leakage-Free Time-Series Splitting & Distributed Scaling:**
   * Sorts each client's records chronologically via `collected_at` and splits them into **70% Train / 15% Validation / 15% Test**.
   * Computes global feature standardization ($\mu, \sigma$) using distributed sufficient statistics ($N, \sum x, \sum x^2$) without centralizing raw client data.
3. **Personalized & Adaptive FL:**
   * **Global FedAvg:** Trains a shared global MLP across 6 federated communication rounds.
   * **Personalized FL:** Fine-tunes the global model locally on each client's data.
   * **Adaptive FL:** Uses validation MAE to dynamically select per client whether the fine-tuned personalized model or the global model performs better, preventing negative transfer.
4. **Non-IID Heterogeneity Diagnostics:** Quantifies feature distribution distance and target shift across all 25 locations and correlates them with client-level test MAE.

## Repository Contents

* `federated_weather_forecasting.py` — Complete self-contained pipeline (data partitioning, MLP training, FedAvg, personalization, evaluation, and heterogeneity analysis).
* `Dataset.csv` — Multi-location meteorological dataset used for federated partitioning.
* `requirements.txt` — Python package dependencies (`numpy`, `pandas`).

## How to Run

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/namratadangar29-sketch/federated-weather-forecasting.git](https://github.com/namratadangar29-sketch/federated-weather-forecasting.git)
   cd federated-weather-forecasting