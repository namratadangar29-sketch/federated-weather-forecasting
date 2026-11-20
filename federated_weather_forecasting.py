import os
import argparse
import numpy as np
import pandas as pd

# ==============================================================================
# 1. CONFIGURATION & CONSTANTS
# ==============================================================================
FEATURE_COLUMNS = [
    "humidity_pct",
    "pressure_hpa",
    "rainfall_mm",
    "wind_speed_ms",
    "cloud_cover_pct",
]
TARGET_COLUMN = "temperature_c"


# ==============================================================================
# 2. DATA LOADING, PARTITIONING & FEDERATED SCALING
# ==============================================================================
def load_and_partition_clients(csv_path, save_dir=None):
    """Load dataset and split into independent geographical FL clients."""
    df = pd.read_csv(csv_path)
    locations = sorted(df["location"].dropna().unique())
    clients = {}

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    for i, location in enumerate(locations, start=1):
        client_id = f"client_{i:02d}"
        client_data = df[df["location"] == location].copy()
        client_data["collected_at"] = pd.to_datetime(
            client_data["collected_at"], errors="coerce"
        )
        client_data = client_data.sort_values("collected_at").reset_index(drop=True)
        clients[client_id] = client_data

        if save_dir:
            client_data.to_csv(os.path.join(save_dir, f"{client_id}.csv"), index=False)

    assert len(clients) == len(locations), "Client count mismatch."
    return clients


def prepare_federated_splits(clients):
    """Split each client chronologically (70/15/15) and compute shared global scaling."""
    client_splits = {}
    total_count = 0
    X_sum = np.zeros(len(FEATURE_COLUMNS))
    X_sq_sum = np.zeros(len(FEATURE_COLUMNS))
    y_sum = 0.0
    y_sq_sum = 0.0

    for client_id, client_df in clients.items():
        data = (
            client_df.sort_values("collected_at")[FEATURE_COLUMNS + [TARGET_COLUMN]]
            .dropna()
            .copy()
        )
        n = len(data)
        train_end = int(n * 0.70)
        val_end = int(n * 0.85)

        train_df = data.iloc[:train_end]
        val_df = data.iloc[train_end:val_end]
        test_df = data.iloc[val_end:]

        X_train = train_df[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
        y_train = train_df[[TARGET_COLUMN]].to_numpy(dtype=np.float64)

        client_splits[client_id] = {
            "X_train": X_train,
            "y_train": y_train,
            "X_val": val_df[FEATURE_COLUMNS].to_numpy(dtype=np.float64),
            "y_val": val_df[[TARGET_COLUMN]].to_numpy(dtype=np.float64),
            "X_test": test_df[FEATURE_COLUMNS].to_numpy(dtype=np.float64),
            "y_test": test_df[[TARGET_COLUMN]].to_numpy(dtype=np.float64),
        }

        total_count += len(X_train)
        X_sum += np.sum(X_train, axis=0)
        X_sq_sum += np.sum(X_train ** 2, axis=0)
        y_sum += float(np.sum(y_train))
        y_sq_sum += float(np.sum(y_train ** 2))

    global_X_mean = X_sum / total_count
    global_y_mean = y_sum / total_count

    global_X_std = np.sqrt(np.maximum(X_sq_sum / total_count - global_X_mean ** 2, 0.0))
    global_y_std = float(np.sqrt(max(y_sq_sum / total_count - global_y_mean ** 2, 0.0)))

    global_X_std[global_X_std == 0] = 1.0
    if global_y_std == 0:
        global_y_std = 1.0

    scaler = {
        "X_mean": global_X_mean,
        "X_std": global_X_std,
        "y_mean": global_y_mean,
        "y_std": global_y_std,
    }

    scaled_clients = {}
    for client_id, splits in client_splits.items():
        scaled_clients[client_id] = {
            "X_train": (splits["X_train"] - global_X_mean) / global_X_std,
            "y_train": (splits["y_train"] - global_y_mean) / global_y_std,
            "X_val": (splits["X_val"] - global_X_mean) / global_X_std,
            "y_val": (splits["y_val"] - global_y_mean) / global_y_std,
            "X_test": (splits["X_test"] - global_X_mean) / global_X_std,
            "y_test": (splits["y_test"] - global_y_mean) / global_y_std,
        }

    return client_splits, scaled_clients, scaler


# ==============================================================================
# 3. NEURAL NETWORK (MLP) FROM SCRATCH
# ==============================================================================
def initialize_mlp(input_size=5, hidden_size=8, output_size=1, seed=42):
    rng = np.random.default_rng(seed)
    return {
        "W1": rng.standard_normal((input_size, hidden_size)) * 0.1,
        "b1": np.zeros((1, hidden_size)),
        "W2": rng.standard_normal((hidden_size, output_size)) * 0.1,
        "b2": np.zeros((1, output_size)),
    }


def relu(x):
    return np.maximum(0, x)


def forward_propagation(X, parameters):
    Z1 = np.dot(X, parameters["W1"]) + parameters["b1"]
    A1 = relu(Z1)
    Z2 = np.dot(A1, parameters["W2"]) + parameters["b2"]
    return Z2, (X, Z1, A1, Z2)


def backward_propagation(y_true, parameters, cache):
    X, Z1, A1, Z2 = cache
    n = X.shape[0]

    dZ2 = (2.0 / n) * (Z2 - y_true)
    dW2 = np.dot(A1.T, dZ2)
    db2 = np.sum(dZ2, axis=0, keepdims=True)

    dA1 = np.dot(dZ2, parameters["W2"].T)
    dZ1 = dA1 * (Z1 > 0)
    dW1 = np.dot(X.T, dZ1)
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}


# ==============================================================================
# 4. FEDERATED LEARNING (LOCAL SGD, FEDAVG, PERSONALIZATION)
# ==============================================================================
def train_local_model(
    X_train, y_train, initial_parameters, epochs=5, batch_size=64, learning_rate=0.01, seed=42
):
    parameters = {k: v.copy() for k, v in initial_parameters.items()}
    rng = np.random.default_rng(seed)
    n = len(X_train)

    for _ in range(epochs):
        indices = rng.permutation(n)
        for start in range(0, n, batch_size):
            batch_idx = indices[start : start + batch_size]
            _, cache = forward_propagation(X_train[batch_idx], parameters)
            grads = backward_propagation(y_train[batch_idx], parameters, cache)
            for name in parameters:
                parameters[name] -= learning_rate * grads[name]

    return parameters


def fedavg(local_updates, client_sample_counts):
    total_samples = sum(client_sample_counts.values())
    sample_params = next(iter(local_updates.values()))
    aggregated = {name: np.zeros_like(val) for name, val in sample_params.items()}

    for client_id, params in local_updates.items():
        weight = client_sample_counts[client_id] / total_samples
        for name in aggregated:
            aggregated[name] += weight * params[name]

    return aggregated


# ==============================================================================
# 5. EVALUATION & HETEROGENEITY METRICS
# ==============================================================================
def regression_metrics(y_true, y_pred):
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1.0 - (ss_res / ss_tot))
    return mae, rmse, r2


def tolerance_accuracy(y_true, y_pred, tolerance):
    errors = np.abs(y_true - y_pred)
    return float((np.sum(errors <= tolerance) / len(y_true)) * 100.0)


# ==============================================================================
# 6. MAIN EXECUTION PIPELINE
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Federated Weather Forecasting")
    parser.add_argument("--data-path", type=str, default="Dataset.csv", help="Path to Dataset.csv")
    parser.add_argument("--rounds", type=int, default=6, help="Number of FedAvg rounds")
    parser.add_argument("--local-epochs", type=int, default=5, help="Local epochs per round")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=0.01, help="Global learning rate")
    parser.add_argument("--finetune-lr", type=float, default=0.001, help="Personalization LR")
    args = parser.parse_args()

    # Step 1: Load & partition clients
    print(f"1. Loading dataset from '{args.data_path}'...")
    clients = load_and_partition_clients(args.data_path)
    client_splits, scaled_clients, scaler = prepare_federated_splits(clients)
    print(f"   Successfully created {len(clients)} federated clients.\n")

    # Step 2: Initialize global MLP
    global_params = initialize_mlp(input_size=len(FEATURE_COLUMNS), hidden_size=8, output_size=1, seed=42)

    # Step 3: Federated Averaging training rounds
    print(f"2. Running {args.rounds} Federated Learning rounds...")
    for round_num in range(1, args.rounds + 1):
        local_updates = {}
        sample_counts = {}

        for idx, (client_id, data) in enumerate(scaled_clients.items()):
            updated_params = train_local_model(
                data["X_train"],
                data["y_train"],
                global_params,
                epochs=args.local_epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                seed=1000 * round_num + idx,
            )
            local_updates[client_id] = updated_params
            sample_counts[client_id] = len(data["X_train"])

        global_params = fedavg(local_updates, sample_counts)

        # Validation evaluation
        val_actual, val_preds = [], []
        for data in scaled_clients.values():
            pred, _ = forward_propagation(data["X_val"], global_params)
            val_actual.append(data["y_val"] * scaler["y_std"] + scaler["y_mean"])
            val_preds.append(pred * scaler["y_std"] + scaler["y_mean"])

        val_mae, val_rmse, val_r2 = regression_metrics(
            np.concatenate(val_actual), np.concatenate(val_preds)
        )
        print(
            f"   Round {round_num:02d}/{args.rounds:02d} | "
            f"Val MAE: {val_mae:.4f} °C | Val RMSE: {val_rmse:.4f} °C | Val R²: {val_r2:.4f}"
        )

    # Step 4: Personalization & Adaptive Selection
    print("\n3. Fine-tuning personalized models & evaluating adaptive selection...")
    personalized_models = {}
    use_personalized = {}

    for idx, (client_id, data) in enumerate(scaled_clients.items()):
        p_params = train_local_model(
            data["X_train"],
            data["y_train"],
            global_params,
            epochs=3,
            batch_size=args.batch_size,
            learning_rate=args.finetune_lr,
            seed=5000 + idx,
        )
        personalized_models[client_id] = p_params

        g_val_pred, _ = forward_propagation(data["X_val"], global_params)
        p_val_pred, _ = forward_propagation(data["X_val"], p_params)

        actual_val = data["y_val"] * scaler["y_std"] + scaler["y_mean"]
        g_val_mae = np.mean(np.abs(actual_val - (g_val_pred * scaler["y_std"] + scaler["y_mean"])))
        p_val_mae = np.mean(np.abs(actual_val - (p_val_pred * scaler["y_std"] + scaler["y_mean"])))

        use_personalized[client_id] = p_val_mae < g_val_mae

    # Step 5: Final Test Evaluation
    y_actual_all, y_global_all, y_personal_all, y_adaptive_all = [], [], [], []
    client_test_mae = {}

    for client_id, data in scaled_clients.items():
        actual = data["y_test"] * scaler["y_std"] + scaler["y_mean"]
        g_pred, _ = forward_propagation(data["X_test"], global_params)
        p_pred, _ = forward_propagation(data["X_test"], personalized_models[client_id])

        g_celsius = g_pred * scaler["y_std"] + scaler["y_mean"]
        p_celsius = p_pred * scaler["y_std"] + scaler["y_mean"]
        a_celsius = p_celsius if use_personalized[client_id] else g_celsius

        y_actual_all.append(actual)
        y_global_all.append(g_celsius)
        y_personal_all.append(p_celsius)
        y_adaptive_all.append(a_celsius)
        client_test_mae[client_id] = float(np.mean(np.abs(actual - g_celsius)))

    y_true = np.concatenate(y_actual_all)
    strategies = {
        "Global FedAvg": np.concatenate(y_global_all),
        "Personalized FL": np.concatenate(y_personal_all),
        "Adaptive FL": np.concatenate(y_adaptive_all),
    }

    results = []
    for name, preds in strategies.items():
        mae, rmse, r2 = regression_metrics(y_true, preds)
        results.append({
            "Strategy": name,
            "MAE (°C)": round(mae, 4),
            "RMSE (°C)": round(rmse, 4),
            "R²": round(r2, 4),
            "Acc ±0.5°C (%)": round(tolerance_accuracy(y_true, preds, 0.5), 2),
            "Acc ±1.0°C (%)": round(tolerance_accuracy(y_true, preds, 1.0), 2),
            "Acc ±2.0°C (%)": round(tolerance_accuracy(y_true, preds, 2.0), 2),
        })

    print("\n=== FINAL POOLED TEST COMPARISON ===")
    print(pd.DataFrame(results).to_string(index=False))

    # Step 6: Non-IID Heterogeneity Diagnosis
    het_rows = []
    for client_id, data in client_splits.items():
        mean_diff = (np.mean(data["X_train"], axis=0) - scaler["X_mean"]) / scaler["X_std"]
        std_diff = (np.std(data["X_train"], axis=0) - scaler["X_std"]) / scaler["X_std"]
        het_rows.append({
            "Feature Mean Distance": np.sqrt(np.mean(mean_diff ** 2)),
            "Feature Std Distance": np.sqrt(np.mean(std_diff ** 2)),
            "Target Mean Shift": abs(np.mean(data["y_train"]) - scaler["y_mean"]) / scaler["y_std"],
            "Test MAE": client_test_mae[client_id],
        })
    het_df = pd.DataFrame(het_rows)

    print("\n=== NON-IID HETEROGENEITY CORRELATIONS WITH TEST MAE ===")
    for col in ["Feature Mean Distance", "Feature Std Distance", "Target Mean Shift"]:
        print(f"   {col}: r = {het_df[col].corr(het_df['Test MAE']):.4f}")


if __name__ == "__main__":
    main()