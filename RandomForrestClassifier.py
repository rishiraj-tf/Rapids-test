import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cuml

from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

# Utility function to measure execution time
def time_execution(func, *args, **kwargs):
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    return result, end_time - start_time

def main():
    print("\n" + "="*50)
    print("PART 1: Random Forest Classification (Full Dataset)")
    print("="*50)

    # Load the entire Covertype dataset
    print("Loading full Covertype dataset...")
    data = fetch_covtype()
    X, y = data.data, data.target
    print(f"Full dataset shape: {X.shape}")

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ===========================================
    # First: Run with standard scikit-learn
    # ===========================================
    print("\nRunning with standard scikit-learn:")
    rf_sklearn = RandomForestClassifier(
        n_estimators=100, max_depth=20, random_state=42, n_jobs=-1
    )
    _, sklearn_time = time_execution(rf_sklearn.fit, X_train, y_train)
    print(f"Training time with scikit-learn: {sklearn_time:.2f} seconds")

    test_subset = 10000
    sklearn_preds, sklearn_pred_time = time_execution(
        rf_sklearn.predict, X_test[:test_subset]
    )
    print(f"Prediction time with scikit-learn (on {test_subset} samples): {sklearn_pred_time:.2f} seconds")
    sklearn_accuracy = accuracy_score(y_test[:test_subset], sklearn_preds)
    print(f"Accuracy with scikit-learn: {sklearn_accuracy:.4f}")

    # ===========================================
    # Now: Run with cuML acceleration
    # ===========================================
    print("\nRunning with cuML acceleration:")

    from cuml.ensemble import RandomForestClassifier as cuRF  # re-import after accel

    rf_cuml = cuRF(
        n_estimators=100, max_depth=20, random_state=42
    )
    _, cuml_time = time_execution(rf_cuml.fit, X_train, y_train)
    print(f"Training time with cuML: {cuml_time:.2f} seconds")

    cuml_preds, cuml_pred_time = time_execution(
        rf_cuml.predict, X_test[:test_subset]
    )
    print(f"Prediction time with cuML (on {test_subset} samples): {cuml_pred_time:.2f} seconds")
    cuml_accuracy = accuracy_score(y_test[:test_subset], cuml_preds)
    print(f"Accuracy with cuML: {cuml_accuracy:.4f}")

    # Calculate speedup
    rf_speedup = sklearn_time / cuml_time if cuml_time else 0
    rf_pred_speedup = sklearn_pred_time / cuml_pred_time if cuml_pred_time else 0
    print(f"\nSpeedup for Random Forest training: {rf_speedup:.2f}x")
    print(f"Speedup for Random Forest prediction: {rf_pred_speedup:.2f}x")

    # Visualize timing
    labels = ['scikit-learn', 'cuML (GPU)']
    train_times = [sklearn_time, cuml_time]
    pred_times = [sklearn_pred_time, cuml_pred_time]

    plt.figure(figsize=(12, 5))

    # Training time comparison
    plt.subplot(1, 2, 1)
    plt.bar(labels, train_times)
    plt.title('Random Forest Training Time (seconds)')
    plt.ylabel('Time (seconds)')
    for i, v in enumerate(train_times):
        plt.text(i, v + max(train_times)*0.05, f"{v:.2f}s", ha='center')

    # Prediction time comparison
    plt.subplot(1, 2, 2)
    plt.bar(labels, pred_times)
    plt.title('Random Forest Prediction Time (seconds)')
    plt.ylabel('Time (seconds)')
    for i, v in enumerate(pred_times):
        plt.text(i, v + max(pred_times)*0.05, f"{v:.2f}s", ha='center')

    plt.tight_layout()
    plt.savefig('rf_timing_comparison_large.png')
    plt.show()

if __name__ == "__main__":
    main()