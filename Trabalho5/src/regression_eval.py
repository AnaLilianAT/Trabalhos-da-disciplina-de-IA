"""Etapa 5 — Regressão com MLP (alvo: BMI).

Treina a MLP de regressão (reaproveitando :mod:`src.train`), avalia com
MAE/MSE/RMSE/R², gera os gráficos (real×predito, resíduos, curva de loss) e,
como extra, compara com modelos clássicos (Linear/Ridge/Lasso/RF/XGBoost).

Convenções de saída:
- Figuras   -> outputs/figures/etapa5_*.png
- Métricas  -> outputs/metrics/regression_metrics.json
- Modelo    -> outputs/models/mlp_regression.keras

Execução:
    python -m src.regression_eval
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")  # backend não interativo (salva figuras sem Tk)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import config
from src import plotting


# --------------------------------------------------------------------------- #
# Métricas
# --------------------------------------------------------------------------- #
def evaluate_regression(model, X, y) -> dict:
    """MAE, MSE, RMSE e R² do modelo em (X, y)."""
    y_pred = np.asarray(model.predict(X, batch_size=512, verbose=0)).ravel()
    mse = float(mean_squared_error(y, y_pred))
    return {
        "mae": float(mean_absolute_error(y, y_pred)),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y, y_pred)),
    }


# --------------------------------------------------------------------------- #
# Gráficos
# --------------------------------------------------------------------------- #
def plot_learning_curve(history: dict, filename="etapa5_loss.png"):
    """Curva de aprendizado: loss (MSE) de treino × validação por época."""
    plotting.set_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history["loss"], label="treino")
    ax.plot(history["val_loss"], label="validação")
    ax.set_xlabel("Época")
    ax.set_ylabel("Loss (MSE)")
    ax.set_title("Curva de aprendizado (loss) — regressão (BMI)")
    ax.legend()
    return plotting.save_fig(fig, filename)


def plot_real_vs_pred(y_true, y_pred, filename="etapa5_real_vs_pred.png"):
    """Dispersão valores reais × preditos com a reta y = x."""
    plotting.set_style()
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, s=6, alpha=0.2, color="#2c7fb8", edgecolors="none")
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "--", color="red", label="y = x (ideal)")
    ax.set_xlabel("BMI real")
    ax.set_ylabel("BMI predito")
    ax.set_title("Real × Predito — regressão (BMI)")
    ax.legend(loc="upper left")
    return plotting.save_fig(fig, filename)


def plot_residuals(y_true, y_pred, filename="etapa5_residuos.png"):
    """Resíduos: (a) resíduo × predito e (b) histograma dos resíduos."""
    plotting.set_style()
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    resid = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].scatter(y_pred, resid, s=6, alpha=0.2, color="#2c7fb8", edgecolors="none")
    axes[0].axhline(0, color="red", linestyle="--")
    axes[0].set_xlabel("BMI predito")
    axes[0].set_ylabel("Resíduo (real − predito)")
    axes[0].set_title("Resíduo × predito")

    axes[1].hist(resid, bins=60, color="#2c7fb8")
    axes[1].axvline(0, color="red", linestyle="--")
    axes[1].set_xlabel("Resíduo")
    axes[1].set_ylabel("Frequência")
    axes[1].set_title(f"Histograma dos resíduos (μ={resid.mean():.2f}, σ={resid.std():.2f})")

    fig.suptitle("Análise de resíduos — regressão (BMI)", fontweight="bold")
    return plotting.save_fig(fig, filename)


# --------------------------------------------------------------------------- #
# Extra — comparação com modelos clássicos de regressão (Etapa 5.3)
# --------------------------------------------------------------------------- #
def _regressors() -> dict:
    """Instancia os regressores clássicos para comparação."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Lasso, LinearRegression, Ridge

    regs = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0, random_state=config.SEED),
        "Lasso": Lasso(alpha=0.001, random_state=config.SEED),
        "RandomForest": RandomForestRegressor(
            n_estimators=200, n_jobs=-1, random_state=config.SEED),
    }
    try:
        from xgboost import XGBRegressor

        regs["XGBoost"] = XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            n_jobs=-1, random_state=config.SEED)
    except ImportError:
        pass
    return regs


def evaluate_classical(data) -> pd.DataFrame:
    """Treina/avalia os regressores clássicos no teste, medindo tempos."""
    import time

    config.set_seeds(config.SEED)
    rows = []
    for name, model in _regressors().items():
        t0 = time.perf_counter()
        model.fit(data.X_train, data.y_train)
        train_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        y_pred = model.predict(data.X_test)
        infer_time = time.perf_counter() - t0

        mse = mean_squared_error(data.y_test, y_pred)
        rows.append({
            "modelo": name,
            "mae": round(float(mean_absolute_error(data.y_test, y_pred)), 4),
            "rmse": round(float(np.sqrt(mse)), 4),
            "r2": round(float(r2_score(data.y_test, y_pred)), 4),
            "tempo_treino_s": round(train_time, 2),
            "tempo_inferencia_s": round(infer_time, 4),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Orquestração da Etapa 5
# --------------------------------------------------------------------------- #
def run(with_classical: bool = True) -> dict:
    """Pipeline completo de regressão (Etapa 5) e salva `regression_metrics.json`."""
    from src import preprocessing, train

    print("\n=== Etapa 5 — Regressão com MLP (BMI) ===")
    data = preprocessing.prepare_data("regression")
    print(f"  features: {data.X_train.shape[1]} | alvo: {data.target}")

    # MLP baseline de regressão (dropout 0.2; mse/mae; Adam 1e-3).
    res = train.train_model(
        "regression", data.X_train, data.y_train, data.X_val, data.y_val,
        hidden_units=(64, 32), dropout=0.2, epochs=100, batch_size=256)
    print(f"  treino: {res.epochs} épocas em {res.train_time:.1f}s")

    model_path = config.MODELS_DIR / "mlp_regression.keras"
    res.model.save(model_path)

    val_metrics = evaluate_regression(res.model, data.X_val, data.y_val)
    test_metrics = evaluate_regression(res.model, data.X_test, data.y_test)

    # Figuras (no conjunto de teste).
    y_pred_test = np.asarray(
        res.model.predict(data.X_test, batch_size=512, verbose=0)).ravel()
    plot_learning_curve(res.history)
    plot_real_vs_pred(data.y_test, y_pred_test)
    plot_residuals(data.y_test, y_pred_test)

    result = {
        "task": "regression",
        "target": data.target,
        "n_features": int(data.X_train.shape[1]),
        "epochs": res.epochs,
        "train_time_s": round(res.train_time, 2),
        "validation": val_metrics,
        "test": test_metrics,
    }

    if with_classical:
        print("  comparando com modelos clássicos...")
        comp = evaluate_classical(data)
        comp.to_csv(config.METRICS_DIR / "classical_comparison_regression.csv", index=False)
        result["classical_comparison"] = comp.to_dict(orient="records")
        print(comp.to_string(index=False))

    path = config.METRICS_DIR / "regression_metrics.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)

    print(f"  TESTE -> MAE={test_metrics['mae']:.3f} RMSE={test_metrics['rmse']:.3f} "
          f"R²={test_metrics['r2']:.4f}")
    print(f"Métricas salvas em {path}")
    print("Etapa 5 concluída.")
    return result


if __name__ == "__main__":
    run()
