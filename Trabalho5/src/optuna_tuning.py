"""Etapa 6 — Otimização de hiperparâmetros com Optuna.

Otimiza a MLP de cada tarefa (prioridade: binária). Usa `MedianPruner` com um
callback de pruning por época para descartar trials ruins cedo e, ao final,
**retreina** o modelo com os melhores hiperparâmetros e o avalia **uma vez**
no teste.

Espaço de busca (justificado no relatório):
- n_layers: 1–4
- units_l{i}: {32, 64, 128, 256}
- activation: {relu, tanh, elu}
- dropout: 0.0–0.5
- learning_rate: log-uniforme 1e-4 … 1e-2
- optimizer: {adam, rmsprop, sgd} (+ momentum 0.0–0.9 se sgd)
- batch_size: {64, 128, 256, 512}
- l2 (weight decay): log-uniforme 1e-6 … 1e-2

Função objetivo (na validação):
- binária   -> maximizar ROC-AUC
- multiclasse -> maximizar F1 macro
- regressão -> minimizar RMSE

Entregáveis (outputs/):
- best_params_<tarefa>.json, optuna_study_<tarefa>.pkl
- etapa6_optuna_history_<tarefa>.png, etapa6_optuna_param_importance_<tarefa>.png
- optuna_comparison.csv (original × otimizado, no teste)

Execução:
    python -m src.optuna_tuning binary --trials 40
    python -m src.optuna_tuning                      # todas, 40 trials
"""
from __future__ import annotations

import argparse
import json

import matplotlib
matplotlib.use("Agg")  # backend não interativo (salva figuras sem Tk)
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import f1_score, mean_squared_error, roc_auc_score

import config
from src import plotting, preprocessing, train

# Silencia o log verboso do Optuna (mantém só o resumo impresso por nós).
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Direção da otimização e métrica intermediária (para pruning) por tarefa.
STUDY_DIRECTION = {"binary": "maximize", "multiclass": "maximize", "regression": "minimize"}
PRUNE_MONITOR = {"binary": "val_auc", "multiclass": "val_accuracy", "regression": "val_loss"}
OBJ_METRIC = {"binary": "ROC-AUC", "multiclass": "F1 macro", "regression": "RMSE"}


# --------------------------------------------------------------------------- #
# Callback de pruning (reporta a métrica de validação ao trial a cada época)
# --------------------------------------------------------------------------- #
def _make_pruning_callback(trial, monitor):
    from tensorflow import keras

    class _PruningCallback(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            value = (logs or {}).get(monitor)
            if value is None:
                return
            trial.report(float(value), step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

    return _PruningCallback()


# --------------------------------------------------------------------------- #
# Avaliação da métrica-objetivo na validação
# --------------------------------------------------------------------------- #
def _objective_metric(task: str, model, X, y) -> float:
    pred = model.predict(X, batch_size=512, verbose=0)
    if task == "binary":
        return float(roc_auc_score(y, pred.ravel()))
    if task == "multiclass":
        return float(f1_score(y, pred.argmax(axis=1), average="macro"))
    return float(np.sqrt(mean_squared_error(y, pred.ravel())))  # RMSE


# --------------------------------------------------------------------------- #
# Sugestão de hiperparâmetros + treino de um trial
# --------------------------------------------------------------------------- #
def _suggest_params(trial) -> dict:
    n_layers = trial.suggest_int("n_layers", 1, 4)
    hidden_units = tuple(
        trial.suggest_categorical(f"units_l{i}", [32, 64, 128, 256])
        for i in range(n_layers)
    )
    optimizer = trial.suggest_categorical("optimizer", ["adam", "rmsprop", "sgd"])
    return {
        "hidden_units": hidden_units,
        "activation": trial.suggest_categorical("activation", ["relu", "tanh", "elu"]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        "optimizer": optimizer,
        "momentum": trial.suggest_float("momentum", 0.0, 0.9) if optimizer == "sgd" else 0.0,
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256, 512]),
        "l2": trial.suggest_float("l2", 1e-6, 1e-2, log=True),
    }


def _make_objective(task: str, data, epochs: int):
    n_classes = 3 if task == "multiclass" else 2

    def objective(trial):
        params = _suggest_params(trial)
        pruning_cb = _make_pruning_callback(trial, PRUNE_MONITOR[task])
        res = train.train_model(
            task, data.X_train, data.y_train, data.X_val, data.y_val,
            hidden_units=params["hidden_units"], dropout=params["dropout"],
            activation=params["activation"], learning_rate=params["learning_rate"],
            optimizer=params["optimizer"], momentum=params["momentum"], l2=params["l2"],
            batch_size=params["batch_size"], epochs=epochs, n_classes=n_classes,
            extra_callbacks=[pruning_cb], verbose=0,
        )
        return _objective_metric(task, res.model, data.X_val, data.y_val)

    return objective


# --------------------------------------------------------------------------- #
# Figuras do Optuna
# --------------------------------------------------------------------------- #
def _save_optuna_figures(study, task: str):
    from optuna.visualization.matplotlib import (
        plot_optimization_history, plot_param_importances,
    )

    plotting.set_style()
    ax = plot_optimization_history(study)
    fig = ax.figure
    fig.set_size_inches(9, 5)
    plotting.save_fig(fig, f"etapa6_optuna_history_{task}.png")

    try:
        ax = plot_param_importances(study)
        fig = ax.figure
        fig.set_size_inches(9, 6)
        plotting.save_fig(fig, f"etapa6_optuna_param_importance_{task}.png")
    except (RuntimeError, ValueError) as exc:
        # Importância exige trials concluídos suficientes; não é crítico.
        print(f"  (importância de hiperparâmetros indisponível: {exc})")


# --------------------------------------------------------------------------- #
# Reconstrução dos kwargs de build a partir do best_params
# --------------------------------------------------------------------------- #
def _params_to_build_kwargs(params: dict) -> dict:
    n_layers = params["n_layers"]
    return {
        "hidden_units": tuple(params[f"units_l{i}"] for i in range(n_layers)),
        "activation": params["activation"],
        "dropout": params["dropout"],
        "learning_rate": params["learning_rate"],
        "optimizer": params["optimizer"],
        "momentum": params.get("momentum", 0.0),
        "l2": params["l2"],
        "batch_size": params["batch_size"],
    }


# --------------------------------------------------------------------------- #
# Baseline (Etapas 4/5) para a comparação original × otimizado
# --------------------------------------------------------------------------- #
def _baseline_test_metric(task: str):
    """Lê a métrica-objetivo e o tempo de treino do baseline salvo (Etapas 4/5)."""
    if task == "regression":
        path = config.METRICS_DIR / "regression_metrics.json"
        if not path.exists():
            return None, None
        d = json.load(open(path, encoding="utf-8"))
        return d["test"]["rmse"], d.get("train_time_s")
    path = config.METRICS_DIR / "classification_metrics.json"
    if not path.exists():
        return None, None
    d = json.load(open(path, encoding="utf-8"))
    key = "roc_auc" if task == "binary" else "f1_macro"
    return d[task]["test"][key], d[task].get("train_time_s")


def _optimized_test_metric(task: str, model, data) -> float:
    return _objective_metric(task, model, data.X_test, data.y_test)


# --------------------------------------------------------------------------- #
# Orquestração de uma tarefa
# --------------------------------------------------------------------------- #
def tune_task(task: str, n_trials: int = 40, epochs: int = 100) -> dict:
    """Roda o estudo Optuna, retreina com os melhores params e avalia no teste."""
    print(f"\n=== Etapa 6 — Optuna ({task}) | {n_trials} trials | objetivo: "
          f"{OBJ_METRIC[task]} ({STUDY_DIRECTION[task]}) ===")
    config.set_seeds(config.SEED)
    data = preprocessing.prepare_data(task)

    sampler = optuna.samplers.TPESampler(seed=config.SEED)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    study = optuna.create_study(
        direction=STUDY_DIRECTION[task], sampler=sampler, pruner=pruner)

    study.optimize(_make_objective(task, data, epochs), n_trials=n_trials,
                   show_progress_bar=True)

    n_pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
    print(f"  trials concluídos={n_trials} (podados={n_pruned}) | "
          f"melhor {OBJ_METRIC[task]} (val)={study.best_value:.4f}")

    # Persiste study, best_params e figuras.
    best_params = dict(study.best_params)
    with open(config.MODELS_DIR / f"best_params_{task}.json", "w", encoding="utf-8") as fh:
        json.dump({"objetivo": OBJ_METRIC[task], "direcao": STUDY_DIRECTION[task],
                   "valor_validacao": study.best_value, "n_trials": n_trials,
                   "n_podados": n_pruned, "best_params": best_params},
                  fh, ensure_ascii=False, indent=2)
    import joblib
    joblib.dump(study, config.MODELS_DIR / f"optuna_study_{task}.pkl")
    _save_optuna_figures(study, task)

    # Retreina com os melhores hiperparâmetros e avalia UMA vez no teste.
    print("  retreinando o modelo final com os melhores hiperparâmetros...")
    bk = _params_to_build_kwargs(best_params)
    n_classes = 3 if task == "multiclass" else 2
    res = train.train_model(
        task, data.X_train, data.y_train, data.X_val, data.y_val,
        hidden_units=bk["hidden_units"], dropout=bk["dropout"],
        activation=bk["activation"], learning_rate=bk["learning_rate"],
        optimizer=bk["optimizer"], momentum=bk["momentum"], l2=bk["l2"],
        batch_size=bk["batch_size"], epochs=epochs, n_classes=n_classes, verbose=0)
    res.model.save(config.MODELS_DIR / f"mlp_{task}_optimized.keras")

    opt_test = _optimized_test_metric(task, res.model, data)
    base_test, base_time = _baseline_test_metric(task)

    print(f"  TESTE -> baseline {OBJ_METRIC[task]}={base_test} | "
          f"otimizado={opt_test:.4f}")

    return {
        "tarefa": task,
        "metrica": OBJ_METRIC[task],
        "baseline_teste": base_test,
        "otimizado_teste": round(opt_test, 4),
        "delta": round(opt_test - base_test, 4) if base_test is not None else None,
        "baseline_tempo_treino_s": base_time,
        "otimizado_tempo_treino_s": round(res.train_time, 2),
        "n_trials": n_trials,
        "n_podados": n_pruned,
    }


# --------------------------------------------------------------------------- #
# Orquestração geral + tabela comparativa
# --------------------------------------------------------------------------- #
def run(tasks=("binary", "multiclass", "regression"), n_trials: int = 40,
        epochs: int = 100):
    rows = [tune_task(t, n_trials=n_trials, epochs=epochs) for t in tasks]
    comparison = pd.DataFrame(rows)
    comparison.to_csv(config.METRICS_DIR / "optuna_comparison.csv", index=False)
    print("\nComparação original × otimizado salva em outputs/metrics/optuna_comparison.csv")
    print(comparison.to_string(index=False))
    print("\nEtapa 6 concluída.")
    return comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Etapa 6 — Optuna HPO")
    parser.add_argument("tasks", nargs="*",
                        default=["binary", "multiclass", "regression"])
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()
    run(tuple(args.tasks), n_trials=args.trials, epochs=args.epochs)
