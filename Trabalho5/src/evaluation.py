"""Avaliação das classificações (Etapa 4): métricas, matrizes e curvas.

Funções de métrica para as MLPs binária e multiclasse e helpers de gráfico
(curva de aprendizado, métrica por época, matriz de confusão normalizada e
curva ROC). As métricas de regressão (Etapa 5) ficam em módulo próprio.

Convenções de saída:
- Figuras   -> outputs/figures/etapa4_*.png
- Métricas  -> outputs/metrics/classification_metrics.json
"""
from __future__ import annotations

import json
import sys

import matplotlib
matplotlib.use("Agg")  # backend não interativo (salva figuras sem Tk)
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score, roc_curve,
)

import config
from src import plotting


# --------------------------------------------------------------------------- #
# Predição
# --------------------------------------------------------------------------- #
def predict_proba(model, X, batch_size: int = 512) -> np.ndarray:
    """Saída bruta do modelo (probabilidades)."""
    return model.predict(X, batch_size=batch_size, verbose=0)


def predict_labels(task: str, proba: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Converte probabilidades em rótulos preditos."""
    if task == "binary":
        return (proba.ravel() >= threshold).astype(int)
    return proba.argmax(axis=1)


# --------------------------------------------------------------------------- #
# Métricas
# --------------------------------------------------------------------------- #
def evaluate_classification(task: str, model, X, y) -> dict:
    """Calcula as métricas da tarefa em (X, y).

    Binária: accuracy, precision, recall, F1, ROC-AUC, PR-AUC.
    Multiclasse: accuracy, precision/recall/F1 macro.
    Inclui a matriz de confusão (contagens) em ambos os casos.
    """
    proba = predict_proba(model, X)
    y_pred = predict_labels(task, proba)

    if task == "binary":
        scores = proba.ravel()
        metrics = {
            "accuracy": float(accuracy_score(y, y_pred)),
            "precision": float(precision_score(y, y_pred, zero_division=0)),
            "recall": float(recall_score(y, y_pred, zero_division=0)),
            "f1": float(f1_score(y, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y, scores)),
            "pr_auc": float(average_precision_score(y, scores)),
        }
    else:
        metrics = {
            "accuracy": float(accuracy_score(y, y_pred)),
            "precision_macro": float(precision_score(y, y_pred, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y, y_pred, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(y, y_pred, average="macro", zero_division=0)),
        }

    metrics["confusion_matrix"] = confusion_matrix(y, y_pred).tolist()
    return metrics


# --------------------------------------------------------------------------- #
# Gráficos
# --------------------------------------------------------------------------- #
def plot_learning_curve(history: dict, task: str):
    """Curva de aprendizado: loss de treino × validação por época."""
    plotting.set_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history["loss"], label="treino")
    ax.plot(history["val_loss"], label="validação")
    ax.set_xlabel("Época")
    ax.set_ylabel("Loss")
    ax.set_title(f"Curva de aprendizado (loss) — {task}")
    ax.legend()
    return plotting.save_fig(fig, f"etapa4_loss_{task}.png")


def plot_metric_curve(history: dict, task: str):
    """Evolução da métrica principal por época (AUC binária / accuracy multiclasse)."""
    plotting.set_style()
    key = "auc" if task == "binary" else "accuracy"
    label = "ROC-AUC" if task == "binary" else "Accuracy"
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history[key], label="treino")
    ax.plot(history[f"val_{key}"], label="validação")
    ax.set_xlabel("Época")
    ax.set_ylabel(label)
    ax.set_title(f"{label} por época — {task}")
    ax.legend()
    return plotting.save_fig(fig, f"etapa4_metric_{task}.png")


def plot_confusion_matrix(cm, task: str, normalized: bool = True):
    """Matriz de confusão (normalizada por linha) com rótulos legíveis."""
    import seaborn as sns

    plotting.set_style()
    cm = np.asarray(cm, dtype=float)
    if normalized:
        cm = cm / cm.sum(axis=1, keepdims=True)

    labels_map = (config.CLASS_LABELS_BINARY if task == "binary"
                  else config.CLASS_LABELS_MULTICLASS)
    labels = [labels_map[i] for i in range(len(labels_map))]

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt=".2f" if normalized else ".0f",
                cmap="Blues", xticklabels=labels, yticklabels=labels,
                cbar=True, ax=ax)
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    title = "Matriz de confusão" + (" (normalizada)" if normalized else "")
    ax.set_title(f"{title} — {task}")
    return plotting.save_fig(fig, f"etapa4_confmat_{task}.png")


def plot_roc_curve(y_true, scores, filename="etapa4_roc_binaria.png"):
    """Curva ROC da tarefa binária."""
    plotting.set_style()
    fpr, tpr, _ = roc_curve(y_true, scores)
    auc = roc_auc_score(y_true, scores)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, label=f"MLP (AUC = {auc:.3f})", color="#2c7fb8")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="aleatório")
    ax.set_xlabel("Taxa de falso positivo")
    ax.set_ylabel("Taxa de verdadeiro positivo")
    ax.set_title("Curva ROC — binária")
    ax.legend(loc="lower right")
    return plotting.save_fig(fig, filename)


# --------------------------------------------------------------------------- #
# Orquestração da Etapa 4
# --------------------------------------------------------------------------- #
def run_classification(task: str, with_classical: bool = True) -> dict:
    """Pipeline completo de classificação para uma tarefa (Etapa 4).

    Treina a MLP baseline, salva o modelo, avalia em validação e **uma vez**
    no teste, gera as figuras (loss, métrica, matriz de confusão, ROC) e,
    opcionalmente, compara com os modelos clássicos.
    """
    from src import preprocessing, train

    print(f"\n=== Etapa 4 — Classificação ({task}) ===")
    data = preprocessing.prepare_data(task)

    # Treino da MLP baseline (arquitetura/épocas do plano).
    dropout = 0.3
    res = train.train_model(
        task, data.X_train, data.y_train, data.X_val, data.y_val,
        hidden_units=(64, 32), dropout=dropout, epochs=100, batch_size=256)
    print(f"  treino: {res.epochs} épocas em {res.train_time:.1f}s")

    # Salva o modelo final (.keras).
    model_path = config.MODELS_DIR / f"mlp_{task}.keras"
    res.model.save(model_path)

    # Avaliação em validação e teste (teste uma única vez).
    val_metrics = evaluate_classification(task, res.model, data.X_val, data.y_val)
    test_metrics = evaluate_classification(task, res.model, data.X_test, data.y_test)

    # Figuras.
    plot_learning_curve(res.history, task)
    plot_metric_curve(res.history, task)
    plot_confusion_matrix(test_metrics["confusion_matrix"], task, normalized=True)
    if task == "binary":
        scores = predict_proba(res.model, data.X_test).ravel()
        plot_roc_curve(data.y_test, scores)

    result = {
        "task": task,
        "n_features": int(data.X_train.shape[1]),
        "epochs": res.epochs,
        "train_time_s": round(res.train_time, 2),
        "validation": val_metrics,
        "test": test_metrics,
    }

    if with_classical:
        from src import classical_models
        print("  comparando com modelos clássicos...")
        comp = classical_models.evaluate_classical(task, data)
        comp.to_csv(config.METRICS_DIR / f"classical_comparison_{task}.csv", index=False)
        result["classical_comparison"] = comp.to_dict(orient="records")
        print(comp.to_string(index=False))

    _print_summary(task, test_metrics)
    return result


def _print_summary(task: str, m: dict) -> None:
    if task == "binary":
        print(f"  TESTE -> acc={m['accuracy']:.4f} f1={m['f1']:.4f} "
              f"roc_auc={m['roc_auc']:.4f} pr_auc={m['pr_auc']:.4f}")
    else:
        print(f"  TESTE -> acc={m['accuracy']:.4f} f1_macro={m['f1_macro']:.4f} "
              f"recall_macro={m['recall_macro']:.4f}")


def run(tasks=("binary", "multiclass")) -> dict:
    """Executa a Etapa 4 para as tarefas e salva `classification_metrics.json`."""
    results = {task: run_classification(task) for task in tasks}
    path = config.METRICS_DIR / "classification_metrics.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)
    print(f"\nMétricas salvas em {path}")
    print("Etapa 4 concluída.")
    return results


if __name__ == "__main__":
    _tasks = tuple(sys.argv[1:]) or ("binary", "multiclass")
    run(_tasks)
