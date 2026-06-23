"""Etapa 7 — Regularização e overfitting (tarefa de classificação).

Compara, na tarefa binária, duas redes de **mesma arquitetura**:

- **SEM regularização:** sem dropout, sem L2 e **sem early stopping**, treinada
  por muitas épocas para induzir overfitting.
- **COM regularização:** **Dropout + L2 (weight decay) + Early Stopping**
  combinados.

Para tornar o overfitting visível (o dataset completo tem ~152k amostras de
treino, o que naturalmente regulariza), o treino usa uma **subamostra** do
conjunto de treino e uma rede de maior capacidade. Val e test continuam
completos — a avaliação final no teste não é afetada por leakage.

Entregáveis (outputs/):
- etapa7_curvas_comparacao.png  (curvas treino×val lado a lado)
- regularization_comparison.csv (métricas finais treino vs teste)

Execução:
    python -m src.regularization
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # backend não interativo (salva figuras sem Tk)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from src import evaluation, plotting, preprocessing, train

# Arquitetura de maior capacidade (favorece overfitting sem regularização).
HIDDEN_UNITS = (256, 128, 64)
# Subamostra de treino para tornar o overfitting observável.
TRAIN_SUBSAMPLE = 8_000
EPOCHS = 200
L2 = 1e-3
DROPOUT = 0.4


def _subsample(X, y, n, seed=config.SEED):
    """Subamostra estratificada simples do conjunto de treino."""
    if n is None or n >= len(y):
        return X, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y), size=n, replace=False)
    return X[idx], y[idx]


def _loss(model, X, y) -> float:
    """Loss (binary crossentropy) do modelo em (X, y)."""
    return float(model.evaluate(X, y, verbose=0)[0])


def _collect(task, model, data, history) -> dict:
    """Métricas finais de uma rede: loss e ROC-AUC/F1 em treino/val/teste."""
    m_train = evaluation.evaluate_classification(task, model, data._Xtr, data._ytr)
    m_test = evaluation.evaluate_classification(task, model, data.X_test, data.y_test)
    return {
        "epocas": len(history["loss"]),
        "loss_treino": round(_loss(model, data._Xtr, data._ytr), 4),
        "loss_val": round(_loss(model, data.X_val, data.y_val), 4),
        "loss_teste": round(_loss(model, data.X_test, data.y_test), 4),
        "roc_auc_treino": round(m_train["roc_auc"], 4),
        "roc_auc_teste": round(m_test["roc_auc"], 4),
        "f1_treino": round(m_train["f1"], 4),
        "f1_teste": round(m_test["f1"], 4),
        "gap_roc_auc_treino_teste": round(m_train["roc_auc"] - m_test["roc_auc"], 4),
    }


def plot_curves(hist_no_reg, hist_reg, filename="etapa7_curvas_comparacao.png"):
    """Curvas de loss treino×val das duas redes, lado a lado."""
    plotting.set_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, hist, titulo in [
        (axes[0], hist_no_reg, "SEM regularização"),
        (axes[1], hist_reg, "COM regularização (Dropout + L2 + EarlyStopping)"),
    ]:
        ax.plot(hist["loss"], label="treino")
        ax.plot(hist["val_loss"], label="validação")
        ax.set_xlabel("Época")
        ax.set_ylabel("Loss")
        ax.set_title(titulo)
        ax.legend()
    fig.suptitle("Etapa 7 — Overfitting: curvas treino × validação", fontweight="bold")
    return plotting.save_fig(fig, filename)


def run(task: str = "binary"):
    """Executa o experimento de regularização para a tarefa de classificação."""
    print(f"\n=== Etapa 7 — Regularização e overfitting ({task}) ===")
    data = preprocessing.prepare_data(task)

    # Subamostra de treino (anexada ao objeto para reuso nas avaliações).
    data._Xtr, data._ytr = _subsample(data.X_train, data.y_train, TRAIN_SUBSAMPLE)
    print(f"  treino (subamostra)={data._Xtr.shape[0]} | val={data.X_val.shape[0]} | "
          f"teste={data.X_test.shape[0]}")

    n_classes = 3 if task == "multiclass" else 2

    print("  treinando rede SEM regularização (sem dropout/L2/early stopping)...")
    res_no = train.train_model(
        task, data._Xtr, data._ytr, data.X_val, data.y_val,
        hidden_units=HIDDEN_UNITS, dropout=0.0, l2=0.0,
        use_early_stopping=False, epochs=EPOCHS, n_classes=n_classes)

    print("  treinando rede COM regularização (Dropout + L2 + Early Stopping)...")
    res_reg = train.train_model(
        task, data._Xtr, data._ytr, data.X_val, data.y_val,
        hidden_units=HIDDEN_UNITS, dropout=DROPOUT, l2=L2,
        use_early_stopping=True, patience=10, epochs=EPOCHS, n_classes=n_classes)

    no_reg = _collect(task, res_no.model, data, res_no.history)
    reg = _collect(task, res_reg.model, data, res_reg.history)

    plot_curves(res_no.history, res_reg.history)

    comparison = pd.DataFrame([
        {"rede": "sem_regularizacao", "dropout": 0.0, "l2": 0.0,
         "early_stopping": False, **no_reg},
        {"rede": "com_regularizacao", "dropout": DROPOUT, "l2": L2,
         "early_stopping": True, **reg},
    ])
    comparison.to_csv(config.METRICS_DIR / "regularization_comparison.csv", index=False)

    print("\n" + comparison.to_string(index=False))
    print("\nComparação salva em outputs/metrics/regularization_comparison.csv")
    print("Etapa 7 concluída.")
    return comparison


if __name__ == "__main__":
    run()
