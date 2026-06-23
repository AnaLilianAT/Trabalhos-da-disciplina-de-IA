"""Construção das MLPs (Keras) para as três tarefas.

Fornece um ``build_mlp`` parametrizável usado pela seleção de features
(Etapa 3, comparação todas×selecionadas), pela classificação/regressão
(Etapas 4 e 5), pela otimização com Optuna (Etapa 6) e pelos experimentos
de regularização (Etapa 7).
"""
from __future__ import annotations

import os

# Reduz o ruído de log do TensorFlow (mensagens informativas de oneDNN etc.).
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from tensorflow import keras
from tensorflow.keras import layers, regularizers


# Configuração por tarefa: dimensão/ativação de saída, perda e métricas.
def _output_config(task: str, n_classes: int):
    if task == "binary":
        return 1, "sigmoid", "binary_crossentropy", [
            keras.metrics.AUC(name="auc"),
            keras.metrics.BinaryAccuracy(name="accuracy"),
        ]
    if task == "multiclass":
        return n_classes, "softmax", "sparse_categorical_crossentropy", ["accuracy"]
    if task == "regression":
        return 1, "linear", "mse", ["mae"]
    raise ValueError(f"Tarefa desconhecida: {task!r}")


def _make_optimizer(name: str, learning_rate: float, momentum: float = 0.0):
    name = name.lower()
    if name == "adam":
        return keras.optimizers.Adam(learning_rate=learning_rate)
    if name == "rmsprop":
        return keras.optimizers.RMSprop(learning_rate=learning_rate)
    if name == "sgd":
        return keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
    raise ValueError(f"Otimizador desconhecido: {name!r}")


def build_mlp(task: str,
              input_dim: int,
              hidden_units=(64, 32),
              dropout: float | None = None,
              activation: str = "relu",
              learning_rate: float = 1e-3,
              optimizer: str = "adam",
              momentum: float = 0.0,
              l2: float = 0.0,
              n_classes: int = 3) -> keras.Model:
    """Cria e compila uma MLP para a tarefa indicada.

    Arquitetura baseline (justificada no relatório):
    ``Input -> [Dense(units, act) -> Dropout]* -> Dense(saída)``.

    Parameters
    ----------
    task:
        ``"binary"``, ``"multiclass"`` ou ``"regression"``.
    input_dim:
        Nº de features de entrada.
    hidden_units:
        Tupla com o nº de neurônios de cada camada oculta (baseline: 64, 32).
    dropout:
        Taxa de dropout aplicada após cada camada oculta. Se ``None``, usa
        0.3 para classificação e 0.2 para regressão (baseline do plano).
    activation:
        Ativação das camadas ocultas (ReLU no baseline).
    learning_rate, optimizer, momentum, l2:
        Hiperparâmetros de treino/regularização (expostos para a Etapa 6).
    n_classes:
        Nº de classes (apenas multiclasse).
    """
    if dropout is None:
        dropout = 0.2 if task == "regression" else 0.3

    reg = regularizers.l2(l2) if l2 else None
    units_out, out_activation, loss, metrics = _output_config(task, n_classes)

    model = keras.Sequential(name=f"mlp_{task}")
    model.add(keras.Input(shape=(input_dim,)))
    for units in hidden_units:
        model.add(layers.Dense(units, activation=activation, kernel_regularizer=reg))
        if dropout and dropout > 0:
            model.add(layers.Dropout(dropout))
    model.add(layers.Dense(units_out, activation=out_activation))

    model.compile(
        optimizer=_make_optimizer(optimizer, learning_rate, momentum),
        loss=loss,
        metrics=metrics,
    )
    return model
