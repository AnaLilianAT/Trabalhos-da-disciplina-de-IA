"""Rotinas de treino das MLPs + captura do histórico.

Centraliza o `model.fit` usado pelas Etapas 4 (classificação), 5 (regressão),
6 (Optuna) e 7 (regularização), aplicando os padrões do plano: `EarlyStopping`
com `restore_best_weights`, `class_weight='balanced'` nas classificações e
seeds fixas para reprodutibilidade.

Execução de sanidade:
    python -m src.train binary
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass

import numpy as np

import config
from src import models, preprocessing


@dataclass
class TrainResult:
    """Modelo treinado, histórico de treino e tempo de parede (segundos)."""
    model: object
    history: dict
    train_time: float
    epochs: int


def train_model(task: str,
                X_train: np.ndarray, y_train: np.ndarray,
                X_val: np.ndarray, y_val: np.ndarray,
                hidden_units=(64, 32),
                dropout: float | None = None,
                activation: str = "relu",
                learning_rate: float = 1e-3,
                optimizer: str = "adam",
                momentum: float = 0.0,
                l2: float = 0.0,
                epochs: int = 100,
                batch_size: int = 256,
                use_class_weight: bool = True,
                use_early_stopping: bool = True,
                patience: int = 10,
                n_classes: int = 3,
                extra_callbacks=None,
                verbose: int = 0) -> TrainResult:
    """Constrói e treina uma MLP, devolvendo modelo, histórico e tempo.

    Parameters
    ----------
    task:
        ``"binary"``, ``"multiclass"`` ou ``"regression"``.
    use_class_weight:
        Aplica `class_weight='balanced'` (ignorado na regressão).
    use_early_stopping:
        Liga o `EarlyStopping(monitor='val_loss', restore_best_weights=True)`.
        Desligado na Etapa 7 para induzir overfitting na rede sem regularização.

    As demais opções são repassadas a :func:`src.models.build_mlp` e expõem o
    espaço de busca da Etapa 6.
    """
    config.set_seeds(config.SEED)
    model = models.build_mlp(
        task, X_train.shape[1],
        hidden_units=hidden_units, dropout=dropout, activation=activation,
        learning_rate=learning_rate, optimizer=optimizer, momentum=momentum,
        l2=l2, n_classes=n_classes,
    )

    class_weight = None
    if use_class_weight and task != "regression":
        class_weight = preprocessing.compute_class_weights(y_train)

    from tensorflow import keras
    callbacks = []
    if use_early_stopping:
        callbacks.append(keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True))
    if extra_callbacks:
        callbacks.extend(extra_callbacks)

    t0 = time.perf_counter()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs, batch_size=batch_size,
        class_weight=class_weight, callbacks=callbacks, verbose=verbose,
    )
    train_time = time.perf_counter() - t0

    return TrainResult(
        model=model,
        history=history.history,
        train_time=train_time,
        epochs=len(history.history["loss"]),
    )


if __name__ == "__main__":
    _task = sys.argv[1] if len(sys.argv) > 1 else "binary"
    _data = preprocessing.prepare_data(_task)
    _res = train_model(
        _task, _data.X_train, _data.y_train, _data.X_val, _data.y_val, verbose=0)
    print(f"{_task}: {_res.epochs} épocas em {_res.train_time:.1f}s")
    print("últimas métricas:", {k: round(v[-1], 4) for k, v in _res.history.items()})
