"""Extra — comparação da MLP com modelos clássicos.

Etapa 4.3: treina `LogisticRegression`, `RandomForestClassifier` e (extra)
`XGBClassifier` nas mesmas features e compara métricas + custo computacional
(tempo de treino e de inferência) com a MLP.

A comparação de regressão (Etapa 5.3) fica em :mod:`src.regression_eval`.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, f1_score, roc_auc_score,
)

import config


def _classifiers(task: str) -> dict:
    """Instancia os classificadores clássicos (com `class_weight` quando aplicável)."""
    clfs = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=config.SEED),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced",
            n_jobs=-1, random_state=config.SEED),
    }
    try:
        from xgboost import XGBClassifier

        kwargs = dict(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            n_jobs=-1, random_state=config.SEED, eval_metric="logloss",
        )
        if task == "multiclass":
            kwargs.update(objective="multi:softprob", num_class=3)
        else:
            # Compensa o desbalanceamento na binária.
            kwargs.update(objective="binary:logistic")
        clfs["XGBoost"] = XGBClassifier(**kwargs)
    except ImportError:
        pass
    return clfs


def _scores(task: str, model, X) -> np.ndarray:
    """Probabilidades para as métricas baseadas em score (ROC/PR-AUC)."""
    proba = model.predict_proba(X)
    return proba[:, 1] if task == "binary" else proba


def evaluate_classical(task: str, data, sample_weight_balanced: bool = True) -> pd.DataFrame:
    """Treina e avalia os modelos clássicos no teste, medindo tempos.

    Métrica principal: ROC-AUC (binária) / F1 macro (multiclasse). Reporta
    também o tempo de treino e o tempo de inferência no conjunto de teste.
    """
    config.set_seeds(config.SEED)
    rows = []
    for name, model in _classifiers(task).items():
        fit_kwargs = {}
        # XGBoost não tem class_weight; usa sample_weight balanceado na binária.
        if name == "XGBoost" and task == "binary" and sample_weight_balanced:
            from sklearn.utils.class_weight import compute_sample_weight
            fit_kwargs["sample_weight"] = compute_sample_weight("balanced", data.y_train)

        t0 = time.perf_counter()
        model.fit(data.X_train, data.y_train, **fit_kwargs)
        train_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        y_pred = model.predict(data.X_test)
        infer_time = time.perf_counter() - t0
        scores = _scores(task, model, data.X_test)

        if task == "binary":
            row = {
                "modelo": name,
                "roc_auc": round(float(roc_auc_score(data.y_test, scores)), 4),
                "pr_auc": round(float(average_precision_score(data.y_test, scores)), 4),
                "f1": round(float(f1_score(data.y_test, y_pred, zero_division=0)), 4),
            }
        else:
            row = {
                "modelo": name,
                "f1_macro": round(float(f1_score(data.y_test, y_pred, average="macro", zero_division=0)), 4),
            }
        row["tempo_treino_s"] = round(train_time, 2)
        row["tempo_inferencia_s"] = round(infer_time, 4)
        rows.append(row)

    return pd.DataFrame(rows)
