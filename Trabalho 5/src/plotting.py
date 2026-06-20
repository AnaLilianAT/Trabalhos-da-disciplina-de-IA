"""Helpers de gráficos com estilo consistente para todas as etapas.

Centraliza o estilo do seaborn/matplotlib e o salvamento das figuras em
``outputs/figures/`` com nomes padronizados e alta resolução.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

import config


def set_style() -> None:
    """Aplica um estilo visual consistente a todas as figuras."""
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams["figure.dpi"] = 110
    plt.rcParams["savefig.dpi"] = config.FIG_DPI
    plt.rcParams["savefig.bbox"] = "tight"
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["font.size"] = 10


def save_fig(fig: plt.Figure, filename: str) -> Path:
    """Salva uma figura em ``outputs/figures/`` e a fecha.

    Parameters
    ----------
    fig:
        Figura do matplotlib a salvar.
    filename:
        Nome do arquivo (ex.: ``"etapa1_correlacao.png"``).

    Returns
    -------
    pathlib.Path
        Caminho do arquivo salvo.
    """
    path = config.FIGURES_DIR / filename
    fig.savefig(path, dpi=config.FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return path
