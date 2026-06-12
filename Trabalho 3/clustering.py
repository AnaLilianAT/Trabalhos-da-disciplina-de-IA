"""
Trabalho de Clustering — Inteligencia Artificial (UFSM)
Dataset: CDC Diabetes Health Indicators (BRFSS2015), versao 50-50 (70.692 linhas).

==============================================================================
DEFINICAO DO PROBLEMA
==============================================================================
Ao remover o diagnostico de diabetes, os algoritmos de clustering conseguem
separar os respondentes do BRFSS em grupos com perfis de risco distintos
(ex.: grupo saudavel vs. grupo de alto risco metabolico)? Os clusters
encontrados se alinham ao rotulo original (Diabetes_binary)? Variaveis como
pressao alta, IMC, saude geral e idade sao as que mais separam os grupos?
==============================================================================
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import linkage as scipy_linkage
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------
# Configuracao global
# --------------------------------------------------------------------------
RANDOM_STATE = 42  # usado em tudo que aceitar random_state

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
FIG_DIR = BASE_DIR / "figures"
RES_DIR = BASE_DIR / "results"
for _d in (DATA_DIR, FIG_DIR, RES_DIR):
    _d.mkdir(exist_ok=True)

DATASET_PATH = DATA_DIR / "dataset.csv"
ROTULO = "Diabetes_binary"

# Faixa plausivel de IMC para tratamento de outliers
BMI_MIN, BMI_MAX = 12.0, 60.0

# Tamanho da amostra para Agglomerative/DBSCAN (custo O(n^2))
SAMPLE_SIZE = 7000
# Subamostra usada no calculo do Silhouette (evita matriz de distancias gigante)
SIL_SAMPLE = 5000

# Tipologia das variaveis (decisiva para a analise e o pre-processamento)
BINARIAS = [
    "HighBP", "HighChol", "CholCheck", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "DiffWalk", "Sex",
]
CONTINUAS = ["BMI", "MentHlth", "PhysHlth"]
ORDINAIS = ["GenHlth", "Age", "Education", "Income"]
# Variaveis para histogramas/boxplots e correlacoes numericas
NUMERICAS = CONTINUAS + ORDINAIS
FEATURES = BINARIAS + CONTINUAS + ORDINAIS

sns.set_theme(style="whitegrid")

# --------------------------------------------------------------------------
# load_data
# --------------------------------------------------------------------------
def load_data():
    """Carrega o dataset e faz checagens iniciais."""
    print("=" * 74)
    print("CARREGAMENTO DO DATASET")
    print("=" * 74)
    df = pd.read_csv(DATASET_PATH)
    print(f"Dimensoes: {df.shape[0]} linhas x {df.shape[1]} colunas")
    print(f"NaN no total: {int(df.isna().sum().sum())}")

    # Conferir colunas inesperadas
    esperadas = set([ROTULO] + FEATURES)
    extras = set(df.columns) - esperadas
    if extras:
        print(f"AVISO: colunas extras encontradas: {extras}")
    print(f"Rotulo '{ROTULO}' presente: {ROTULO in df.columns}")
    print()
    return df


# --------------------------------------------------------------------------
# Passo 4 — Analise descritiva
# --------------------------------------------------------------------------
def _save_fig(nome):
    """Salva e fecha a figura atual em figures/<nome>."""
    caminho = FIG_DIR / nome
    plt.tight_layout()
    plt.savefig(caminho, dpi=120, bbox_inches="tight")
    plt.close()


def _analise_univariada(df):
    print("-" * 74)
    print("4.1 Analise univariada")
    print("-" * 74)

    # describe() de todas as 21 features -> tabela
    desc = df[FEATURES].describe().T
    desc.to_csv(RES_DIR / "estatisticas_descritivas.csv")
    print(f"Tabela describe() salva em results/estatisticas_descritivas.csv")

    # Histogramas das variaveis continuas/ordinais
    for col in NUMERICAS:
        plt.figure(figsize=(6, 4))
        nbins = min(df[col].nunique(), 50)
        sns.histplot(df[col], bins=nbins, color="#4C72B0")
        plt.title(f"Histograma — {col}")
        plt.xlabel(col)
        plt.ylabel("Frequencia")
        _save_fig(f"hist_{col}.png")
    print(f"Histogramas salvos: hist_*.png ({len(NUMERICAS)} variaveis)")

    # Boxplots das mesmas variaveis (detectar outliers)
    for col in NUMERICAS:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x=df[col], color="#DD8452")
        plt.title(f"Boxplot — {col}")
        plt.xlabel(col)
        _save_fig(f"box_{col}.png")
    print(f"Boxplots salvos: box_*.png ({len(NUMERICAS)} variaveis)")

    # Quantificar outliers de BMI (regra do IQR)
    q1, q3 = df["BMI"].quantile([0.25, 0.75])
    iqr = q3 - q1
    lim_sup = q3 + 1.5 * iqr
    n_out = int((df["BMI"] > lim_sup).sum())
    print(f"BMI: Q1={q1:.1f}, Q3={q3:.1f}, limite sup (IQR)={lim_sup:.1f}, "
          f"outliers acima={n_out} ({n_out / len(df) * 100:.2f}%)")

    # Variaveis binarias: barras de proporcao (0 vs 1)
    prop = df[BINARIAS].mean().sort_values(ascending=False)
    plt.figure(figsize=(9, 6))
    sns.barplot(x=prop.values, y=prop.index, color="#55A868")
    plt.title("Proporcao de '1' nas variaveis binarias")
    plt.xlabel("Proporcao com valor 1")
    plt.ylabel("")
    plt.xlim(0, 1)
    _save_fig("bar_binarias.png")
    print("Barras de proporcao das binarias salvas: bar_binarias.png")
    prop.to_frame("proporcao_1").to_csv(RES_DIR / "proporcao_binarias.csv")
    print()


def _analise_bivariada(df):
    print("-" * 74)
    print("4.2 Analise bivariada")
    print("-" * 74)

    # Pearson e Spearman entre as numericas
    pearson = df[NUMERICAS].corr(method="pearson")
    spearman = df[NUMERICAS].corr(method="spearman")
    pearson.to_csv(RES_DIR / "correlacao_pearson.csv")
    spearman.to_csv(RES_DIR / "correlacao_spearman.csv")
    print("Correlacoes salvas: correlacao_pearson.csv, correlacao_spearman.csv")

    # Maior divergencia entre Pearson e Spearman
    diff_arr = (spearman - pearson).abs().to_numpy(copy=True)
    np.fill_diagonal(diff_arr, 0)
    i, j = np.unravel_index(np.argmax(diff_arr), diff_arr.shape)
    a, b = NUMERICAS[i], NUMERICAS[j]
    print(f"Maior divergencia Pearson x Spearman: {a} x {b} "
          f"(Pearson={pearson.loc[a, b]:.3f}, Spearman={spearman.loc[a, b]:.3f})")

    # Pares mais correlacionados (Spearman) — possiveis redundancias
    sp_arr = spearman.to_numpy(copy=True)
    np.fill_diagonal(sp_arr, 0)
    sp = pd.DataFrame(sp_arr, index=spearman.index, columns=spearman.columns)
    pares = (
        sp.where(np.triu(np.ones(sp.shape), k=1).astype(bool))
        .stack()
        .sort_values(key=abs, ascending=False)
    )
    print("Pares mais correlacionados (Spearman):")
    for (x, y), v in pares.head(5).items():
        print(f"   {x:>9} x {y:<9}  rho = {v:+.3f}")

    # Dispersao dos pares relevantes (com amostra e jitter por serem discretas)
    rng = np.random.default_rng(RANDOM_STATE)
    amostra = df.sample(n=min(5000, len(df)), random_state=RANDOM_STATE)
    pares_scatter = [("GenHlth", "BMI"), ("GenHlth", "PhysHlth"), ("BMI", "PhysHlth")]
    for x, y in pares_scatter:
        jx = amostra[x] + rng.normal(0, 0.08, len(amostra)) if x in ORDINAIS else amostra[x]
        jy = amostra[y] + rng.normal(0, 0.08, len(amostra)) if y in ORDINAIS else amostra[y]
        plt.figure(figsize=(6, 4))
        plt.scatter(jx, jy, s=6, alpha=0.15, color="#4C72B0")
        plt.title(f"Dispersao — {x} vs {y} (amostra n={len(amostra)})")
        plt.xlabel(x)
        plt.ylabel(y)
        _save_fig(f"scatter_{x}_{y}.png")
    print(f"Dispersoes salvas: scatter_*.png ({len(pares_scatter)} pares)")
    print()


def _analise_multivariada(df):
    print("-" * 74)
    print("4.3 Analise multivariada")
    print("-" * 74)

    # Heatmap da correlacao de Spearman de todas as 21 features
    corr = df[FEATURES].corr(method="spearman")
    plt.figure(figsize=(13, 11))
    sns.heatmap(corr, cmap="coolwarm", center=0, square=True,
                linewidths=0.4, cbar_kws={"shrink": 0.7},
                annot=False, vmin=-1, vmax=1)
    plt.title("Matriz de correlacao (Spearman) — 21 features")
    _save_fig("heatmap_corr.png")
    corr.to_csv(RES_DIR / "correlacao_spearman_completa.csv")
    print("Heatmap salvo: heatmap_corr.png (+ correlacao_spearman_completa.csv)")

    # PCA exploratorio: scree plot da variancia explicada
    X = df[FEATURES].values
    X_std = StandardScaler().fit_transform(X)
    pca = PCA(random_state=RANDOM_STATE).fit(X_std)
    var = pca.explained_variance_ratio_
    var_acum = np.cumsum(var)
    n90 = int(np.searchsorted(var_acum, 0.90) + 1)
    print(f"Componentes para ~90% da variancia: {n90} de {len(var)}")
    print(f"Variancia explicada pelos 2 primeiros PCs: {var_acum[1] * 100:.1f}%")

    fig, ax1 = plt.subplots(figsize=(8, 5))
    comp = np.arange(1, len(var) + 1)
    ax1.bar(comp, var * 100, color="#4C72B0", alpha=0.7, label="Individual")
    ax1.set_xlabel("Componente principal")
    ax1.set_ylabel("Variancia explicada (%)", color="#4C72B0")
    ax2 = ax1.twinx()
    ax2.plot(comp, var_acum * 100, color="#C44E52", marker="o", label="Acumulada")
    ax2.axhline(90, color="gray", ls="--", lw=1)
    ax2.axvline(n90, color="green", ls="--", lw=1)
    ax2.set_ylabel("Variancia acumulada (%)", color="#C44E52")
    ax2.set_ylim(0, 105)
    plt.title("PCA — scree plot da variancia explicada")
    _save_fig("pca_variancia.png")
    print("Scree plot salvo: pca_variancia.png")

    # Salvar tabela da variancia explicada
    pd.DataFrame({
        "componente": comp,
        "variancia_explicada": var,
        "variancia_acumulada": var_acum,
    }).to_csv(RES_DIR / "pca_variancia.csv", index=False)
    print("Tabela salva: pca_variancia.csv")
    print()

    
def descriptive_analysis(df):
    """Executa a analise descritiva completa."""
    print("=" * 74)
    print("ANALISE DESCRITIVA")
    print("=" * 74)
    _analise_univariada(df)
    _analise_bivariada(df)
    _analise_multivariada(df)
    print("Analise descritiva concluida.\n")


# --------------------------------------------------------------------------
# Pre-processamento
# --------------------------------------------------------------------------
def preprocess(df):
    """Executa o pre-processamento e retorna um dicionario com:
    X_scaled (np), X_unscaled (DataFrame pos-clip), X_pca2 (2 comps p/ viz),
    y (rotulo separado), feature_names, sample_idx (amostra p/ agglo/dbscan).
    """
    print("=" * 74)
    print("PRE-PROCESSAMENTO")
    print("=" * 74)
    df = df.copy()

    # 1) NaN
    total_nan = int(df.isna().sum().sum())
    print(f"1) Verificacao de NaN: {total_nan} (tratamento desnecessario)")

    # 2) Remover colunas inuteis (indice/ID que o CSV possa ter adicionado)
    lixo = [c for c in df.columns if c.lower().startswith("unnamed")]
    if lixo:
        df = df.drop(columns=lixo)
        print(f"2) Colunas de indice removidas: {lixo}")
    else:
        print("2) Nenhuma coluna de indice/ID para remover.")

    # 3) Tratar outliers de BMI (clipping para faixa plausivel)
    fora = int(((df["BMI"] < BMI_MIN) | (df["BMI"] > BMI_MAX)).sum())
    df["BMI"] = df["BMI"].clip(BMI_MIN, BMI_MAX)
    print(f"3) Outliers de BMI tratados (clip {BMI_MIN:.0f}-{BMI_MAX:.0f}): "
          f"{fora} linhas afetadas ({fora / len(df) * 100:.2f}%)")

    # 4) Separar o rotulo — y nunca entra nos algoritmos
    y = df[ROTULO].copy()
    X = df.drop(columns=[ROTULO])
    feature_names = list(X.columns)
    print(f"4) Rotulo '{ROTULO}' separado. X com {X.shape[1]} features.")

    # 5) Padronizar (StandardScaler) — evita que BMI/Age/Income dominem as binarias
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("5) StandardScaler aplicado (media 0, desvio 1).")

    # 6) PCA para visualizacao (2 componentes)
    pca2 = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca2 = pca2.fit_transform(X_scaled)
    var2 = pca2.explained_variance_ratio_.sum() * 100
    print(f"6) PCA 2D para visualizacao gerado ({var2:.1f}% da variancia).")
    print("   Decisao: a clusterizacao usa X_scaled (todas as 21 dims). PCA p/ 90%")
    print("   exigiria 17/21 componentes (ganho pequeno) — mantemos interpretabilidade.")

    # 7) Verificacao final
    assert not np.isnan(X_scaled).any(), "X_scaled contem NaN!"
    assert X_scaled.dtype.kind == "f", "X_scaled nao e 100% numerico!"
    print("7) Verificacao final OK: X_scaled sem NaN e 100% numerico.")

    # Amostra fixa para Agglomerative/DBSCAN
    rng = np.random.default_rng(RANDOM_STATE)
    sample_idx = np.sort(rng.choice(len(X_scaled), size=SAMPLE_SIZE, replace=False))
    print(f"   Amostra de {SAMPLE_SIZE} linhas reservada para Agglomerative/DBSCAN.\n")

    return {
        "X_scaled": X_scaled,
        "X_unscaled": X.reset_index(drop=True),
        "X_pca2": X_pca2,
        "y": y.reset_index(drop=True),
        "feature_names": feature_names,
        "sample_idx": sample_idx,
        "scaler": scaler,
    }


# --------------------------------------------------------------------------
# Funcao de metricas
# --------------------------------------------------------------------------
def evaluate(X, labels, sample_size=SIL_SAMPLE):
    """Retorna as 3 metricas internas. Ignora ruido (-1) do DBSCAN.
    Retorna NaN quando ha menos de 2 clusters validos.
    """
    labels = np.asarray(labels)
    mask = labels != -1
    n_clusters = len(np.unique(labels[mask]))
    n_ruido = int(np.sum(labels == -1))
    out = {
        "n_clusters": n_clusters,
        "n_ruido": n_ruido,
        "silhouette": np.nan,
        "davies_bouldin": np.nan,
        "calinski_harabasz": np.nan,
    }
    if n_clusters < 2:
        return out

    Xm, Lm = X[mask], labels[mask]
    ss = sample_size if (sample_size and len(Xm) > sample_size) else None
    out["silhouette"] = silhouette_score(Xm, Lm, sample_size=ss,
                                         random_state=RANDOM_STATE)
    out["davies_bouldin"] = davies_bouldin_score(Xm, Lm)
    out["calinski_harabasz"] = calinski_harabasz_score(Xm, Lm)
    return out


# --------------------------------------------------------------------------
# Algoritmos de clustering
# --------------------------------------------------------------------------
def run_kmeans(bundle):
    """K-Means: testa k de 2 a 10; cotovelo e silhouette vs k."""
    print("-" * 74)
    print("K-Means (k = 2..10, n_init=10)")
    print("-" * 74)
    X = bundle["X_scaled"]
    rows, inercias, labels_por_k = [], [], {}
    for k in range(2, 11):
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
        lab = km.fit_predict(X)
        labels_por_k[k] = lab
        inercias.append(km.inertia_)
        m = evaluate(X, lab)
        rows.append({"k": k, "inercia": km.inertia_, **m})
        print(f"  k={k}: silhouette={m['silhouette']:.3f}  "
              f"DB={m['davies_bouldin']:.3f}  CH={m['calinski_harabasz']:.0f}")

    res = pd.DataFrame(rows)
    res.to_csv(RES_DIR / "comparacao_interna_kmeans.csv", index=False)

    # Grafico do cotovelo
    plt.figure(figsize=(7, 4))
    plt.plot(res["k"], res["inercia"], "o-", color="#4C72B0")
    plt.title("K-Means — metodo do cotovelo (inercia)")
    plt.xlabel("k (numero de clusters)")
    plt.ylabel("Inercia")
    _save_fig("kmeans_elbow.png")

    # Silhouette vs k
    plt.figure(figsize=(7, 4))
    plt.plot(res["k"], res["silhouette"], "o-", color="#C44E52")
    plt.title("K-Means — Silhouette vs k")
    plt.xlabel("k (numero de clusters)")
    plt.ylabel("Silhouette Score")
    _save_fig("kmeans_silhouette_k.png")

    melhor_k = int(res.loc[res["silhouette"].idxmax(), "k"])
    print(f"  -> melhor k por Silhouette: {melhor_k}")
    print("  Tabela salva: comparacao_interna_kmeans.csv\n")
    return res, {"algoritmo": "KMeans", "k": melhor_k,
                 "labels": labels_por_k[melhor_k]}


def run_agglomerative(bundle):
    """Agglomerative: linkages x n_clusters, na amostra (custo O(n^2))."""
    print("-" * 74)
    print(f"Agglomerative (amostra de {SAMPLE_SIZE} linhas)")
    print("-" * 74)
    Xs = bundle["X_scaled"][bundle["sample_idx"]]
    n = len(Xs)
    rows, melhor = [], None
    for lk in ["ward", "complete", "average", "single"]:
        for k in [2, 3, 4, 5]:
            ag = AgglomerativeClustering(n_clusters=k, linkage=lk)
            lab = ag.fit_predict(Xs)
            m = evaluate(Xs, lab)
            # fracao do menor cluster (detecta particoes degeneradas)
            menor_frac = np.bincount(lab).min() / n
            rows.append({"linkage": lk, "k": k, "menor_cluster_frac": menor_frac, **m})
            sil = m["silhouette"]
            # Selecao util: exige menor cluster >=5% (evita silhouette alto
            # so por isolar um punhado de pontos — caso average/single).
            valido = not np.isnan(sil) and menor_frac >= 0.05
            if valido and (melhor is None or sil > melhor["sil"]):
                melhor = {"linkage": lk, "k": k, "sil": sil, "labels": lab}
        print(f"  linkage={lk:<9} testado (k=2..5)")

    res = pd.DataFrame(rows)
    res.to_csv(RES_DIR / "comparacao_interna_agglo.csv", index=False)

    # Dendrograma (scipy) em subamostra menor para legibilidade
    sub = Xs[np.random.default_rng(RANDOM_STATE).choice(len(Xs), 1500, replace=False)]
    Z = scipy_linkage(sub, method="ward")
    plt.figure(figsize=(10, 5))
    dendrogram(Z, truncate_mode="level", p=5, no_labels=True)
    plt.title("Dendrograma (ward, subamostra de 1500)")
    plt.xlabel("Amostras (agrupadas)")
    plt.ylabel("Distancia")
    _save_fig("dendrograma.png")

    print(f"  -> melhor config: linkage={melhor['linkage']}, k={melhor['k']} "
          f"(silhouette={melhor['sil']:.3f})")
    print("  Tabela salva: comparacao_interna_agglo.csv\n")
    return res, {"algoritmo": "Agglomerative", "linkage": melhor["linkage"],
                 "k": melhor["k"], "labels": melhor["labels"]}


def run_dbscan(bundle):
    """DBSCAN: k-distance + grade de eps x min_samples, na amostra."""
    print("-" * 74)
    print(f"DBSCAN (amostra de {SAMPLE_SIZE} linhas)")
    print("-" * 74)
    Xs = bundle["X_scaled"][bundle["sample_idx"]]

    # Grafico k-distance (k-esimo vizinho) para estimar eps
    k_viz = 10
    nn = NearestNeighbors(n_neighbors=k_viz).fit(Xs)
    dist, _ = nn.kneighbors(Xs)
    kdist = np.sort(dist[:, -1])
    plt.figure(figsize=(7, 4))
    plt.plot(kdist, color="#4C72B0")
    plt.title(f"DBSCAN — k-distance ({k_viz}o vizinho)")
    plt.xlabel("Pontos ordenados")
    plt.ylabel(f"Distancia ao {k_viz}o vizinho")
    _save_fig("dbscan_kdistance.png")

    n = len(Xs)
    rows, melhor = [], None
    for eps in [1.0, 1.5, 2.0, 2.5, 3.0]:
        for ms in [5, 10, 20, 50]:
            db = DBSCAN(eps=eps, min_samples=ms)
            lab = db.fit_predict(Xs)
            m = evaluate(Xs, lab)
            frac_ruido = m["n_ruido"] / n
            rows.append({"eps": eps, "min_samples": ms, **m})
            # Selecao util: ignora solucoes degeneradas (silhouette alto
            # so porque quase tudo virou ruido). Exige <50% de ruido e um
            # numero moderado de clusters (2..10).
            valido = (
                not np.isnan(m["silhouette"])
                and 2 <= m["n_clusters"] <= 10
                and frac_ruido < 0.50
            )
            if valido and (melhor is None or m["silhouette"] > melhor["sil"]):
                melhor = {"eps": eps, "min_samples": ms,
                          "sil": m["silhouette"], "labels": lab,
                          "n_clusters": m["n_clusters"],
                          "n_ruido": m["n_ruido"]}
        print(f"  eps={eps}: configs testadas (min_samples=5,10,20,50)")

    res = pd.DataFrame(rows)
    res.to_csv(RES_DIR / "comparacao_interna_dbscan.csv", index=False)

    if melhor is None:
        print("  -> nenhuma configuracao gerou >=2 clusters validos.")
        best = {"algoritmo": "DBSCAN", "labels": None}
    else:
        print(f"  -> melhor config: eps={melhor['eps']}, "
              f"min_samples={melhor['min_samples']} "
              f"(silhouette={melhor['sil']:.3f}, clusters={melhor['n_clusters']}, "
              f"ruido={melhor['n_ruido']})")
        best = {"algoritmo": "DBSCAN", "eps": melhor["eps"],
                "min_samples": melhor["min_samples"], "labels": melhor["labels"]}
    print("  Tabela salva: comparacao_interna_dbscan.csv\n")
    return res, best


# --------------------------------------------------------------------------
# Comparacao entre algoritmos
# --------------------------------------------------------------------------
def comparar_algoritmos(res_km, best_km, res_ag, best_ag, res_db, best_db):
    """Monta a Tabela 2: melhor config de cada algoritmo, lado a lado."""
    print("=" * 74)
    print("COMPARACAO ENTRE ALGORITMOS (Tabela 2)")
    print("=" * 74)

    def _linha(res, mask, algoritmo, config):
        r = res[mask].iloc[0]
        return {
            "algoritmo": algoritmo,
            "config": config,
            "espaco": ("dados completos (70.692)" if algoritmo == "KMeans"
                       else f"amostra ({SAMPLE_SIZE})"),
            "n_clusters": int(r["n_clusters"]),
            "n_ruido": int(r["n_ruido"]),
            "silhouette": r["silhouette"],
            "davies_bouldin": r["davies_bouldin"],
            "calinski_harabasz": r["calinski_harabasz"],
        }

    linhas = [
        _linha(res_km, res_km["k"] == best_km["k"], "KMeans",
               f"k={best_km['k']}"),
        _linha(res_ag, (res_ag["linkage"] == best_ag["linkage"])
               & (res_ag["k"] == best_ag["k"]), "Agglomerative",
               f"{best_ag['linkage']}, k={best_ag['k']}"),
        _linha(res_db, (res_db["eps"] == best_db["eps"])
               & (res_db["min_samples"] == best_db["min_samples"]), "DBSCAN",
               f"eps={best_db['eps']}, min_samples={best_db['min_samples']}"),
    ]
    tab = pd.DataFrame(linhas)
    tab.to_csv(RES_DIR / "comparacao_algoritmos.csv", index=False)

    print(tab.to_string(index=False,
                        float_format=lambda v: f"{v:.3f}"))
    print("\n  Tabela salva: comparacao_algoritmos.csv\n")
    return tab


# --------------------------------------------------------------------------
# Interpretacao dos clusters com o rotulo (readiciona y)
# --------------------------------------------------------------------------
KEY_PROFILE = ["taxa_diabetes", "BMI", "HighBP", "HighChol",
               "GenHlth", "Age", "DiffWalk", "PhysHlth"]


def _interpret_one(nome, labels, y, X_unscaled):
    labels = np.asarray(labels)

    # Tabela de contingencia cluster x rotulo
    ct = pd.crosstab(labels, y.values,
                     rownames=["cluster"], colnames=["Diabetes_binary"])
    ct.to_csv(RES_DIR / f"crosstab_{nome.lower()}.csv")

    # Adjusted Rand Index (metrica EXTERNA — nao usada na clusterizacao)
    ari = adjusted_rand_score(y.values, labels)

    # Perfil medio de cada cluster
    df = X_unscaled.copy()
    df["cluster"] = labels
    df["taxa_diabetes"] = y.values
    perfil = df.groupby("cluster").mean(numeric_only=True)
    perfil["n"] = df.groupby("cluster").size()
    perfil.insert(0, "algoritmo", nome)
    ordem = ["algoritmo", "n"] + [c for c in perfil.columns
                                  if c not in ("algoritmo", "n")]
    perfil = perfil[ordem].reset_index()

    print(f"\n  [{nome}]  ARI vs Diabetes_binary = {ari:.3f}  "
          f"(externa, so curiosidade)")
    print("  Tabela de contingencia (cluster x rotulo):")
    print(ct.to_string().replace("\n", "\n    ").rjust(4))
    print("  Perfil dos clusters (variaveis-chave):")
    cols_print = ["cluster", "n"] + KEY_PROFILE
    print(perfil[cols_print].to_string(
        index=False, float_format=lambda v: f"{v:.2f}"))

    return perfil, {"algoritmo": nome, "ari": ari}


def interpret_clusters(bundle, best_km, best_ag, best_db):
    """Crosstab, perfis e ARI para a melhor config de cada algoritmo."""
    print("=" * 74)
    print("INTERPRETACAO DOS CLUSTERS COM O ROTULO")
    print("=" * 74)
    print("REGRA DE OURO: o rotulo so e usado AQUI, para avaliar/interpretar.")

    y_full = bundle["y"]
    X_full = bundle["X_unscaled"]
    idx = bundle["sample_idx"]
    y_amostra = y_full.iloc[idx].reset_index(drop=True)
    X_amostra = X_full.iloc[idx].reset_index(drop=True)

    perfis, aris = [], []

    p, a = _interpret_one("KMeans", best_km["labels"], y_full, X_full)
    perfis.append(p); aris.append(a)

    p, a = _interpret_one("Agglomerative", best_ag["labels"],
                          y_amostra, X_amostra)
    perfis.append(p); aris.append(a)

    if best_db.get("labels") is not None:
        p, a = _interpret_one("DBSCAN", best_db["labels"],
                              y_amostra, X_amostra)
        perfis.append(p); aris.append(a)
    else:
        print("\n  [DBSCAN] sem config valida para interpretar.")

    # Salvar perfis combinados e ARIs
    pd.concat(perfis, ignore_index=True).to_csv(
        RES_DIR / "perfil_clusters.csv", index=False)
    pd.DataFrame(aris).to_csv(RES_DIR / "ari_externo.csv", index=False)
    print("\n  Tabelas salvas: perfil_clusters.csv, ari_externo.csv")
    print("  (+ crosstab_kmeans.csv, crosstab_agglomerative.csv, "
          "crosstab_dbscan.csv)\n")


# --------------------------------------------------------------------------
# Visualizacoes finais
# --------------------------------------------------------------------------
def _scatter_clusters(nome, X2, labels, max_pontos=8000):
    """Scatter 2D (PCA) colorindo por cluster; ruido (-1) em cinza."""
    labels = np.asarray(labels)
    # subamostra para legibilidade quando ha muitos pontos
    if len(X2) > max_pontos:
        idx = np.random.default_rng(RANDOM_STATE).choice(
            len(X2), max_pontos, replace=False)
        X2, labels = X2[idx], labels[idx]

    plt.figure(figsize=(7, 5.5))
    for u in sorted(set(labels)):
        mask = labels == u
        if u == -1:
            plt.scatter(X2[mask, 0], X2[mask, 1], s=6, c="lightgray",
                        alpha=0.4, label="ruido")
        else:
            plt.scatter(X2[mask, 0], X2[mask, 1], s=6, alpha=0.4,
                        label=f"cluster {u}")
    plt.title(f"Clusters projetados via PCA 2D — {nome}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(markerscale=3, fontsize=8, loc="best")
    _save_fig(f"clusters_pca_{nome.lower()}.png")


def plot_metricas(tab):
    """Barras comparando as 3 metricas internas dos 3 algoritmos."""
    metricas = [
        ("silhouette", "Silhouette\n(maior = melhor)"),
        ("davies_bouldin", "Davies-Bouldin\n(menor = melhor)"),
        ("calinski_harabasz", "Calinski-Harabasz\n(maior = melhor)"),
    ]
    cores = ["#4C72B0", "#DD8452", "#55A868"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, (col, titulo) in zip(axes, metricas):
        ax.bar(tab["algoritmo"], tab[col], color=cores)
        ax.set_title(titulo)
        ax.tick_params(axis="x", rotation=15)
        for i, v in enumerate(tab[col]):
            ax.text(i, v, f"{v:.3g}", ha="center", va="bottom", fontsize=9)
    fig.suptitle("Comparacao das metricas internas entre algoritmos")
    _save_fig("comparacao_metricas.png")


def visualizacoes_finais(bundle, best_km, best_ag, best_db, tab):
    print("=" * 74)
    print("VISUALIZACOES FINAIS")
    print("=" * 74)
    X2 = bundle["X_pca2"]
    idx = bundle["sample_idx"]

    _scatter_clusters("KMeans", X2, best_km["labels"])
    _scatter_clusters("Agglomerative", X2[idx], best_ag["labels"])
    if best_db.get("labels") is not None:
        _scatter_clusters("DBSCAN", X2[idx], best_db["labels"])
    print("Scatters salvos: clusters_pca_kmeans/agglomerative/dbscan.png")

    plot_metricas(tab)
    print("Grafico de metricas salvo: comparacao_metricas.png\n")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    df = load_data()
    descriptive_analysis(df)
    bundle = preprocess(df)

    print("=" * 74)
    print("ALGORITMOS DE CLUSTERING")
    print("=" * 74)
    res_km, best_km = run_kmeans(bundle)
    res_ag, best_ag = run_agglomerative(bundle)
    res_db, best_db = run_dbscan(bundle)

    tab = comparar_algoritmos(res_km, best_km, res_ag, best_ag, res_db, best_db)
    interpret_clusters(bundle, best_km, best_ag, best_db)
    visualizacoes_finais(bundle, best_km, best_ag, best_db, tab)

    # Resumo final dos artefatos gerados
    n_fig = len(list(FIG_DIR.glob("*.png")))
    n_res = len(list(RES_DIR.glob("*.csv")))
    print("=" * 74)
    print("PIPELINE CONCLUIDO")
    print("=" * 74)
    print(f"Figuras geradas:  {n_fig} arquivos em figures/")
    print(f"Tabelas geradas:  {n_res} arquivos em results/")
    print("Dataset entregavel: data/dataset.csv")


if __name__ == "__main__":
    main()
