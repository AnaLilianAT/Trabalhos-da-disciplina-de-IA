"""Report assembly helpers for the final markdown and optional PDF deliverables."""

from __future__ import annotations

import subprocess
from pathlib import Path

import markdown
import pandas as pd

from src.config import (
    FIGURES_DIR,
    REPORT_HTML_FILE,
    REPORT_MARKDOWN_FILE,
    REPORT_PDF_INSTRUCTIONS_FILE,
    REPORTS_DIR,
    TABLES_DIR,
)


VARIABLE_DESCRIPTIONS = {
    "HighBP": "indicador de pressao arterial elevada",
    "HighChol": "indicador de colesterol elevado",
    "CholCheck": "realizacao de exame de colesterol",
    "BMI": "indice de massa corporal",
    "Smoker": "historico de tabagismo",
    "Stroke": "historico de AVC",
    "HeartDiseaseorAttack": "historico de doenca cardiaca ou infarto",
    "PhysActivity": "pratica de atividade fisica",
    "Fruits": "consumo de frutas",
    "Veggies": "consumo de vegetais",
    "HvyAlcoholConsump": "consumo excessivo de alcool",
    "AnyHealthcare": "acesso a qualquer cobertura de saude",
    "NoDocbcCost": "dificuldade de consulta por custo",
    "GenHlth": "autoavaliacao da saude geral",
    "MentHlth": "dias com pior saude mental",
    "PhysHlth": "dias com pior saude fisica",
    "DiffWalk": "dificuldade para caminhar",
    "Sex": "sexo",
    "Age": "faixa etaria",
    "Education": "escolaridade",
    "Income": "renda",
}


def _safe_read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV if it exists, otherwise return an empty dataframe."""
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _read_correlation_matrix(path: Path) -> pd.DataFrame:
    """Read a correlation matrix saved with the index exported to CSV."""
    dataframe = _safe_read_csv(path)
    if dataframe.empty:
        return dataframe

    first_column = dataframe.columns[0]
    if str(first_column).lower().startswith("unnamed"):
        dataframe = dataframe.set_index(first_column)
    return dataframe


def _relative_to_reports(path: Path) -> str:
    """Return a relative path from reports/ to another project artifact."""
    return Path("..") / path.relative_to(REPORTS_DIR.parent)


def _markdown_table(dataframe: pd.DataFrame, max_rows: int | None = None) -> str:
    """Convert a dataframe to a markdown table without external dependencies."""
    if dataframe.empty:
        return "_Tabela indisponivel._"

    table_df = dataframe.copy()
    if max_rows is not None:
        table_df = table_df.head(max_rows)

    for column in table_df.columns:
        if pd.api.types.is_float_dtype(table_df[column]):
            table_df[column] = table_df[column].map(lambda value: f"{value:.4f}")

    columns = [str(column) for column in table_df.columns]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for _, row in table_df.iterrows():
        rows.append("| " + " | ".join(str(value) for value in row.tolist()) + " |")

    return "\n".join([header, separator, *rows])


def _top_correlations(correlation_df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """Extract the strongest absolute correlations excluding the diagonal."""
    if correlation_df.empty:
        return pd.DataFrame(columns=["variable_1", "variable_2", "correlation"])

    pairs: list[dict] = []
    columns = list(correlation_df.columns)
    for i, col_i in enumerate(columns):
        for j in range(i + 1, len(columns)):
            col_j = columns[j]
            pairs.append(
                {
                    "variable_1": col_i,
                    "variable_2": col_j,
                    "correlation": correlation_df.loc[col_i, col_j],
                    "abs_correlation": abs(correlation_df.loc[col_i, col_j]),
                }
            )

    pairs_df = pd.DataFrame(pairs)
    if pairs_df.empty:
        return pd.DataFrame(columns=["variable_1", "variable_2", "correlation"])

    return pairs_df.sort_values(by="abs_correlation", ascending=False).head(top_n)[
        ["variable_1", "variable_2", "correlation"]
    ]


def _best_algorithm_row(best_models_df: pd.DataFrame) -> pd.Series:
    """Select the overall best algorithm using the same tie-break as the pipeline."""
    valid_df = best_models_df.dropna(
        subset=[
            "silhouette_score",
            "davies_bouldin_index",
            "calinski_harabasz_score",
        ]
    ).copy()
    sorted_df = valid_df.sort_values(
        by=[
            "silhouette_score",
            "davies_bouldin_index",
            "calinski_harabasz_score",
        ],
        ascending=[False, True, False],
    )
    return sorted_df.iloc[0]


def _highest_diabetes_cluster(percentage_df: pd.DataFrame) -> tuple[str, float] | None:
    """Return the cluster with highest diabetes percentage, if available."""
    if percentage_df.empty:
        return None

    cluster_column = percentage_df.columns[0]
    diabetes_columns = [column for column in percentage_df.columns[1:] if str(column) in {"1.0", "1", "Diabetic"}]
    if not diabetes_columns:
        return None

    diabetes_column = diabetes_columns[0]
    best_row = percentage_df.sort_values(by=diabetes_column, ascending=False).iloc[0]
    return str(best_row[cluster_column]), float(best_row[diabetes_column])


def _cluster_profile_commentary(mean_df: pd.DataFrame, cluster_name: str) -> str:
    """Generate a compact interpretation for a specific cluster."""
    if mean_df.empty:
        return ""

    cluster_df = mean_df.copy()
    first_column = cluster_df.columns[0]
    cluster_df = cluster_df.rename(columns={first_column: "cluster"})
    selected = cluster_df.loc[cluster_df["cluster"].astype(str) == str(cluster_name)]
    if selected.empty:
        return ""

    row = selected.iloc[0]
    comments = [
        f"BMI medio de {row['BMI']:.2f}" if "BMI" in row else None,
        f"proporcao de HighBP de {row['HighBP']:.2f}" if "HighBP" in row else None,
        f"proporcao de HighChol de {row['HighChol']:.2f}" if "HighChol" in row else None,
        f"GenHlth medio de {row['GenHlth']:.2f}" if "GenHlth" in row else None,
        f"Age medio de {row['Age']:.2f}" if "Age" in row else None,
        f"Income medio de {row['Income']:.2f}" if "Income" in row else None,
        f"Education media de {row['Education']:.2f}" if "Education" in row else None,
    ]
    return ", ".join(comment for comment in comments if comment is not None) + "."


def _figure_markdown(title: str, filename: str) -> str:
    """Generate markdown image syntax for a report figure."""
    figure_path = _relative_to_reports(FIGURES_DIR / filename)
    return f"![{title}]({figure_path.as_posix()})"


def _write_html_from_markdown(markdown_content: str) -> Path:
    """Generate an HTML version of the report from markdown."""
    html_body = markdown.markdown(markdown_content, extensions=["tables", "fenced_code"])
    html_content = (
        "<html><head><meta charset='utf-8'>"
        "<title>Relatorio de Clustering em Diabetes</title>"
        "<style>body{font-family:Arial,sans-serif;max-width:1100px;margin:40px auto;line-height:1.6;}"
        "img{max-width:100%;height:auto;} table{border-collapse:collapse;} th,td{border:1px solid #ccc;padding:6px;}"
        "h1,h2,h3{color:#1d3557;} code{background:#f4f4f4;padding:2px 4px;}</style>"
        "</head><body>"
        f"{html_body}"
        "</body></html>"
    )
    REPORT_HTML_FILE.write_text(html_content, encoding="utf-8")
    return REPORT_HTML_FILE


def _attempt_pdf_conversion(markdown_path: Path) -> tuple[bool, str]:
    """Try to convert the markdown report to PDF using pandoc if available."""
    try:
        subprocess.run(
            ["pandoc", str(markdown_path), "-o", str(REPORTS_DIR / "report.pdf")],
            check=True,
            capture_output=True,
            text=True,
        )
        return True, "PDF gerado automaticamente com pandoc."
    except FileNotFoundError:
        return False, "Pandoc nao esta instalado neste ambiente."
    except subprocess.CalledProcessError as exc:
        return False, f"Falha ao converter PDF com pandoc: {exc.stderr.strip()}"


def _write_pdf_instructions() -> Path:
    """Write manual conversion instructions for the user."""
    instructions = (
        "Conversao manual do relatorio para PDF\n\n"
        "1. Instale o pandoc: https://pandoc.org/installing.html\n"
        "2. Opcionalmente instale uma engine LaTeX, como TinyTeX ou MiKTeX.\n"
        "3. Execute na raiz do projeto:\n\n"
        "   pandoc reports/report.md -o reports/report.pdf\n\n"
        "4. Se preferir, abra reports/report.html no navegador e use a impressao em PDF.\n"
    )
    REPORT_PDF_INSTRUCTIONS_FILE.write_text(instructions, encoding="utf-8")
    return REPORT_PDF_INSTRUCTIONS_FILE


def build_report(
    dataset_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    best_models_df: pd.DataFrame | None = None,
) -> dict:
    """Build the final report in markdown and optionally convert it to PDF."""
    print("\n[Report] Building markdown report...")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if best_models_df is None or best_models_df.empty:
        best_models_df = _safe_read_csv(TABLES_DIR / "best_models_summary.csv")

    all_results_df = _safe_read_csv(TABLES_DIR / "all_clustering_results.csv")
    kmeans_results_df = _safe_read_csv(TABLES_DIR / "kmeans_results.csv")
    agg_results_df = _safe_read_csv(TABLES_DIR / "agglomerative_results.csv")
    dbscan_results_df = _safe_read_csv(TABLES_DIR / "dbscan_results.csv")
    univariate_df = _safe_read_csv(TABLES_DIR / "univariate_statistics.csv")
    pearson_df = _read_correlation_matrix(TABLES_DIR / "pearson_correlation_matrix.csv")
    spearman_df = _read_correlation_matrix(TABLES_DIR / "spearman_correlation_matrix.csv")
    pca_variance_df = _safe_read_csv(TABLES_DIR / "pca_explained_variance.csv")
    outlier_df = _safe_read_csv(TABLES_DIR / "outlier_summary_iqr.csv")

    kmeans_pct_df = _safe_read_csv(TABLES_DIR / "kmeans_best_vs_original_labels_percentage.csv")
    agg_pct_df = _safe_read_csv(TABLES_DIR / "agglomerative_best_vs_original_labels_percentage.csv")
    dbscan_pct_df = _safe_read_csv(TABLES_DIR / "dbscan_best_vs_original_labels_percentage.csv")

    kmeans_cont_df = _safe_read_csv(TABLES_DIR / "kmeans_best_vs_original_labels_contingency.csv")
    agg_cont_df = _safe_read_csv(TABLES_DIR / "agglomerative_best_vs_original_labels_contingency.csv")
    dbscan_cont_df = _safe_read_csv(TABLES_DIR / "dbscan_best_vs_original_labels_contingency.csv")

    kmeans_mean_df = _safe_read_csv(TABLES_DIR / "kmeans_best_cluster_profile_mean.csv")
    agg_mean_df = _safe_read_csv(TABLES_DIR / "agglomerative_best_cluster_profile_mean.csv")
    dbscan_mean_df = _safe_read_csv(TABLES_DIR / "dbscan_best_cluster_profile_mean.csv")

    top_pearson_df = _top_correlations(pearson_df)
    top_spearman_df = _top_correlations(spearman_df)
    best_overall = _best_algorithm_row(best_models_df)
    total_outliers = int(outlier_df["n_outliers"].sum()) if not outlier_df.empty else 0
    pca_explained = (
        float(pca_variance_df["cumulative_explained_variance_ratio"].iloc[-1])
        if not pca_variance_df.empty
        else 0.0
    )

    variable_overview = ", ".join(
        f"`{column}` ({VARIABLE_DESCRIPTIONS.get(column, 'variavel do dataset')})"
        for column in feature_df.columns
    )

    kmeans_risk_cluster = _highest_diabetes_cluster(kmeans_pct_df)
    agg_risk_cluster = _highest_diabetes_cluster(agg_pct_df)
    dbscan_risk_cluster = _highest_diabetes_cluster(dbscan_pct_df)

    report_sections = [
        "# Relatorio de Clustering no Dataset CDC Diabetes Health Indicators",
        "",
        "## 1. Introducao",
        (
            "Este trabalho tem como objetivo aplicar tecnicas de clustering a um dataset real da area da saude, "
            "com foco no tema diabetes. A proposta consiste em identificar perfis de individuos com caracteristicas "
            "semelhantes de saude, estilo de vida e demografia, de modo a observar padroes latentes sem usar o "
            "rotulo clinico durante o treinamento dos algoritmos."
        ),
        "",
        "## 2. Descricao do Dataset",
        (
            "O dataset utilizado foi o **CDC Diabetes Health Indicators**, derivado do **Behavioral Risk Factor "
            "Surveillance System (BRFSS) 2015** do CDC. A base analisada contem "
            f"**{len(dataset_df)} instancias** e **{len(dataset_df.columns)} variaveis** no arquivo original, "
            f"sendo **{len(feature_df.columns)} variaveis de entrada** usadas na clusterizacao."
        ),
        (
            "As variaveis representam indicadores de saude, comportamento, acesso a servicos e aspectos "
            f"demograficos, incluindo {variable_overview}. O campo `Diabetes_binary` corresponde ao rotulo "
            "original do problema e foi removido antes da clusterizacao, sendo readicionado apenas depois para "
            "interpretacao dos grupos encontrados."
        ),
        "",
        "## 3. Problema Definido",
        (
            "O problema investigado neste trabalho consiste em verificar se individuos podem ser agrupados em "
            "perfis distintos de saude e risco associados ao diabetes, com base em indicadores como BMI, pressao "
            "alta, colesterol alto, idade, saude geral, atividade fisica, renda e escolaridade."
        ),
        (
            "Apos a clusterizacao, o rotulo original `Diabetes_binary` foi utilizado apenas para interpretar os "
            "clusters e nunca como entrada nos algoritmos nem como criterio principal de selecao dos modelos."
        ),
        "",
        "## 4. Analise Descritiva",
        (
            "Na analise univariada foram calculadas estatisticas descritivas como media, mediana, desvio padrao, "
            "quartis e valores extremos para todas as variaveis numericas. A Tabela a seguir apresenta um recorte "
            "dessas estatisticas:"
        ),
        "",
        _markdown_table(univariate_df, max_rows=10),
        "",
        "Os graficos abaixo ilustram distribuicoes relevantes para o relatorio:",
        "",
        _figure_markdown("Histograma de BMI", "histogram_bmi.png"),
        "",
        _figure_markdown("Boxplot de BMI", "boxplot_bmi.png"),
        "",
        "Na analise bivariada foram calculadas correlacoes de Pearson e Spearman. As correlacoes absolutas mais fortes observadas foram:",
        "",
        "**Top correlacoes de Pearson**",
        "",
        _markdown_table(top_pearson_df),
        "",
        "**Top correlacoes de Spearman**",
        "",
        _markdown_table(top_spearman_df),
        "",
        _figure_markdown("Heatmap de correlacao de Pearson", "pearson_correlation_heatmap.png"),
        "",
        _figure_markdown("Dispersao entre BMI e GenHlth", "scatter_bmi_vs_genhlth.png"),
        "",
        (
            "Na analise multivariada foi aplicado PCA com dois componentes apenas para visualizacao exploratoria. "
            f"A variancia explicada acumulada pelos dois primeiros componentes foi de **{pca_explained:.4f}**, o que "
            "indica que a projecao em 2D deve ser interpretada como apoio visual e nao como representacao completa "
            "da estrutura dos dados."
        ),
        "",
        _figure_markdown("PCA 2D exploratorio", "pca_scatter_2_components.png"),
        "",
        "## 5. Pre-processamento",
        (
            "O pre-processamento foi executado apenas sobre as variaveis de entrada. Nao foram identificadas "
            "colunas de identificacao, nomes ou datas a serem removidas. Tambem nao foram observados valores "
            "faltantes no CSV utilizado, de modo que nao foi necessario imputar medias, medianas ou modas."
        ),
        (
            f"Mesmo sem faltantes, foram detectados **{total_outliers} valores potencialmente discrepantes** pelo "
            "criterio do intervalo interquartil (IQR). Em vez de remover instancias, foi aplicado clipping nos "
            "limites inferior e superior de cada variavel numerica. Na etapa final, as variaveis foram "
            "padronizadas com `StandardScaler` para adequar a escala aos algoritmos baseados em distancia. "
            "O PCA foi usado apenas para visualizacao e nao como base principal de clusterizacao."
        ),
        "",
        "## 6. Algoritmos de Clustering",
        (
            "Foram executados tres algoritmos de clustering, todos sem utilizar o rotulo `Diabetes_binary`:"
        ),
        "",
        "- **K-Means**: teste de `k` de 2 a 10, com `random_state` fixo e `n_init = 10`.",
        "- **Agglomerative Clustering**: teste de `n_clusters` de 2 a 10 com os linkages `ward`, `complete`, `average` e `single`.",
        "- **DBSCAN**: teste de `eps` em `[0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0]` e `min_samples` em `[5, 10, 20, 50]`.",
        "",
        (
            "Como o dataset possui mais de 250 mil registros, foi utilizada amostragem reprodutivel em algoritmos "
            "mais custosos. No experimento final, K-Means trabalhou com 20.000 instancias, Agglomerative com "
            "10.000 instancias estratificadas pelo rotulo apenas para tornar a amostra representativa, e DBSCAN "
            "com 10.000 instancias aleatorias. Em nenhum caso o rotulo foi usado no treinamento."
        ),
        "",
        "## 7. Metricas de Avaliacao",
        (
            "A comparacao dos algoritmos foi feita com metricas internas de clustering: **Silhouette Score**, "
            "**Davies-Bouldin Index** e **Calinski-Harabasz Score**. O Silhouette Score mede coesao e separacao "
            "entre grupos, sendo melhor quanto maior. O Davies-Bouldin Index mede sobreposicao entre clusters, "
            "sendo melhor quanto menor. O Calinski-Harabasz Score compara dispersao entre clusters e dispersao "
            "interna, sendo melhor quanto maior."
        ),
        (
            "Essas metricas sao internas e nao usam o rotulo `Diabetes_binary`, o que esta alinhado com a natureza "
            "nao supervisionada do problema."
        ),
        "",
        "## 8. Resultados",
        "A Tabela seguinte resume a melhor configuracao de cada algoritmo:",
        "",
        _markdown_table(best_models_df),
        "",
        "Os resultados completos por algoritmo foram salvos em CSV e seus principais trechos sao apresentados a seguir.",
        "",
        "**K-Means**",
        "",
        _markdown_table(kmeans_results_df),
        "",
        "**Agglomerative Clustering**",
        "",
        _markdown_table(agg_results_df.sort_values(by="silhouette_score", ascending=False), max_rows=10),
        "",
        "**DBSCAN**",
        "",
        _markdown_table(dbscan_results_df.sort_values(by="silhouette_score", ascending=False), max_rows=10),
        "",
        "As figuras de comparacao entre algoritmos sao apresentadas abaixo:",
        "",
        _figure_markdown("Melhor Silhouette por algoritmo", "final_best_silhouette_score_by_algorithm.png"),
        "",
        _figure_markdown("Melhor Davies-Bouldin por algoritmo", "final_best_davies_bouldin_index_by_algorithm.png"),
        "",
        _figure_markdown("Melhor Calinski-Harabasz por algoritmo", "final_best_calinski_harabasz_score_by_algorithm.png"),
        "",
        (
            f"Pelo criterio principal adotado, o melhor algoritmo foi **{best_overall['algorithm']}**, com "
            f"Silhouette Score de **{best_overall['silhouette_score']:.4f}**."
        ),
        "",
        "## 9. Interpretacao dos Clusters",
        "A tabela de contingencia do melhor K-Means frente ao rotulo original foi:",
        "",
        _markdown_table(kmeans_cont_df),
        "",
        "A tabela percentual correspondente foi:",
        "",
        _markdown_table(kmeans_pct_df),
        "",
        "Para Agglomerative Clustering:",
        "",
        _markdown_table(agg_cont_df),
        "",
        _markdown_table(agg_pct_df),
        "",
        "Para DBSCAN:",
        "",
        _markdown_table(dbscan_cont_df),
        "",
        _markdown_table(dbscan_pct_df),
        "",
        (
            "No melhor K-Means, o cluster com maior proporcao de individuos com diabetes foi o cluster "
            f"**{kmeans_risk_cluster[0]}**, com **{kmeans_risk_cluster[1]:.2f}%** de casos positivos."
            if kmeans_risk_cluster
            else "Nao foi possivel calcular a concentracao de diabetes para o melhor K-Means."
        ),
        "",
        (
            "Esse cluster apresentou "
            f"{_cluster_profile_commentary(kmeans_mean_df, kmeans_risk_cluster[0])}"
            if kmeans_risk_cluster
            else ""
        ),
        "",
        (
            "No melhor Agglomerative, o cluster com maior proporcao de diabetes foi "
            f"**{agg_risk_cluster[0]}**, com **{agg_risk_cluster[1]:.2f}%**."
            if agg_risk_cluster
            else ""
        ),
        "",
        (
            "No melhor DBSCAN, a maior proporcao de diabetes apareceu em "
            f"**{dbscan_risk_cluster[0]}**, com **{dbscan_risk_cluster[1]:.2f}%**."
            if dbscan_risk_cluster
            else ""
        ),
        "",
        (
            "Os perfis medios reforcam a relevancia de variaveis como BMI, HighBP, HighChol, GenHlth, Age, Income "
            "e Education. No K-Means, o cluster de maior risco mostrou maior BMI medio, maior prevalencia de "
            "pressao alta e colesterol alto, pior saude geral, maior idade media e menor renda media, o que "
            "e coerente com um perfil de vulnerabilidade cardiometabolica."
        ),
        "",
        _figure_markdown("Distribuicao de Diabetes por cluster - K-Means", "final_kmeans_label_distribution_by_cluster.png"),
        "",
        _figure_markdown("Distribuicao de Diabetes por cluster - Agglomerative", "final_agglomerative_label_distribution_by_cluster.png"),
        "",
        _figure_markdown("Distribuicao de Diabetes por cluster - DBSCAN", "final_dbscan_label_distribution_by_cluster.png"),
        "",
        _figure_markdown("PCA do melhor K-Means", "final_kmeans_best_pca.png"),
        "",
        _figure_markdown("PCA do melhor Agglomerative", "final_agglomerative_best_pca.png"),
        "",
        _figure_markdown("PCA do melhor DBSCAN", "final_dbscan_best_pca.png"),
        "",
        _figure_markdown(
            "Heatmap do perfil medio padronizado do melhor algoritmo",
            f"final_{str(best_overall['algorithm']).lower()}_best_overall_cluster_profile_heatmap.png",
        ),
        "",
        "## 10. Discussao",
        (
            "Os algoritmos apresentaram comportamentos distintos. O K-Means produziu uma segmentacao simples e "
            "interpretavel, destacando dois perfis relativamente contrastantes. O Agglomerative Clustering obteve "
            "resultado um pouco melhor em Silhouette do que o K-Means, mas a melhor configuracao encontrada por "
            "single linkage gerou um cluster muito pequeno, o que limita sua utilidade pratica."
        ),
        (
            "O DBSCAN obteve o melhor Silhouette Score geral, mas isso ocorreu ao custo de classificar "
            f"**{best_overall['noise_percentage']:.2f}%** das instancias da amostra como ruido quando o melhor "
            "algoritmo geral foi DBSCAN, o que reduz fortemente a interpretabilidade e a cobertura da solucao."
            if str(best_overall["algorithm"]) == "dbscan"
            else "O DBSCAN nao foi o melhor algoritmo geral, mas ainda assim mostrou comportamento sensivel a parametros e alta proporcao de ruido em varias configuracoes."
        ),
        (
            "Entre as principais limitacoes do estudo, destacam-se: "
            "(1) o DBSCAN tende a perder poder discriminativo em alta dimensionalidade; "
            "(2) metricas internas nem sempre correspondem ao rotulo clinico real; "
            "(3) o BRFSS e um survey baseado em autorrelato, sujeito a vieses de memoria e declaracao."
        ),
        "",
        "## 11. Conclusao",
        (
            f"O melhor algoritmo segundo o criterio principal adotado foi **{best_overall['algorithm']}**. "
            "Ainda assim, a interpretacao substantiva sugere que a qualidade de um clustering em saude nao deve ser "
            "avaliada apenas por uma metrica interna isolada, mas tambem por estabilidade, cobertura e capacidade "
            "de produzir perfis clinicamente plausiveis."
        ),
        (
            "Os resultados indicaram a existencia de perfis com maior risco cardiometabolico, caracterizados por "
            "maior BMI, maior frequencia de pressao alta e colesterol alto, pior saude geral, idade mais avancada "
            "e menor renda. Em todo o processo, o rotulo `Diabetes_binary` foi mantido fora da clusterizacao e "
            "utilizado apenas posteriormente para interpretacao dos grupos."
        ),
        "",
        "## 12. Referencias",
        "- Centers for Disease Control and Prevention (CDC). *Behavioral Risk Factor Surveillance System, 2015*. Disponivel em: https://www.cdc.gov/brfss/annual_data/annual_2015.html",
        "- Liu X, et al. *Association between diabetes, metabolic syndrome and heart attack in US adults: a cross-sectional analysis using the Behavioral Risk Factor Surveillance System 2015*. BMJ Open, 2019. Disponivel em: https://pmc.ncbi.nlm.nih.gov/articles/PMC6747668/",
        "- Scikit-learn Developers. *KMeans*. Disponivel em: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html",
        "- Scikit-learn Developers. *AgglomerativeClustering*. Disponivel em: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html",
        "- Scikit-learn Developers. *DBSCAN*. Disponivel em: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html",
    ]

    markdown_content = "\n".join(section for section in report_sections if section is not None)
    REPORT_MARKDOWN_FILE.write_text(markdown_content, encoding="utf-8")
    print(f"[Report] Markdown report saved to: {REPORT_MARKDOWN_FILE}")

    html_path = _write_html_from_markdown(markdown_content)
    print(f"[Report] HTML report saved to: {html_path}")

    pdf_generated, pdf_message = _attempt_pdf_conversion(REPORT_MARKDOWN_FILE)
    instructions_path = _write_pdf_instructions()
    print(f"[Report] {pdf_message}")
    print(f"[Report] PDF conversion instructions saved to: {instructions_path}")

    return {
        "markdown_path": str(REPORT_MARKDOWN_FILE),
        "html_path": str(html_path),
        "pdf_generated": pdf_generated,
        "pdf_message": pdf_message,
        "instructions_path": str(instructions_path),
    }
