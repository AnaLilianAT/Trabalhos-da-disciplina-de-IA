# Diabetes Clustering

Projeto base para um trabalho pratico de Inteligencia Artificial sobre clustering
aplicado ao dataset **CDC Diabetes Health Indicators / BRFSS 2015**.

## Objetivo

Esta estrutura foi preparada para suportar um pipeline reprodutivel de:

- carregamento de dados;
- analise exploratoria;
- preprocessamento sem usar o rotulo de diabetes na clusterizacao;
- execucao de K-Means, Agglomerative Clustering e DBSCAN;
- avaliacao com metricas internas;
- geracao de tabelas, figuras e relatorio final.

## Estrutura

```text
diabetes_clustering/
  data/
    raw/
    processed/
  outputs/
    figures/
    tables/
  reports/
  src/
    config.py
    data_loader.py
    eda.py
    preprocessing.py
    clustering.py
    evaluation.py
    visualization.py
    report_builder.py
    main.py
  requirements.txt
  README.md
```

## Dataset

O arquivo bruto foi colocado em:

`data/raw/diabetes_binary_health_indicators_BRFSS2015.csv`

## Observacao sobre Agglomerative Clustering

Como Agglomerative Clustering tem custo computacional alto em bases grandes, o
projeto aplica amostragem reprodutivel apenas dentro dessa etapa quando
`X_scaled` tiver mais de 10000 linhas.

- se `y` estiver disponivel, a amostra tenta preservar a distribuicao do rotulo
  por meio de amostragem estratificada;
- se `y` nao estiver disponivel, ou se a estratificacao nao for viavel, o codigo
  usa amostragem aleatoria simples;
- o rotulo `Diabetes_binary` nao e usado no treinamento, apenas opcionalmente
  para definir a amostra de forma mais representativa.

## Observacao sobre DBSCAN

DBSCAN pode ter desempenho pior em dados de alta dimensionalidade, porque a
nocao de densidade tende a ficar menos discriminativa quando o numero de
variaveis aumenta.

- por isso, o projeto permite amostragem reprodutivel dentro da etapa de
  DBSCAN quando a base estiver muito grande;
- as metricas de DBSCAN sao calculadas apenas sobre os pontos que nao foram
  classificados como ruido;
- pontos de ruido aparecem como categoria separada nas visualizacoes em PCA.

## Como executar

Crie o ambiente virtual, instale as dependencias e rode:

```bash
python -m src.main
```

O `main.py` executa carregamento, EDA, preprocessamento e os experimentos de
K-Means, Agglomerative Clustering e DBSCAN.
