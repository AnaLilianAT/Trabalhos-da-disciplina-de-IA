# Escolhas de Pré-processamento e Justificativas

| Aspecto | Escolha | Justificativa |
| --- | --- | --- |
| Split | Estratificado 60/20/20 (train/val/test) | Treino para ajuste; validação para early stopping/Optuna; teste avaliado uma única vez. Estratificação preserva a proporção das classes (e dos decis de BMI na regressão) nos três conjuntos, importante dado o desbalanceamento. |
| Prevenção de vazamento | `fit` apenas no treino | Imputador, scaler e (opcional) clipper aprendem estatísticas só do treino e são aplicados via `transform` em val/test, encapsulados em `ColumnTransformer`/`Pipeline`. |
| Valores ausentes (contínuas/ordinais) | `SimpleImputer(strategy='median')` | A mediana é robusta a outliers (relevantes em BMI/MentHlth/PhysHlth). O dataset não tem ausentes, mas a etapa cumpre o requisito e dá robustez a novas amostras. |
| Valores ausentes (binárias) | `SimpleImputer(strategy='most_frequent')` | Moda é a estatística adequada para variáveis 0/1. |
| Encoding | Manter codificação numérica original | Binárias já são 0/1; ordinais (`GenHlth`, `Age`, `Education`, `Income`) têm ordem natural — **não** se aplica One-Hot para não perder a ordem nem inflar a dimensão. |
| Scaling | `StandardScaler` em ordinais+contínuas; binárias `passthrough` | Padroniza escalas distintas (ex.: BMI ~12–98 vs. dias 0–30), ajudando a convergência da MLP. Padronizar variáveis 0/1 é desnecessário e prejudica a interpretabilidade, por isso passam direto. |
| Alternativa de scaling | `RobustScaler` (configurável) | Usa mediana/IQR; alternativa menos sensível aos outliers de BMI/MentHlth/PhysHlth. |
| Outliers (opcional) | Clipping de BMI no percentil 99 (flag `clip_bmi`) | Atenua a cauda extrema do BMI sem descartar amostras; limite aprendido no treino. Desativado por padrão; impacto documentado. |
| Desbalanceamento | `class_weight='balanced'` no treino (Etapa 4) | Compensa as classes minoritárias sem alterar os dados; complementado por métricas robustas (ROC-AUC, PR-AUC, F1 macro). |
