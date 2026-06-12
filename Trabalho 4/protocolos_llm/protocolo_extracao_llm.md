# Protocolo de Extração de Conhecimento com LLM

> Estratégia 3 da Fase 3. Documenta o **prompt exato**, a **resposta bruta** do LLM e a
> **revisão realizada pelos integrantes do grupo** (com o que foi corrigido). As triplas revisadas
> ficam em `dados_brutos/llm_triplas_revisadas.json` e são consumidas pelo
> `scripts/03_extrair_conhecimento.py` para gerar indivíduos/relações na ontologia.

- **Modelo usado:** Claude (Anthropic), família Claude 4.x.
- **Data da extração:** 2026-06-05.
- **Texto-fonte:** `dados_brutos/amostra_local_taim.txt` (parágrafo ecológico sobre a ESEC Taim,
  baseado em conhecimento geral da região; o mesmo arquivo serve de *fallback* caso a
  Wikipedia esteja bloqueada).

---

## 1. Prompt exato enviado ao LLM

```
Você é um extrator de conhecimento para construção de ontologias. Dado o TEXTO a seguir
sobre o Banhado do Taim (RS), extraia as relações afirmadas no texto no formato de triplas
(sujeito, relacao, objeto). Regras:
- Use relações curtas e normalizadas (verbos no infinitivo ou nomes curtos).
- Não invente fatos que não estejam no texto.
- Retorne SOMENTE um JSON válido: uma lista de objetos {"sujeito","relacao","objeto"}.

TEXTO:
<conteúdo de amostra_local_taim.txt>
```

## 2. Resposta BRUTA do LLM (antes da revisão)

```json
[
  {"sujeito": "BR-471", "relacao": "atravessa", "objeto": "ESEC Taim"},
  {"sujeito": "BR-471", "relacao": "corta", "objeto": "Banhado do Taim"},
  {"sujeito": "Capivara", "relacao": "habita", "objeto": "Banhado"},
  {"sujeito": "Capivara", "relacao": "atravessa", "objeto": "BR-471"},
  {"sujeito": "Capivara", "relacao": "e atropelada em", "objeto": "BR-471"},
  {"sujeito": "Ratao-do-banhado", "relacao": "habita", "objeto": "Banhado"},
  {"sujeito": "Jacare-do-papo-amarelo", "relacao": "e atropelado", "objeto": "-"},
  {"sujeito": "Neblina", "relacao": "reduz", "objeto": "visibilidade"},
  {"sujeito": "Chuva forte", "relacao": "reduz", "objeto": "visibilidade"},
  {"sujeito": "Capivara", "relacao": "atravessa rodovia ao", "objeto": "entardecer"},
  {"sujeito": "Passagens de fauna", "relacao": "mitigam", "objeto": "atropelamentos"}
]
```

## 3. Revisão realizada pelo grupo — o que foi corrigido

A saída do LLM **não foi usada cega**. Revisão feita pelo grupo:

| # | Tripla bruta | Decisão na revisão | Motivo |
|---|---|---|---|
| 1 | BR-471 atravessa ESEC Taim | **Mantida** (contexto) | Fato correto, mas sem classe própria na ontologia para "ESEC"; usada só como contexto. |
| 2 | BR-471 corta Banhado do Taim | **Mantida** (contexto) | Idem; modelada indiretamente via `proximoA` nos trechos. |
| 3 | Capivara habita Banhado | **Mapeada** → `habita` | Bate com a object property `habita`. |
| 4 | Capivara atravessa BR-471 | **Mapeada** → `atravessa` | Bate com a object property `atravessa`. |
| 5 | Capivara "e atropelada em" BR-471 | **Descartada (redundante)** | Atropelamento já é reificado como classe `EventoAtropelamento`; relação binária perderia atributos. |
| 6 | Ratão-do-banhado habita Banhado | **Mapeada** → novo indivíduo + `habita` | Espécie real do Taim ausente na ABox; vira indivíduo derivado. |
| 7 | Jacaré-do-papo-amarelo "e atropelado" "-" | **DESCARTADA (malformada)** | Objeto vazio ("-"); tripla sem informação válida — típico ruído de LLM. |
| 8 | Neblina reduz visibilidade | **Anotada, não mapeada como relação** | A ontologia modela visibilidade como `FatorVisibilidade`/`visibilidadeMetros`, não como relação binária; registrada como observação. |
| 9 | Chuva forte reduz visibilidade | **Anotada, não mapeada** | Idem. |
| 10 | Capivara "atravessa rodovia ao" entardecer | **DESCARTADA (ruído)** | Relação não-normalizada misturando verbo e período; "entardecer" mapeável a `PeriodoDia`, mas a tripla está malformada. |
| 11 | Passagens de fauna mitigam atropelamentos | **Descartada (fora do escopo)** | Verdadeiro, porém não há classe de medida de mitigação na ontologia; possível, mas não modelado. |

**Resumo da revisão:** de 11 triplas brutas, 3 foram aproveitadas e mapeadas para propriedades
existentes (`habita`, `atravessa`), 1 gerou um novo indivíduo (ratão-do-banhado), 2 foram
descartadas por estarem malformadas/ruidosas (itens 7 e 10 — alucinação estrutural típica) e
as demais foram mantidas apenas como contexto ou anotação. As triplas aproveitadas estão em
`dados_brutos/llm_triplas_revisadas.json`.
