# Checklist da Rubrica — Autoavaliação

Mapeia **cada requisito do enunciado** para **onde foi atendido** (arquivo/linha ou artefato).
Serve de autoavaliação e de roteiro para os itens manuais finais do grupo.

Legenda: ✅ feito · ⏳ opcional/pendente · ✋ manual (responsabilidade do grupo).

---

## Ontologia (TBox) — Fase 1

| Requisito | Status | Onde |
|---|---|---|
| ≥15 classes com hierarquia (subClassOf) | ✅ (32 classes) | [`01_construir_ontologia.py`](../scripts/01_construir_ontologia.py) linhas 58–153 |
| ≥10 object properties com domain/range | ✅ (14) | linhas 162–225 |
| ≥10 data properties com domain/range | ✅ (14) | linhas 233–289 |
| Restrições (cardinalidade, existencial, universal) | ✅ (min/exact/some/only) | linhas 299–308 |
| Classe definida (equivalência) | ✅ `TrechoCritico` | linhas 312–319 |
| Classes disjuntas | ✅ `AllDisjoint` | linhas 321–322 |
| ≥1 relação temporal | ✅ `dataHora`, `ocorreNoPeriodo`, `ocorreNaEstacao`, `precedeEvento` | linhas 175, 180, 223, 263 |
| ≥1 relação espacial | ✅ `proximoA`, `adjacenteA`, `latitude`, `longitude` | linhas 185, 218, 253, 258 |
| Características OWL (funcional/simétrica/transitiva) | ✅ | linhas 190, 218, 223, 263 |
| Arquivo `.owl` em RDF/XML | ✅ | [`taim.owl`](../taim.owl) |
| Abre no Protégé | ✋ | conferir abrindo `taim.owl` no Protégé |
| Reasoner sem inconsistência | ✅ HermiT via `sync_reasoner` | log da Fase 1/2/3 + [`AMBIENTE.md`](../AMBIENTE.md) |

## Povoamento (ABox) — Fase 2

| Requisito | Status | Onde |
|---|---|---|
| ≥100 indivíduos | ✅ (111 após Fase 3; 108 na Fase 2) | [`02_povoar_ontologia.py`](../scripts/02_povoar_ontologia.py) |
| Script de povoamento reprodutível | ✅ `random.seed(42)` | linha ~33 |
| Eventos com data/hora, ≥1 animal, exatamente 1 trecho | ✅ validado | bloco de asserções no `main()` |
| ≥3 exemplos concretos de indivíduos | ✅ impressos | saída do script (seção "EXEMPLOS") |
| Explicação da geração dos dados | ✅ | [`LIMITACOES_DADOS.md`](LIMITACOES_DADOS.md) seções 1–3 |

## Fontes e extração — Fase 3

| Requisito | Status | Onde |
|---|---|---|
| ≥2 estratégias de extração | ✅ (3: Wikipedia, Wikidata, LLM) | [`03_extrair_conhecimento.py`](../scripts/03_extrair_conhecimento.py) |
| Protocolo de LLM (prompt + resposta + revisão) | ✅ | [`protocolos_llm/protocolo_extracao_llm.md`](../protocolos_llm/protocolo_extracao_llm.md) |
| Revisão obrigatória da saída do LLM | ✅ (2 triplas descartadas) | mesmo protocolo, seção 3 |
| Indivíduos derivados de extração (rastreáveis) | ✅ (ratão, cervo, jaçanã + QIDs) | `rdfs:comment` nos indivíduos |
| Documento de fontes, método, limitações e erros | ✅ | [`FONTES_E_EXTRACAO.md`](FONTES_E_EXTRACAO.md) |

## Consultas SPARQL — Fase 4

| Requisito | Status | Onde |
|---|---|---|
| ≥30 consultas | ✅ (32) | [`04_consultas_sparql.py`](../scripts/04_consultas_sparql.py) |
| Cobertura das 5 categorias | ✅ (Simples 7, Múltiplas 9, Filtros 6, Agregação 6, Cenário 4) | mesmo script |
| Cada consulta com descrição + código + resultado | ✅ | [`consultas/resultados_sparql.md`](../consultas/resultados_sparql.md) |
| Agregação (COUNT/SUM/AVG/GROUP BY) | ✅ | consultas 21–26 |
| Exemplos do enunciado reaproveitados | ✅ | consulta 7 (capivara em trecho próximo a banhado) |

## Relatório — Fase 6

| Requisito | Status | Onde |
|---|---|---|
| Decisões de modelagem justificadas | ✅ | [`DECISOES_MODELAGEM.md`](DECISOES_MODELAGEM.md) seção 2 |
| 2 alternativas descartadas explicadas | ✅ | mesmo arquivo, seção 3 |
| Limitações dos dados discutidas | ✅ | [`LIMITACOES_DADOS.md`](LIMITACOES_DADOS.md) |
| Checklist da rubrica | ✅ | este arquivo |

## Integração com ML — Fase 5 (opcional)

| Requisito | Status | Onde |
|---|---|---|
| Modelo de risco + predições simuladas | ⏳ opcional, **não implementado** | (Fase 5 não solicitada até o momento) |
| Explicação derivada da ontologia | ⏳ opcional | a base já existe: classe `TrechoCritico` + consultas 27–29 |

> A Fase 5 é **opcional**. A ontologia já está preparada para ela: a classe definida
> `TrechoCritico` e as consultas 27–29 fornecem os "fatos" para gerar explicações em
> linguagem natural caso o grupo decida implementá-la.

## Itens manuais finais — Fase 7 (responsabilidade do grupo)

| Item | Status |
|---|---|
| Abrir `taim.owl` no Protégé e capturar prints (Classes, Object Properties, OntoGraf, reasoner) | ✋ |
| Montar o PDF final juntando os `.md` + prints | ✋ |
| Gravar o vídeo de até 5 minutos | ✋ |

---

## Resumo

Todos os requisitos **obrigatórios** das Fases 1–4 e 6 estão atendidos. A Fase 5 (ML) é
opcional e não foi implementada. Restam apenas os itens **manuais** da Fase 7 (prints do
Protégé, PDF e vídeo), além do empacotamento final em `entregas/`.
