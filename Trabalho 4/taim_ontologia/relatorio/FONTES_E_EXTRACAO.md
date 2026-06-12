# Fontes de Informação e Extração de Conhecimento (Fase 3)

Este documento descreve **quais fontes** foram usadas, **como** os dados foram extraídos,
**como** foram integrados à ontologia e quais foram as **limitações e erros** do processo.
Foram implementadas **três** estratégias de extração (o enunciado pede no mínimo duas).

Script: [`scripts/03_extrair_conhecimento.py`](../scripts/03_extrair_conhecimento.py).
Dados brutos em [`dados_brutos/`](../dados_brutos/); protocolo do LLM em
[`protocolos_llm/`](../protocolos_llm/); log de execução em
[`dados_brutos/log_extracao.txt`](../dados_brutos/log_extracao.txt).

---

## 1. Wikipedia — *scraping* (requests + BeautifulSoup)

- **Fonte:** página em português *Estação Ecológica do Taim*
  (`https://pt.wikipedia.org/wiki/Estação_Ecológica_do_Taim`).
- **Como foi extraído:** download HTTP com `requests` (cabeçalho `User-Agent` próprio —
  obrigatório, ver limitações) e *parsing* do HTML com `BeautifulSoup`. Dentro do bloco de
  conteúdo (`#mw-content-text`) foram coletados:
  - **nomes científicos**: textos em itálico (`<i>`) no padrão binomial *Genus species*;
  - **termos de fauna**: textos de links que contêm palavras-chave de fauna
    (capivara, jacaré, ratão, bugio, garça, lontra, tachã, etc.).
- **Resultado:** 11 termos de fauna extraídos — ex.: *capivara, ratão-do-banhado,
  jacaré-de-papo-amarelo, biguá, cisne-de-pescoço-preto, garça-moura, lontra, tachã,
  tartaruga, tuco-tuco*. Salvo em `dados_brutos/wikipedia_termos.json`.
- **Uso na ontologia:** serviu de **validação ecológica** das espécies modeladas e apontou
  espécies reais ausentes da ABox (ratão-do-banhado), depois incorporadas via Wikidata/LLM.

## 2. Wikidata — SPARQL (endpoint oficial)

- **Fonte:** *Wikidata Query Service* (`https://query.wikidata.org/sparql`).
- **Como foi extraído:** consulta SPARQL parametrizada por nome científico (`VALUES`),
  buscando o item (`wdt:P225` = nome do táxon) e o nome vernáculo em português
  (`wdt:P1843`, `FILTER(LANG = "pt")`). Resposta em JSON, salva em
  `dados_brutos/wikidata_taxa.json`.
- **Resultado (QIDs resolvidos):**

  | Nome científico | QID Wikidata | Vernáculo (pt) |
  |---|---|---|
  | *Hydrochoerus hydrochaeris* (capivara) | Q131538 | — |
  | *Caiman latirostris* (jacaré-do-papo-amarelo) | Q644453 | — |
  | *Myocastor coypus* (ratão-do-banhado) | Q187704 | — |
  | *Blastocerus dichotomus* (cervo-do-pantanal) | Q504501 | — |
  | *Jacana jacana* (jaçanã) | Q856201 | jaçanã, jaçanã-de-fronte-vermelha |

- **Uso na ontologia:** os QIDs viraram **proveniência rastreável** (`rdfs:comment`) e
  fundamentaram a criação de novos indivíduos (cervo-do-pantanal, jaçanã) e o enriquecimento
  de existentes (capivara_01 ganhou referência ao QID Q131538).

## 3. LLM — extração de triplas (protocolo revisado)

- **Fonte:** texto ecológico sobre a ESEC Taim (`dados_brutos/amostra_local_taim.txt`),
  submetido a um LLM (Claude 4.x) com um prompt de extração de triplas.
- **Como foi extraído:** o **protocolo completo** (prompt exato + resposta bruta + revisão)
  está em [`protocolos_llm/protocolo_extracao_llm.md`](../protocolos_llm/protocolo_extracao_llm.md).
  A saída bruta teve **11 triplas**; após a **revisão humana obrigatória**, 3 foram
  aproveitadas e mapeadas para propriedades da ontologia (`habita`, `atravessa`), salvas em
  `dados_brutos/llm_triplas_revisadas.json`.
- **Uso na ontologia:** a tripla *(Ratão-do-banhado | habita | Banhado)* gerou o indivíduo
  `ratao_do_banhado` (cujo QID foi cruzado com o Wikidata — Q187704).

---

## 4. Integração na ontologia (rastreabilidade)

Três indivíduos **derivam diretamente da extração** e carregam `rdfs:comment` com a fonte:

| Indivíduo | Classe | Fonte | Exemplo no estilo do enunciado |
|---|---|---|---|
| `ratao_do_banhado` | `Mamifero` | LLM + Wikidata (Q187704) | `ratao_do_banhado rdf:type Mamifero; habita banhado_do_taim; atravessa BR_471` |
| `cervo_do_pantanal` | `Mamifero` | Wikidata (Q504501) | `cervo_do_pantanal rdf:type Mamifero; viveEm area_alagada_sul` |
| `jacana` | `AveAquatica` | Wikidata (Q856201) | `jacana rdf:type AveAquatica; viveEm lagoa_mangueira` |

Total de indivíduos passou de **108 → 111**. O reasoner HermiT roda **sem inconsistência**
após a integração.

---

## 5. Limitações e erros do processo

- **Wikipedia exige `User-Agent`:** sem cabeçalho, a Wikipedia retorna **HTTP 403** (bloqueio
  por política anti-bot). Foi necessário enviar um `User-Agent` identificável. O script tem
  *fallback*: se a página estiver inacessível, extrai termos do texto local
  `amostra_local_taim.txt` e registra a tentativa em `log_extracao.txt`.
- **Nomes científicos via itálico:** a heurística de binomiais em `<i>` retornou **0** nesta
  página (os táxons aparecem mais como links de nomes comuns do que em itálico). Não é erro
  de rede, e sim limitação da heurística; os termos de fauna por palavra-chave compensaram.
- **Vernáculos do Wikidata incompletos:** `wdt:P1843` em português só existia para a jaçanã;
  para os demais táxons o nome vernáculo veio vazio e usamos um rótulo padrão. Mostra que a
  cobertura do Wikidata é desigual entre espécies.
- **Alucinação/ruído do LLM:** a saída bruta continha triplas **malformadas** (objeto vazio
  "-") e **não-normalizadas** (verbo misturado com período), descartadas na revisão. Reforça
  que a saída de LLM **não pode ser usada sem revisão** — risco de inserir fatos inválidos.
- **Dependência de rede:** as estratégias 1 e 2 dependem de serviços externos que podem ficar
  fora do ar ou mudar de formato; por isso há captura de exceção, *log* e *fallback*.
- **Dados de coordenadas/atributos continuam sintéticos:** a extração enriquece *espécies e
  proveniência*, mas atributos quantitativos (lat/long dos trechos, datas dos eventos) seguem
  sintéticos — ver `relatorio/LIMITACOES_DADOS.md` (Fase 6).
