# Limitações dos Dados e do Povoamento

Este documento discute as **limitações, vieses e inconsistências** do povoamento da ontologia
(Fase 2) e da extração de conhecimento (Fase 3). É uma exigência da rubrica reconhecer
honestamente o que os dados **não** garantem.

---

## 1. Os dados são SINTÉTICOS (gerados, não medidos)

Os ~108 indivíduos foram **gerados por script** ([`02_povoar_ontologia.py`](../scripts/02_povoar_ontologia.py))
com `random.seed(42)` para reprodutibilidade. **Não** provêm de uma base real de
atropelamentos (como os registros do CBEE/UFLA ou de PMVS). Em consequência:

- As **datas/horas** dos 50 eventos são plausíveis, mas fictícias.
- Os **trechos** (km 498–542) e suas coordenadas são aproximações, não levantamentos de campo.
- O **número de animais por evento** e o **nível de risco** dos trechos foram sorteados.

**Uso adequado:** a ontologia demonstra a *capacidade de representação e consulta*; os números
**não** devem ser lidos como estatística real do Taim.

## 2. Vieses introduzidos de propósito (plausibilidade)

Para que os dados fossem *plausíveis*, o gerador embutiu vieses ancorados em fatos conhecidos:

- **Viés noturno:** o horário dos eventos é sorteado com peso maior para madrugada/noite, pois
  é quando a fauna mais atravessa a rodovia. Isso aparece nas consultas (noite + madrugada
  concentram a maioria) — **mas é um viés que nós impusemos**, não um achado.
- **Viés de baixa visibilidade à noite:** eventos noturnos recebem com mais frequência os
  climas `neblina`/`chuva_forte`. De novo, é uma suposição modelada, não medida.
- **Capivara como espécie emblemática:** ela é a espécie mais associada a atropelamentos no
  Taim na literatura, e o povoamento reflete isso — porém a distribuição exata por espécie é
  arbitrária.

Esses vieses são **circulares** para fins analíticos: as consultas "descobrem" padrões que
foram colocados ali pela geração. Isso é aceitável para um trabalho de modelagem, desde que
fique explícito (como aqui).

## 3. Coordenadas geográficas aproximadas

`latitude`/`longitude` partem da base real da região (≈ -32.5, -52.5, área da ESEC Taim /
Lagoa Mangueira) com **ruído aleatório de ±0,2°**. Isso equivale a uma dispersão de dezenas de
quilômetros — **bom o suficiente para localização aproximada**, mas **insuficiente** para
qualquer análise espacial fina (a posição de um trecho específico não é o ponto real da BR-471).

## 4. Limitações da extração de conhecimento (Fase 3)

### 4.1 Scraping de Wikipedia
- Depende de **disponibilidade de rede** e do **`User-Agent`** (sem ele, HTTP 403).
- A heurística de **nomes científicos em itálico** retornou 0 nesta página (os táxons aparecem
  como links de nomes comuns), ou seja, a extração é **sensível ao layout** da página.
- Termos extraídos por palavra-chave podem trazer **ruído** (ex.: "Ronco do Bugio", que é o
  nome de uma área de proteção, não a espécie) — exige curadoria.

### 4.2 Wikidata (SPARQL)
- A cobertura de **nomes vernáculos em português** (`P1843`) é **desigual**: só a jaçanã trouxe
  vernáculo; os demais táxons vieram sem, exigindo rótulo padrão.
- A consulta depende do **endpoint** estar no ar e pode sofrer *timeout*/limites de uso.

### 4.3 LLM (extração de triplas) — risco de alucinação
- A saída bruta do LLM continha triplas **malformadas** (objeto vazio "-") e
  **não-normalizadas** (verbo misturado com período), que **só não entraram na ontologia
  porque houve revisão humana** (ver [protocolo](../protocolos_llm/protocolo_extracao_llm.md)).
- Generalizando: **nenhuma** saída de LLM deve ser inserida sem revisão — o risco de inserir
  fatos inventados (alucinação) é real e foi observado neste próprio trabalho.

## 5. Inconsistências potenciais e como foram tratadas

- **Mundo aberto (OWA):** OWL adota *open-world*; a ausência de um fato não é tratada como
  falso. Por isso a restrição `Banhado contemVegetacao some Vegetacao` **não** seria violada se
  esquecêssemos de ligar a vegetação — mas o povoamento liga explicitamente
  `banhado_do_taim → vegetacao_palustre/mata_ciliar` para que o fato fique afirmado.
- **Disjunção:** como as classes-topo são disjuntas, qualquer indivíduo com tipos conflitantes
  tornaria a ontologia inconsistente. O reasoner HermiT foi executado nas Fases 1, 2 e 3 e
  **não acusou inconsistência**.
- **Cardinalidade dos eventos:** o script valida, no nível dos dados, que **todo**
  `EventoAtropelamento` tem `dataHora`, ≥1 animal e exatamente 1 trecho — coerente com as
  restrições da TBox.

## 6. O que seria necessário para um uso real

- Substituir os dados sintéticos por **registros reais** de atropelamento (com licença de uso).
- **Georreferenciar** os trechos com precisão (GPS / OpenStreetMap).
- Validar espécies e ocorrências com **especialistas** e bases como GBIF/SiBBr.
- Ampliar a extração com **revisão sistemática** e múltiplas execuções para medir reprodutibilidade.
