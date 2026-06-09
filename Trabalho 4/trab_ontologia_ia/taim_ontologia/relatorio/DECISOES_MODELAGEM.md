# Decisões de Modelagem — Ontologia do Banhado do Taim

Este documento justifica as principais decisões de modelagem da ontologia e apresenta
**duas alternativas que foram descartadas**, com os respectivos motivos. As referências
de linha apontam para [`scripts/01_construir_ontologia.py`](../scripts/01_construir_ontologia.py).

---

## 1. Visão geral do domínio

O domínio é o de **atropelamentos de fauna na BR-471, no Banhado do Taim (RS)**. O objetivo
da ontologia é representar *onde*, *quando*, *sob que condições* e *com quais espécies* os
atropelamentos ocorrem, de modo a permitir consultas analíticas (Fase 4) e, opcionalmente,
servir de camada de explicabilidade para um modelo de risco (Fase 5).

---

## 2. Principais decisões e suas justificativas

### 2.1 Reificar o atropelamento como a **classe** `Evento` / `EventoAtropelamento`
*(linhas 121–127)*

Um atropelamento **não** é uma simples relação entre uma rodovia e um animal: é um
acontecimento com **múltiplos participantes e atributos** — data/hora, trecho, clima,
período do dia, estação, número de animais. Para capturar tudo isso, o atropelamento foi
**reificado** como uma classe (`EventoAtropelamento`, subclasse de `Evento`), à qual se
ligam todas essas dimensões via *object properties* (`envolveAnimal`, `ocorreEmTrecho`,
`ocorreSobClima`, `ocorreNoPeriodo`, `ocorreNaEstacao`) e *data properties* (`dataHora`,
`numeroAnimaisEnvolvidos`). Esse é o padrão clássico de **reificação de evento n-ário** em
ontologias. (Ver alternativa descartada A, na seção 3.)

### 2.2 Condições ambientais como **classes**, não como strings
*(linhas 131–140)*

`CondicaoClimatica`, `PeriodoDia` e `Estacao` foram modeladas como **classes** com
indivíduos (`chuva_forte`, `neblina`, `madrugada`, `verao`...), e não como simples valores
de texto no evento. Isso permite (i) **anexar atributos** ao clima (`temperaturaC`,
`visibilidadeMetros`, `descricaoClima`), (ii) **consultar e agregar** por instância de clima
(ex.: "total de animais atropelados sob garoa", consulta 26) e (iii) **reaproveitar** a mesma
instância em vários eventos. (Ver alternativa descartada B, na seção 3.)

### 2.3 Escolha da **relação temporal**
*(propriedades nas linhas 175, 180, 223; data property na linha 263)*

A dimensão temporal é representada de forma redundante e complementar:
- `dataHora` (*data property*, **funcional** — linha 263): o instante exato do evento;
- `ocorreNoPeriodo` e `ocorreNaEstacao` (linhas 175, 180): granularidades qualitativas
  (madrugada/noite; estação) que são mais úteis para agregação e para a explicabilidade;
- `precedeEvento` (linha 223, **transitiva**): ordena eventos no tempo, permitindo cadeias
  temporais (`A precede B`, `B precede C` ⟹ `A precede C`).

A redundância é proposital: a data exata serve à precisão; o período/estação servem à
análise de padrões (atropelamentos concentram-se à noite — confirmado na consulta 25).

### 2.4 Escolha da **relação espacial**
*(propriedades nas linhas 185, 218; data properties 253, 258)*

A dimensão espacial usa:
- `proximoA` (linha 185): liga um `TrechoRodovia` a um `Habitat` — é o que conecta a
  infraestrutura viária ao ambiente da fauna e fundamenta a noção de trecho crítico;
- `adjacenteA` (linha 218, **simétrica**): vizinhança entre trechos contíguos da rodovia;
- `latitude`/`longitude` (linhas 253, 258): coordenadas geográficas, com **domínio em união**
  (`TrechoRodovia ∪ Habitat`), pois ambos os tipos possuem localização.

### 2.5 Características OWL e restrições
*(linhas 299–321)*

Marcaram-se características que enriquecem a inferência: `pertenceARodovia` **funcional**
(linha 190), `adjacenteA` **simétrica**, `precedeEvento` **transitiva**, `dataHora`
**funcional**. As restrições obrigatórias foram:
- `envolveAnimal min 1 Animal` em `EventoAtropelamento` (cardinalidade mínima — linha 299);
- `ocorreEmTrecho exactly 1 TrechoRodovia` (cardinalidade exata — linha 302);
- `Banhado contemVegetacao some Vegetacao` (existencial — linha 305);
- `EventoTravessia envolveAnimal only Animal` (universal — linha 308).

### 2.6 Classe **definida** `TrechoCritico` (equivalência)
*(linhas 312–319)*

`TrechoCritico ≡ TrechoRodovia ⊓ (proximoA some Banhado) ⊓ (temFatorRisco some
FatorProximidadeAgua)`. É uma **classe definida por condições necessárias e suficientes**, o
que permite ao **reasoner classificar automaticamente** quais trechos são críticos (na Fase 2
o HermiT classificou km506, km518, km530, km538). A mesma lógica foi replicada em SPARQL
(consulta 27) e os resultados **coincidiram** — validação cruzada entre reasoner e consulta.

### 2.7 Classes **disjuntas**
*(linhas 321–322)*

`AllDisjoint([Animal, Habitat, Evento, InfraestruturaViaria, CondicaoAmbiental, FatorRisco])`.
A disjunção entre as classes-topo permite ao reasoner **detectar inconsistências** (ex.: um
indivíduo que fosse simultaneamente `Animal` e `Habitat` tornaria a ontologia inconsistente).

---

## 3. Alternativas descartadas

### Alternativa A — Modelar atropelamento como propriedade binária `atropela(Rodovia, Animal)`

**Descrição:** em vez da classe `EventoAtropelamento`, usar uma *object property* direta
ligando rodovia e animal (`rodovia atropela animal`).

**Por que foi descartada:** uma relação binária só liga **dois** indivíduos e **não comporta
atributos**. Um atropelamento real tem data/hora, trecho específico, clima, período, estação e
pode envolver **vários animais** — nada disso caberia numa aresta `atropela`. Seria impossível
perguntar "quantos atropelamentos de capivara houve à noite sob neblina no km 530" (consultas
28–29). A reificação como classe (decisão 2.1) resolve isso ao transformar o evento em um
indivíduo de pleno direito, com todas as suas relações e dados.

### Alternativa B — Representar o clima como *data property* string no evento (`clima="chuva_forte"`)

**Descrição:** anexar o clima diretamente ao evento como um texto
(`evento ex:clima "chuva_forte"`), em vez da classe `CondicaoClimatica` com indivíduos.

**Por que foi descartada:** uma string é **opaca** — não se pode anexar a ela `temperaturaC`,
`visibilidadeMetros` nem agrupá-la de forma confiável (sujeita a erros de digitação:
"chuva_forte" vs "Chuva Forte"). Modelar o clima como **classe + indivíduos** (decisão 2.2)
permite anexar atributos quantitativos, reutilizar a mesma instância em vários eventos e fazer
agregações limpas (consulta 26, SUM por clima). Também deixa a ontologia extensível: novos
tipos de clima entram como novos indivíduos, sem mudar o esquema.

---

## 4. Resumo dos números do esquema

| Item | Quantidade | Mínimo do enunciado |
|---|---|---|
| Classes | 32 | 15 |
| Object properties | 14 | 10 |
| Data properties | 14 | 10 |
| Restrições | 4 + 1 classe definida | — |
| Relações temporais | `dataHora`, `ocorreNoPeriodo`, `ocorreNaEstacao`, `precedeEvento` | ≥1 |
| Relações espaciais | `proximoA`, `adjacenteA`, `latitude`, `longitude` | ≥1 |
