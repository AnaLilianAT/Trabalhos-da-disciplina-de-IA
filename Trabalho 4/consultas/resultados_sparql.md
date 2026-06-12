# Resultados das Consultas SPARQL — Ontologia do Taim

Geradas por [`scripts/04_consultas_sparql.py`](../scripts/04_consultas_sparql.py) sobre `taim.owl` (1080 triplas), via **rdflib** (SPARQL 1.1).

**Total: 32 consultas** — Simples: 7, Múltiplas relações: 9, Com filtros: 6, Agregação: 6, Cenário do domínio: 4.

Todas usam os prefixos:

```sparql
PREFIX ex:   <http://www.taim.org/ontologia#>
PREFIX rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX xsd:  <http://www.w3.org/2001/XMLSchema#>
```

---

## Consulta 1 — Simples

**Descrição:** Listar todos os eventos de atropelamento.

```sparql
SELECT ?evento WHERE {
  ?evento rdf:type ex:EventoAtropelamento .
} ORDER BY ?evento
```

**Resultado:**

| evento |
| --- |
| ex:atropelamento_001 |
| ex:atropelamento_002 |
| ex:atropelamento_003 |
| ex:atropelamento_004 |
| ex:atropelamento_005 |
| ex:atropelamento_006 |
| ex:atropelamento_007 |
| ex:atropelamento_008 |
| ex:atropelamento_009 |
| ex:atropelamento_010 |
| ex:atropelamento_011 |
| ex:atropelamento_012 |
| ex:atropelamento_013 |
| ex:atropelamento_014 |
| ex:atropelamento_015 |
| ex:atropelamento_016 |
| ex:atropelamento_017 |
| ex:atropelamento_018 |
| ex:atropelamento_019 |
| ex:atropelamento_020 |
| ex:atropelamento_021 |
| ex:atropelamento_022 |
| ex:atropelamento_023 |
| ex:atropelamento_024 |
| ex:atropelamento_025 |

_(50 linhas no total; exibindo as primeiras 25.)_


---

## Consulta 2 — Simples

**Descrição:** Listar todas as capivaras (indivíduos da classe Capivara).

```sparql
SELECT ?capivara ?nome WHERE {
  ?capivara rdf:type ex:Capivara .
  OPTIONAL { ?capivara ex:nomeComum ?nome }
} ORDER BY ?capivara
```

**Resultado:**

| capivara | nome |
| --- | --- |
| ex:capivara_01 | Capivara |
| ex:capivara_02 | Capivara |
| ex:capivara_03 | Capivara |


---

## Consulta 3 — Simples

**Descrição:** Listar todos os trechos de rodovia com seus km de início e fim.

```sparql
SELECT ?trecho ?kmInicio ?kmFim WHERE {
  ?trecho rdf:type ex:TrechoRodovia .
  ?trecho ex:kmInicio ?kmInicio .
  ?trecho ex:kmFim ?kmFim .
} ORDER BY ?kmInicio
```

**Resultado:**

| trecho | kmInicio | kmFim |
| --- | --- | --- |
| ex:trecho_BR471_km498 | 498.0 | 502.0 |
| ex:trecho_BR471_km502 | 502.0 | 506.0 |
| ex:trecho_BR471_km506 | 506.0 | 510.0 |
| ex:trecho_BR471_km510 | 510.0 | 514.0 |
| ex:trecho_BR471_km514 | 514.0 | 518.0 |
| ex:trecho_BR471_km518 | 518.0 | 522.0 |
| ex:trecho_BR471_km522 | 522.0 | 526.0 |
| ex:trecho_BR471_km526 | 526.0 | 530.0 |
| ex:trecho_BR471_km530 | 530.0 | 534.0 |
| ex:trecho_BR471_km534 | 534.0 | 538.0 |
| ex:trecho_BR471_km538 | 538.0 | 542.0 |
| ex:trecho_BR471_km542 | 542.0 | 546.0 |


---

## Consulta 4 — Simples

**Descrição:** Listar todos os animais (qualquer subclasse de Animal) e seu nome comum.

```sparql
SELECT ?animal ?nomeComum WHERE {
  ?animal rdf:type/rdfs:subClassOf* ex:Animal .
  ?animal ex:nomeComum ?nomeComum .
} ORDER BY ?nomeComum
```

**Resultado:**

| animal | nomeComum |
| --- | --- |
| ex:bugio_01 | Bugio-ruivo |
| ex:bugio_02 | Bugio-ruivo |
| ex:capivara_01 | Capivara |
| ex:capivara_02 | Capivara |
| ex:capivara_03 | Capivara |
| ex:cervo_do_pantanal | Cervo-do-pantanal |
| ex:colhereiro_01 | Colhereiro |
| ex:cagado_01 | Cágado-de-barbicha |
| ex:cagado_02 | Cágado-de-barbicha |
| ex:frango_dagua_01 | Frango-d'água |
| ex:garca_branca_01 | Garça-branca-grande |
| ex:graxaim_01 | Graxaim-do-mato |
| ex:graxaim_02 | Graxaim-do-mato |
| ex:jacare_01 | Jacaré-do-papo-amarelo |
| ex:jacare_02 | Jacaré-do-papo-amarelo |
| ex:marreco_01 | Marreco-pardo |
| ex:ratao_do_banhado | Ratão-do-banhado |
| ex:jacana | jaçanã |


---

## Consulta 5 — Simples

**Descrição:** Listar todas as condições climáticas com sua descrição.

```sparql
SELECT ?clima ?descricao WHERE {
  ?clima rdf:type ex:CondicaoClimatica .
  ?clima ex:descricaoClima ?descricao .
} ORDER BY ?clima
```

**Resultado:**

| clima | descricao |
| --- | --- |
| ex:chuva_forte | Chuva forte com redução de visibilidade |
| ex:garoa | Garoa/chuvisco |
| ex:neblina | Neblina densa, comum nas madrugadas do Taim |
| ex:sol | Tempo bom, céu claro |


---

## Consulta 6 — Simples

**Descrição:** Listar todos os habitats (qualquer subclasse de Habitat).

```sparql
SELECT DISTINCT ?habitat WHERE {
  ?habitat rdf:type/rdfs:subClassOf* ex:Habitat .
} ORDER BY ?habitat
```

**Resultado:**

| habitat |
| --- |
| ex:area_alagada_norte |
| ex:area_alagada_sul |
| ex:banhado_do_taim |
| ex:canal_sao_goncalo |
| ex:canal_taim |
| ex:lagoa_mangueira |
| ex:lagoa_mirim |
| ex:lagoa_nicola |
| ex:vegetacao_mata_ciliar |
| ex:vegetacao_palustre |


---

## Consulta 7 — Múltiplas relações

**Descrição:** Atropelamentos de capivara em trecho próximo a um banhado (exemplo do enunciado).

```sparql
SELECT ?evento ?capivara ?trecho ?banhado WHERE {
  ?evento rdf:type ex:EventoAtropelamento ;
          ex:envolveAnimal ?capivara ;
          ex:ocorreEmTrecho ?trecho .
  ?capivara rdf:type ex:Capivara .
  ?trecho ex:proximoA ?banhado .
  ?banhado rdf:type ex:Banhado .
} ORDER BY ?evento
```

**Resultado:**

| evento | capivara | trecho | banhado |
| --- | --- | --- | --- |
| ex:atropelamento_006 | ex:capivara_02 | ex:trecho_BR471_km518 | ex:banhado_do_taim |
| ex:atropelamento_011 | ex:capivara_01 | ex:trecho_BR471_km538 | ex:banhado_do_taim |
| ex:atropelamento_015 | ex:capivara_01 | ex:trecho_BR471_km506 | ex:banhado_do_taim |
| ex:atropelamento_018 | ex:capivara_01 | ex:trecho_BR471_km530 | ex:banhado_do_taim |
| ex:atropelamento_018 | ex:capivara_03 | ex:trecho_BR471_km530 | ex:banhado_do_taim |
| ex:atropelamento_021 | ex:capivara_03 | ex:trecho_BR471_km538 | ex:banhado_do_taim |
| ex:atropelamento_033 | ex:capivara_03 | ex:trecho_BR471_km506 | ex:banhado_do_taim |
| ex:atropelamento_050 | ex:capivara_03 | ex:trecho_BR471_km538 | ex:banhado_do_taim |


---

## Consulta 8 — Múltiplas relações

**Descrição:** Cada animal e os habitats que ele habita.

```sparql
SELECT ?animal ?nome ?habitat WHERE {
  ?animal rdf:type/rdfs:subClassOf* ex:Animal ;
          ex:habita ?habitat .
  OPTIONAL { ?animal ex:nomeComum ?nome }
} ORDER BY ?animal
```

**Resultado:**

| animal | nome | habitat |
| --- | --- | --- |
| ex:bugio_01 | Bugio-ruivo | ex:canal_taim |
| ex:bugio_02 | Bugio-ruivo | ex:canal_sao_goncalo |
| ex:cagado_01 | Cágado-de-barbicha | ex:canal_taim |
| ex:cagado_02 | Cágado-de-barbicha | ex:canal_sao_goncalo |
| ex:capivara_01 | Capivara | ex:banhado_do_taim |
| ex:capivara_02 | Capivara | ex:lagoa_mirim |
| ex:capivara_03 | Capivara | ex:area_alagada_norte |
| ex:colhereiro_01 | Colhereiro | ex:area_alagada_norte |
| ex:frango_dagua_01 | Frango-d'água | ex:lagoa_mangueira |
| ex:garca_branca_01 | Garça-branca-grande | ex:canal_taim |
| ex:graxaim_01 | Graxaim-do-mato | ex:lagoa_mirim |
| ex:graxaim_02 | Graxaim-do-mato | ex:lagoa_nicola |
| ex:jacare_01 | Jacaré-do-papo-amarelo | ex:lagoa_mangueira |
| ex:jacare_02 | Jacaré-do-papo-amarelo | ex:canal_taim |
| ex:marreco_01 | Marreco-pardo | ex:lagoa_mangueira |
| ex:ratao_do_banhado | Ratão-do-banhado | ex:banhado_do_taim |


---

## Consulta 9 — Múltiplas relações

**Descrição:** Para cada evento, o trecho onde ocorreu e a rodovia a que o trecho pertence.

```sparql
SELECT ?evento ?trecho ?rodovia WHERE {
  ?evento ex:ocorreEmTrecho ?trecho .
  ?trecho ex:pertenceARodovia ?rodovia .
} ORDER BY ?evento
```

**Resultado:**

| evento | trecho | rodovia |
| --- | --- | --- |
| ex:atropelamento_001 | ex:trecho_BR471_km526 | ex:BR_471 |
| ex:atropelamento_002 | ex:trecho_BR471_km534 | ex:BR_471 |
| ex:atropelamento_003 | ex:trecho_BR471_km530 | ex:BR_471 |
| ex:atropelamento_004 | ex:trecho_BR471_km534 | ex:BR_471 |
| ex:atropelamento_005 | ex:trecho_BR471_km538 | ex:BR_471 |
| ex:atropelamento_006 | ex:trecho_BR471_km518 | ex:BR_471 |
| ex:atropelamento_007 | ex:trecho_BR471_km542 | ex:BR_471 |
| ex:atropelamento_008 | ex:trecho_BR471_km510 | ex:BR_471 |
| ex:atropelamento_009 | ex:trecho_BR471_km534 | ex:BR_471 |
| ex:atropelamento_010 | ex:trecho_BR471_km502 | ex:BR_471 |
| ex:atropelamento_011 | ex:trecho_BR471_km538 | ex:BR_471 |
| ex:atropelamento_012 | ex:trecho_BR471_km510 | ex:BR_471 |
| ex:atropelamento_013 | ex:trecho_BR471_km502 | ex:BR_471 |
| ex:atropelamento_014 | ex:trecho_BR471_km510 | ex:BR_471 |
| ex:atropelamento_015 | ex:trecho_BR471_km506 | ex:BR_471 |
| ex:atropelamento_016 | ex:trecho_BR471_km522 | ex:BR_471 |
| ex:atropelamento_017 | ex:trecho_BR471_km498 | ex:BR_471 |
| ex:atropelamento_018 | ex:trecho_BR471_km530 | ex:BR_471 |
| ex:atropelamento_019 | ex:trecho_BR471_km534 | ex:BR_471 |
| ex:atropelamento_020 | ex:trecho_BR471_km534 | ex:BR_471 |
| ex:atropelamento_021 | ex:trecho_BR471_km538 | ex:BR_471 |
| ex:atropelamento_022 | ex:trecho_BR471_km502 | ex:BR_471 |
| ex:atropelamento_023 | ex:trecho_BR471_km502 | ex:BR_471 |
| ex:atropelamento_024 | ex:trecho_BR471_km534 | ex:BR_471 |
| ex:atropelamento_025 | ex:trecho_BR471_km502 | ex:BR_471 |

_(50 linhas no total; exibindo as primeiras 25.)_


---

## Consulta 10 — Múltiplas relações

**Descrição:** Eventos com o animal envolvido e o clima sob o qual ocorreram.

```sparql
SELECT ?evento ?animal ?clima WHERE {
  ?evento ex:envolveAnimal ?animal ;
          ex:ocorreSobClima ?clima .
} ORDER BY ?evento
```

**Resultado:**

| evento | animal | clima |
| --- | --- | --- |
| ex:atropelamento_001 | ex:graxaim_01 | ex:chuva_forte |
| ex:atropelamento_002 | ex:cagado_02 | ex:chuva_forte |
| ex:atropelamento_003 | ex:jacare_02 | ex:neblina |
| ex:atropelamento_004 | ex:marreco_01 | ex:sol |
| ex:atropelamento_005 | ex:colhereiro_01 | ex:garoa |
| ex:atropelamento_006 | ex:capivara_02 | ex:chuva_forte |
| ex:atropelamento_007 | ex:capivara_02 | ex:chuva_forte |
| ex:atropelamento_008 | ex:marreco_01 | ex:garoa |
| ex:atropelamento_009 | ex:cagado_02 | ex:sol |
| ex:atropelamento_010 | ex:capivara_02 | ex:sol |
| ex:atropelamento_011 | ex:capivara_01 | ex:neblina |
| ex:atropelamento_011 | ex:colhereiro_01 | ex:neblina |
| ex:atropelamento_012 | ex:cagado_02 | ex:sol |
| ex:atropelamento_012 | ex:bugio_01 | ex:sol |
| ex:atropelamento_013 | ex:cagado_01 | ex:sol |
| ex:atropelamento_014 | ex:cagado_01 | ex:chuva_forte |
| ex:atropelamento_014 | ex:capivara_01 | ex:chuva_forte |
| ex:atropelamento_014 | ex:capivara_02 | ex:chuva_forte |
| ex:atropelamento_015 | ex:cagado_02 | ex:sol |
| ex:atropelamento_015 | ex:capivara_01 | ex:sol |
| ex:atropelamento_016 | ex:garca_branca_01 | ex:sol |
| ex:atropelamento_016 | ex:bugio_02 | ex:sol |
| ex:atropelamento_017 | ex:marreco_01 | ex:garoa |
| ex:atropelamento_017 | ex:frango_dagua_01 | ex:garoa |
| ex:atropelamento_017 | ex:jacare_01 | ex:garoa |

_(79 linhas no total; exibindo as primeiras 25.)_


---

## Consulta 11 — Múltiplas relações

**Descrição:** Trechos e os habitats aos quais estão próximos.

```sparql
SELECT ?trecho ?habitat WHERE {
  ?trecho rdf:type ex:TrechoRodovia ;
          ex:proximoA ?habitat .
} ORDER BY ?trecho
```

**Resultado:**

| trecho | habitat |
| --- | --- |
| ex:trecho_BR471_km498 | ex:lagoa_mangueira |
| ex:trecho_BR471_km502 | ex:canal_sao_goncalo |
| ex:trecho_BR471_km506 | ex:banhado_do_taim |
| ex:trecho_BR471_km510 | ex:lagoa_mirim |
| ex:trecho_BR471_km514 | ex:lagoa_nicola |
| ex:trecho_BR471_km518 | ex:banhado_do_taim |
| ex:trecho_BR471_km522 | ex:area_alagada_norte |
| ex:trecho_BR471_km526 | ex:area_alagada_norte |
| ex:trecho_BR471_km530 | ex:banhado_do_taim |
| ex:trecho_BR471_km534 | ex:area_alagada_sul |
| ex:trecho_BR471_km538 | ex:banhado_do_taim |
| ex:trecho_BR471_km542 | ex:lagoa_mirim |


---

## Consulta 12 — Múltiplas relações

**Descrição:** Atropelamentos noturnos (madrugada ou noite) com a espécie do animal envolvido.

```sparql
SELECT ?evento ?periodo ?especie WHERE {
  ?evento ex:ocorreNoPeriodo ?periodo ;
          ex:envolveAnimal ?animal .
  ?animal rdf:type ?especie .
  ?especie rdfs:subClassOf* ex:Animal .
  FILTER (?periodo IN (ex:madrugada, ex:noite))
} ORDER BY ?evento
```

**Resultado:**

| evento | periodo | especie |
| --- | --- | --- |
| ex:atropelamento_002 | ex:noite | ex:Cagado |
| ex:atropelamento_003 | ex:noite | ex:Jacare |
| ex:atropelamento_004 | ex:noite | ex:AveAquatica |
| ex:atropelamento_005 | ex:noite | ex:AveAquatica |
| ex:atropelamento_006 | ex:noite | ex:Capivara |
| ex:atropelamento_007 | ex:noite | ex:Capivara |
| ex:atropelamento_008 | ex:madrugada | ex:AveAquatica |
| ex:atropelamento_009 | ex:noite | ex:Cagado |
| ex:atropelamento_010 | ex:noite | ex:Capivara |
| ex:atropelamento_011 | ex:madrugada | ex:Capivara |
| ex:atropelamento_011 | ex:madrugada | ex:AveAquatica |
| ex:atropelamento_012 | ex:noite | ex:Bugio |
| ex:atropelamento_012 | ex:noite | ex:Cagado |
| ex:atropelamento_013 | ex:madrugada | ex:Cagado |
| ex:atropelamento_014 | ex:madrugada | ex:Capivara |
| ex:atropelamento_014 | ex:madrugada | ex:Capivara |
| ex:atropelamento_014 | ex:madrugada | ex:Cagado |
| ex:atropelamento_015 | ex:noite | ex:Capivara |
| ex:atropelamento_015 | ex:noite | ex:Cagado |
| ex:atropelamento_016 | ex:noite | ex:Bugio |
| ex:atropelamento_016 | ex:noite | ex:AveAquatica |
| ex:atropelamento_017 | ex:noite | ex:AveAquatica |
| ex:atropelamento_017 | ex:noite | ex:AveAquatica |
| ex:atropelamento_017 | ex:noite | ex:Jacare |
| ex:atropelamento_018 | ex:noite | ex:Capivara |

_(71 linhas no total; exibindo as primeiras 25.)_


---

## Consulta 13 — Múltiplas relações

**Descrição:** Animais que atravessam a BR-471 e onde habitam.

```sparql
SELECT ?animal ?nome ?habitat WHERE {
  ?animal ex:atravessa ex:BR_471 .
  OPTIONAL { ?animal ex:nomeComum ?nome }
  OPTIONAL { ?animal ex:habita ?habitat }
} ORDER BY ?animal
```

**Resultado:**

| animal | nome | habitat |
| --- | --- | --- |
| ex:bugio_01 | Bugio-ruivo | ex:canal_taim |
| ex:bugio_02 | Bugio-ruivo | ex:canal_sao_goncalo |
| ex:cagado_01 | Cágado-de-barbicha | ex:canal_taim |
| ex:cagado_02 | Cágado-de-barbicha | ex:canal_sao_goncalo |
| ex:capivara_02 | Capivara | ex:lagoa_mirim |
| ex:capivara_03 | Capivara | ex:area_alagada_norte |
| ex:colhereiro_01 | Colhereiro | ex:area_alagada_norte |
| ex:frango_dagua_01 | Frango-d'água | ex:lagoa_mangueira |
| ex:garca_branca_01 | Garça-branca-grande | ex:canal_taim |
| ex:marreco_01 | Marreco-pardo | ex:lagoa_mangueira |
| ex:ratao_do_banhado | Ratão-do-banhado | ex:banhado_do_taim |


---

## Consulta 14 — Múltiplas relações

**Descrição:** Eventos e os fatores de risco do trecho em que ocorreram.

```sparql
SELECT ?evento ?trecho ?fator WHERE {
  ?evento ex:ocorreEmTrecho ?trecho .
  ?trecho ex:temFatorRisco ?fator .
} ORDER BY ?trecho ?evento
```

**Resultado:**

| evento | trecho | fator |
| --- | --- | --- |
| ex:atropelamento_017 | ex:trecho_BR471_km498 | ex:trafego_baixo |
| ex:atropelamento_017 | ex:trecho_BR471_km498 | ex:visibilidade_media |
| ex:atropelamento_027 | ex:trecho_BR471_km498 | ex:trafego_baixo |
| ex:atropelamento_027 | ex:trecho_BR471_km498 | ex:visibilidade_media |
| ex:atropelamento_031 | ex:trecho_BR471_km498 | ex:trafego_baixo |
| ex:atropelamento_031 | ex:trecho_BR471_km498 | ex:visibilidade_media |
| ex:atropelamento_036 | ex:trecho_BR471_km498 | ex:trafego_baixo |
| ex:atropelamento_036 | ex:trecho_BR471_km498 | ex:visibilidade_media |
| ex:atropelamento_048 | ex:trecho_BR471_km498 | ex:trafego_baixo |
| ex:atropelamento_048 | ex:trecho_BR471_km498 | ex:visibilidade_media |
| ex:atropelamento_010 | ex:trecho_BR471_km502 | ex:trafego_alto |
| ex:atropelamento_013 | ex:trecho_BR471_km502 | ex:trafego_alto |
| ex:atropelamento_022 | ex:trecho_BR471_km502 | ex:trafego_alto |
| ex:atropelamento_023 | ex:trecho_BR471_km502 | ex:trafego_alto |
| ex:atropelamento_025 | ex:trecho_BR471_km502 | ex:trafego_alto |
| ex:atropelamento_029 | ex:trecho_BR471_km502 | ex:trafego_alto |
| ex:atropelamento_038 | ex:trecho_BR471_km502 | ex:trafego_alto |
| ex:atropelamento_042 | ex:trecho_BR471_km502 | ex:trafego_alto |
| ex:atropelamento_047 | ex:trecho_BR471_km502 | ex:trafego_alto |
| ex:atropelamento_015 | ex:trecho_BR471_km506 | ex:trafego_medio |
| ex:atropelamento_015 | ex:trecho_BR471_km506 | ex:visibilidade_media |
| ex:atropelamento_015 | ex:trecho_BR471_km506 | ex:proximidade_agua_alta |
| ex:atropelamento_028 | ex:trecho_BR471_km506 | ex:trafego_medio |
| ex:atropelamento_028 | ex:trecho_BR471_km506 | ex:visibilidade_media |
| ex:atropelamento_028 | ex:trecho_BR471_km506 | ex:proximidade_agua_alta |

_(90 linhas no total; exibindo as primeiras 25.)_


---

## Consulta 15 — Com filtros

**Descrição:** Eventos ocorridos sob chuva forte.

```sparql
SELECT ?evento WHERE {
  ?evento ex:ocorreSobClima ex:chuva_forte .
} ORDER BY ?evento
```

**Resultado:**

| evento |
| --- |
| ex:atropelamento_001 |
| ex:atropelamento_002 |
| ex:atropelamento_006 |
| ex:atropelamento_007 |
| ex:atropelamento_014 |
| ex:atropelamento_018 |
| ex:atropelamento_019 |
| ex:atropelamento_025 |
| ex:atropelamento_027 |
| ex:atropelamento_029 |
| ex:atropelamento_035 |
| ex:atropelamento_041 |
| ex:atropelamento_043 |
| ex:atropelamento_049 |


---

## Consulta 16 — Com filtros

**Descrição:** Eventos ocorridos no período da madrugada.

```sparql
SELECT ?evento WHERE {
  ?evento ex:ocorreNoPeriodo ex:madrugada .
} ORDER BY ?evento
```

**Resultado:**

| evento |
| --- |
| ex:atropelamento_008 |
| ex:atropelamento_011 |
| ex:atropelamento_013 |
| ex:atropelamento_014 |
| ex:atropelamento_019 |
| ex:atropelamento_020 |
| ex:atropelamento_022 |
| ex:atropelamento_024 |
| ex:atropelamento_026 |
| ex:atropelamento_033 |
| ex:atropelamento_035 |
| ex:atropelamento_038 |
| ex:atropelamento_041 |
| ex:atropelamento_042 |
| ex:atropelamento_043 |
| ex:atropelamento_044 |
| ex:atropelamento_045 |
| ex:atropelamento_049 |
| ex:atropelamento_050 |


---

## Consulta 17 — Com filtros

**Descrição:** Eventos sob clima com temperatura abaixo de 15 °C (FILTER em data property).

```sparql
SELECT ?evento ?clima ?temp WHERE {
  ?evento ex:ocorreSobClima ?clima .
  ?clima ex:temperaturaC ?temp .
  FILTER (?temp < 15.0)
} ORDER BY ?temp
```

**Resultado:**

| evento | clima | temp |
| --- | --- | --- |
| ex:atropelamento_003 | ex:neblina | 14.0 |
| ex:atropelamento_011 | ex:neblina | 14.0 |
| ex:atropelamento_022 | ex:neblina | 14.0 |
| ex:atropelamento_023 | ex:neblina | 14.0 |
| ex:atropelamento_024 | ex:neblina | 14.0 |
| ex:atropelamento_036 | ex:neblina | 14.0 |
| ex:atropelamento_046 | ex:neblina | 14.0 |
| ex:atropelamento_047 | ex:neblina | 14.0 |
| ex:atropelamento_048 | ex:neblina | 14.0 |
| ex:atropelamento_050 | ex:neblina | 14.0 |


---

## Consulta 18 — Com filtros

**Descrição:** Trechos com nível de risco acima de 0,7 (FILTER).

```sparql
SELECT ?trecho ?nivelRisco WHERE {
  ?trecho ex:nivelRisco ?nivelRisco .
  FILTER (?nivelRisco > 0.7)
} ORDER BY DESC(?nivelRisco)
```

**Resultado:**

| trecho | nivelRisco |
| --- | --- |
| ex:trecho_BR471_km514 | 0.94 |
| ex:trecho_BR471_km538 | 0.85 |
| ex:trecho_BR471_km542 | 0.84 |


---

## Consulta 19 — Com filtros

**Descrição:** Animais de grande porte: peso médio acima de 20 kg (FILTER).

```sparql
SELECT ?animal ?nome ?peso WHERE {
  ?animal ex:pesoMedioKg ?peso .
  OPTIONAL { ?animal ex:nomeComum ?nome }
  FILTER (?peso > 20.0)
} ORDER BY DESC(?peso)
```

**Resultado:**

| animal | nome | peso |
| --- | --- | --- |
| ex:cervo_do_pantanal | Cervo-do-pantanal | 120.0 |
| ex:capivara_03 | Capivara | 61.2 |
| ex:capivara_01 | Capivara | 55.0 |
| ex:capivara_02 | Capivara | 48.5 |
| ex:jacare_01 | Jacaré-do-papo-amarelo | 30.0 |
| ex:jacare_02 | Jacaré-do-papo-amarelo | 24.5 |


---

## Consulta 20 — Com filtros

**Descrição:** Atropelamentos com 2 ou mais animais envolvidos (FILTER).

```sparql
SELECT ?evento ?n WHERE {
  ?evento ex:numeroAnimaisEnvolvidos ?n .
  FILTER (?n >= 2)
} ORDER BY DESC(?n)
```

**Resultado:**

| evento | n |
| --- | --- |
| ex:atropelamento_014 | 3 |
| ex:atropelamento_017 | 3 |
| ex:atropelamento_018 | 3 |
| ex:atropelamento_034 | 3 |
| ex:atropelamento_025 | 3 |
| ex:atropelamento_033 | 3 |
| ex:atropelamento_038 | 3 |
| ex:atropelamento_042 | 3 |
| ex:atropelamento_031 | 2 |
| ex:atropelamento_029 | 2 |
| ex:atropelamento_011 | 2 |
| ex:atropelamento_012 | 2 |
| ex:atropelamento_015 | 2 |
| ex:atropelamento_016 | 2 |
| ex:atropelamento_019 | 2 |
| ex:atropelamento_024 | 2 |
| ex:atropelamento_030 | 2 |
| ex:atropelamento_035 | 2 |
| ex:atropelamento_036 | 2 |
| ex:atropelamento_044 | 2 |
| ex:atropelamento_047 | 2 |


---

## Consulta 21 — Agregação

**Descrição:** Número de atropelamentos por trecho (COUNT + GROUP BY).

```sparql
SELECT ?trecho (COUNT(?evento) AS ?total) WHERE {
  ?evento rdf:type ex:EventoAtropelamento ;
          ex:ocorreEmTrecho ?trecho .
} GROUP BY ?trecho ORDER BY DESC(?total)
```

**Resultado:**

| trecho | total |
| --- | --- |
| ex:trecho_BR471_km534 | 9 |
| ex:trecho_BR471_km502 | 9 |
| ex:trecho_BR471_km510 | 6 |
| ex:trecho_BR471_km530 | 5 |
| ex:trecho_BR471_km498 | 5 |
| ex:trecho_BR471_km538 | 4 |
| ex:trecho_BR471_km506 | 4 |
| ex:trecho_BR471_km542 | 3 |
| ex:trecho_BR471_km518 | 2 |
| ex:trecho_BR471_km526 | 1 |
| ex:trecho_BR471_km522 | 1 |
| ex:trecho_BR471_km514 | 1 |


---

## Consulta 22 — Agregação

**Descrição:** Número de atropelamentos por estação do ano.

```sparql
SELECT ?estacao (COUNT(?evento) AS ?total) WHERE {
  ?evento ex:ocorreNaEstacao ?estacao .
} GROUP BY ?estacao ORDER BY DESC(?total)
```

**Resultado:**

| estacao | total |
| --- | --- |
| ex:outono | 15 |
| ex:primavera | 13 |
| ex:inverno | 12 |
| ex:verao | 10 |


---

## Consulta 23 — Agregação

**Descrição:** Número de atropelamentos por espécie (classe do animal envolvido).

```sparql
SELECT ?especie (COUNT(DISTINCT ?evento) AS ?total) WHERE {
  ?evento rdf:type ex:EventoAtropelamento ;
          ex:envolveAnimal ?animal .
  ?animal rdf:type ?especie .
  ?especie rdfs:subClassOf* ex:Animal .
} GROUP BY ?especie ORDER BY DESC(?total)
```

**Resultado:**

| especie | total |
| --- | --- |
| ex:AveAquatica | 20 |
| ex:Capivara | 16 |
| ex:Cagado | 11 |
| ex:Jacare | 10 |
| ex:GraxaimDoMato | 8 |
| ex:Bugio | 7 |


---

## Consulta 24 — Agregação

**Descrição:** Média do nível de risco dos trechos (AVG).

```sparql
SELECT (AVG(?nivelRisco) AS ?mediaRisco) (COUNT(?trecho) AS ?nTrechos) WHERE {
  ?trecho ex:nivelRisco ?nivelRisco .
}
```

**Resultado:**

| mediaRisco | nTrechos |
| --- | --- |
| 0.4883333333333333333333333333 | 12 |


---

## Consulta 25 — Agregação

**Descrição:** Número de atropelamentos por período do dia.

```sparql
SELECT ?periodo (COUNT(?evento) AS ?total) WHERE {
  ?evento ex:ocorreNoPeriodo ?periodo .
} GROUP BY ?periodo ORDER BY DESC(?total)
```

**Resultado:**

| periodo | total |
| --- | --- |
| ex:noite | 26 |
| ex:madrugada | 19 |
| ex:manha | 3 |
| ex:tarde | 2 |


---

## Consulta 26 — Agregação

**Descrição:** Total de animais envolvidos (SUM de numeroAnimaisEnvolvidos) por clima.

```sparql
SELECT ?clima (SUM(?n) AS ?somaAnimais) (COUNT(?evento) AS ?nEventos) WHERE {
  ?evento ex:ocorreSobClima ?clima ;
          ex:numeroAnimaisEnvolvidos ?n .
} GROUP BY ?clima ORDER BY DESC(?somaAnimais)
```

**Resultado:**

| clima | somaAnimais | nEventos |
| --- | --- | --- |
| ex:garoa | 26 | 16 |
| ex:chuva_forte | 23 | 14 |
| ex:sol | 16 | 10 |
| ex:neblina | 14 | 10 |


---

## Consulta 27 — Cenário do domínio

**Descrição:** Trechos críticos: próximos a um banhado E com fator de proximidade da água (replica a definição da classe TrechoCritico via SPARQL).

```sparql
SELECT DISTINCT ?trecho WHERE {
  ?trecho rdf:type ex:TrechoRodovia ;
          ex:proximoA ?banhado ;
          ex:temFatorRisco ?fator .
  ?banhado rdf:type ex:Banhado .
  ?fator rdf:type ex:FatorProximidadeAgua .
} ORDER BY ?trecho
```

**Resultado:**

| trecho |
| --- |
| ex:trecho_BR471_km506 |
| ex:trecho_BR471_km518 |
| ex:trecho_BR471_km530 |
| ex:trecho_BR471_km538 |


---

## Consulta 28 — Cenário do domínio

**Descrição:** Trechos críticos (proximidade de banhado + fator água) e quantos atropelamentos NOTURNOS concentram.

```sparql
SELECT ?trecho (COUNT(?evento) AS ?atropelamentosNoturnos) WHERE {
  ?trecho rdf:type ex:TrechoRodovia ;
          ex:proximoA ?banhado ;
          ex:temFatorRisco ?fator .
  ?banhado rdf:type ex:Banhado .
  ?fator rdf:type ex:FatorProximidadeAgua .
  ?evento ex:ocorreEmTrecho ?trecho ;
          ex:ocorreNoPeriodo ?periodo .
  FILTER (?periodo IN (ex:madrugada, ex:noite))
} GROUP BY ?trecho ORDER BY DESC(?atropelamentosNoturnos)
```

**Resultado:**

| trecho | atropelamentosNoturnos |
| --- | --- |
| ex:trecho_BR471_km530 | 5 |
| ex:trecho_BR471_km538 | 4 |
| ex:trecho_BR471_km506 | 3 |
| ex:trecho_BR471_km518 | 2 |


---

## Consulta 29 — Cenário do domínio

**Descrição:** Espécie mais atropelada à noite/madrugada sob baixa visibilidade (neblina ou chuva forte).

```sparql
SELECT ?especie (COUNT(DISTINCT ?evento) AS ?total) WHERE {
  ?evento rdf:type ex:EventoAtropelamento ;
          ex:envolveAnimal ?animal ;
          ex:ocorreNoPeriodo ?periodo ;
          ex:ocorreSobClima ?clima .
  ?animal rdf:type ?especie .
  ?especie rdfs:subClassOf* ex:Animal .
  FILTER (?periodo IN (ex:madrugada, ex:noite))
  FILTER (?clima IN (ex:neblina, ex:chuva_forte))
} GROUP BY ?especie ORDER BY DESC(?total)
```

**Resultado:**

| especie | total |
| --- | --- |
| ex:Capivara | 9 |
| ex:AveAquatica | 8 |
| ex:Cagado | 5 |
| ex:Jacare | 3 |
| ex:Bugio | 3 |
| ex:GraxaimDoMato | 2 |


---

## Consulta 30 — Cenário do domínio

**Descrição:** Trechos de alto tráfego (volume diário >= 4000) e quantos atropelamentos têm.

```sparql
SELECT ?trecho ?volume (COUNT(?evento) AS ?atropelamentos) WHERE {
  ?trecho rdf:type ex:TrechoRodovia ;
          ex:volumeTrafegoDiario ?volume .
  FILTER (?volume >= 4000)
  OPTIONAL { ?evento ex:ocorreEmTrecho ?trecho }
} GROUP BY ?trecho ?volume ORDER BY DESC(?atropelamentos)
```

**Resultado:**

| trecho | volume | atropelamentos |
| --- | --- | --- |
| ex:trecho_BR471_km510 | 4200 | 6 |
| ex:trecho_BR471_km514 | 4200 | 1 |
| ex:trecho_BR471_km526 | 4200 | 1 |


---

## Consulta 31 — Múltiplas relações

**Descrição:** Pares de eventos encadeados pela relação temporal precedeEvento.

```sparql
SELECT ?anterior ?posterior WHERE {
  ?anterior ex:precedeEvento ?posterior .
} ORDER BY ?anterior
```

**Resultado:**

| anterior | posterior |
| --- | --- |
| ex:atropelamento_003 | ex:atropelamento_031 |
| ex:atropelamento_004 | ex:atropelamento_029 |
| ex:atropelamento_013 | ex:atropelamento_039 |
| ex:atropelamento_016 | ex:atropelamento_012 |
| ex:atropelamento_018 | ex:atropelamento_034 |
| ex:atropelamento_019 | ex:atropelamento_017 |
| ex:atropelamento_032 | ex:atropelamento_027 |
| ex:atropelamento_035 | ex:atropelamento_008 |
| ex:atropelamento_041 | ex:atropelamento_002 |
| ex:atropelamento_044 | ex:atropelamento_005 |


---

## Consulta 32 — Simples

**Descrição:** Indivíduos derivados de extração externa (possuem rdfs:comment de proveniência).

```sparql
SELECT ?individuo ?proveniencia WHERE {
  ?individuo rdfs:comment ?proveniencia .
} ORDER BY ?individuo
```

**Resultado:**

| individuo | proveniencia |
| --- | --- |
| ex:capivara_01 | Nome científico confirmado no Wikidata (Q131538). |
| ex:cervo_do_pantanal | Derivado do Wikidata (Q504501). |
| ex:jacana | Derivado do Wikidata (Q856201). |
| ex:ratao_do_banhado | Derivado de extração LLM (tripla 'habita Banhado') e confirmado no Wikidata (Q187704). |


---
