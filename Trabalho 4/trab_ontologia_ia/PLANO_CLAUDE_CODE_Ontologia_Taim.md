# Plano de Execução — Ontologia do Banhado do Taim (Trabalho de IA)

> **Para o Claude Code:** Este documento é a especificação completa do trabalho. Execute as fases **em ordem**. Cada fase tem objetivo, tarefas concretas, arquivos a produzir e **critérios de aceitação** ligados à rubrica do enunciado. Não pule os critérios de aceitação — eles são o que será avaliado. Use Python + Owlready2 para construir a ontologia e rdflib para as consultas SPARQL. Comente o código em português.

---

## 0. Visão geral e entregáveis

O objetivo é uma ontologia OWL do domínio de **atropelamentos de fauna no Banhado do Taim (RS)**, povoada com 100+ indivíduos, com 30+ consultas SPARQL e (opcionalmente) uma integração com ML para explicabilidade.

Entregáveis finais (pasta `entregas/`):

| Arquivo | Descrição |
|---|---|
| `taim.owl` | Ontologia em OWL (RDF/XML) |
| `scripts/01_construir_ontologia.py` | Define TBox: classes, hierarquia, propriedades, restrições |
| `scripts/02_povoar_ontologia.py` | Gera 100+ indivíduos (ABox) |
| `scripts/03_extrair_conhecimento.py` | Extração de fontes externas (Wikipedia/Wikidata/LLM) |
| `scripts/04_consultas_sparql.py` | Executa as 30+ consultas e salva resultados |
| `scripts/05_ml_explicabilidade.py` | (Opcional) modelo de ML simulado + explicação via ontologia |
| `consultas/resultados_sparql.md` | As 30 consultas com descrição, código e resultado |
| `relatorio/DECISOES_MODELAGEM.md` | Justificativas e alternativas descartadas |
| `relatorio/FONTES_E_EXTRACAO.md` | Fontes usadas, como extraídas, limitações |
| `relatorio/LIMITACOES_DADOS.md` | Inconsistências e limitações do povoamento |
| `protocolos_llm/` | Prompts e respostas, se LLM for usado |

> **Itens que o grupo faz manualmente (não automatizar):** prints do Protégé, montagem final do PDF, vídeo de 5 min. O Claude Code deve **deixar tudo pronto** para esses passos (ontologia abrível no Protégé, texto do relatório em Markdown, resultados das consultas).

---

## 1. Fase 0 — Ambiente

**Objetivo:** Preparar o ambiente Python reprodutível.

Tarefas:
1. Criar estrutura de pastas:
   ```
   taim_ontologia/
   ├── scripts/
   ├── consultas/
   ├── relatorio/
   ├── protocolos_llm/
   ├── dados_brutos/        # textos/JSON baixados das fontes
   └── entregas/
   ```
2. Criar `requirements.txt`:
   ```
   owlready2>=0.46
   rdflib>=7.0
   requests
   beautifulsoup4
   ```
   (spaCy e scikit-learn são opcionais — só se as fases 4/6 forem feitas.)
3. Instalar dependências e registrar versões num `AMBIENTE.md`.

**Critério de aceitação:** `python -c "import owlready2, rdflib"` roda sem erro.

---

## 2. Fase 1 — Construção da ontologia (TBox)

**Objetivo:** Definir classes, hierarquia, propriedades e restrições. Atende aos mínimos: **15 classes, 10 object properties, 10 data properties, relação temporal, relação espacial, restrições.**

Arquivo: `scripts/01_construir_ontologia.py`. IRI base: `http://www.taim.org/ontologia#`.

### 2.1 Classes e hierarquia (mínimo 15 — modelar ~25)

Hierarquia sugerida (ajustável):

```
Thing
├── Animal
│   ├── Mamifero
│   │   ├── Capivara
│   │   ├── Bugio
│   │   └── GraxaimDoMato
│   ├── Ave
│   │   └── AveAquatica
│   └── Reptil
│       ├── Jacare
│       └── Cagado
├── Habitat
│   ├── Banhado
│   ├── CorpoDagua
│   │   ├── Lagoa
│   │   └── Canal
│   ├── AreaAlagada
│   └── Vegetacao
├── InfraestruturaViaria
│   ├── Rodovia
│   └── TrechoRodovia
├── Evento
│   ├── EventoAtropelamento
│   └── EventoTravessia
├── CondicaoAmbiental
│   ├── CondicaoClimatica        # ex.: chuva_forte, sol, neblina
│   ├── PeriodoDia               # ex.: madrugada, manhã, tarde, noite
│   └── Estacao                  # ex.: verão, outono, inverno, primavera
└── FatorRisco
    ├── FatorTrafego
    ├── FatorVisibilidade
    └── FatorProximidadeAgua
```

> Use classes **disjuntas** onde fizer sentido (`AllDisjoint` entre Animal, Habitat, Evento, InfraestruturaViaria) — isso permite o reasoner detectar inconsistências.

### 2.2 Propriedades de objeto (mínimo 10 — modelar ~13)

Defina com **domain** e **range** sempre:

| Propriedade | Domain → Range | Observação |
|---|---|---|
| `envolveAnimal` | EventoAtropelamento → Animal | participação |
| `ocorreEmTrecho` | Evento → TrechoRodovia | |
| `ocorreSobClima` | Evento → CondicaoClimatica | |
| `ocorreNoPeriodo` | Evento → PeriodoDia | **temporal** |
| `ocorreNaEstacao` | Evento → Estacao | **temporal** |
| `proximoA` | TrechoRodovia → Habitat | **espacial** (simétrica? avaliar) |
| `pertenceARodovia` | TrechoRodovia → Rodovia | |
| `habita` / `viveEm` | Animal → Habitat | ecológica |
| `atravessa` | Animal → Rodovia | |
| `temFatorRisco` | TrechoRodovia → FatorRisco | |
| `contemVegetacao` | Habitat → Vegetacao | |
| `adjacenteA` | TrechoRodovia → TrechoRodovia | **espacial**, simétrica |
| `precedeEvento` | Evento → Evento | **temporal**, transitiva |

> Marque características OWL onde couber: `adjacenteA` e `proximoA` simétricas; `precedeEvento` transitiva; `pertenceARodovia` funcional.

### 2.3 Propriedades de dados (mínimo 10 — modelar ~14)

| Propriedade | Domain | Range (xsd) |
|---|---|---|
| `nomeComum` | Animal | string |
| `nomeCientifico` | Animal | string |
| `pesoMedioKg` | Animal | float |
| `kmInicio` | TrechoRodovia | float |
| `kmFim` | TrechoRodovia | float |
| `latitude` | TrechoRodovia/Habitat | float (**espacial**) |
| `longitude` | TrechoRodovia/Habitat | float (**espacial**) |
| `dataHora` | EventoAtropelamento | dateTime (**temporal**) |
| `numeroAnimaisEnvolvidos` | EventoAtropelamento | int |
| `descricaoClima` | CondicaoClimatica | string |
| `temperaturaC` | CondicaoClimatica | float |
| `visibilidadeMetros` | CondicaoClimatica | float |
| `volumeTrafegoDiario` | FatorTrafego / TrechoRodovia | int |
| `nivelRisco` | TrechoRodovia | float (0–1) |

> `dataHora` deve ser **funcional** (cada evento tem uma data/hora). Marque-a como FunctionalProperty.

### 2.4 Restrições (obrigatório)

Implementar pelo menos estas, via `Restriction` do Owlready2:

- **Cardinalidade mínima:** `EventoAtropelamento` `envolveAnimal min 1 Animal` (todo atropelamento envolve ≥1 animal).
- **Cardinalidade exata:** `EventoAtropelamento` `ocorreEmTrecho exactly 1 TrechoRodovia`.
- **Restrição existencial (some):** `Banhado` `contemVegetacao some Vegetacao`.
- **Restrição universal (only):** `EventoTravessia` `envolveAnimal only Animal`.
- **Classe definida (equivalência):** definir `TrechoCritico` ≡ `TrechoRodovia and (proximoA some Banhado) and (temFatorRisco some FatorProximidadeAgua)` — útil para o reasoner classificar trechos automaticamente e para a fase de explicabilidade.

**Critérios de aceitação da Fase 1:**
- [ ] ≥15 classes, ≥10 object properties, ≥10 data properties.
- [ ] Hierarquia de classes presente (subClassOf).
- [ ] ≥1 relação temporal e ≥1 espacial identificadas.
- [ ] Domain/range em todas as propriedades; ≥4 restrições + 1 classe definida.
- [ ] `taim.owl` salvo em RDF/XML e **abre sem erro no Protégé**.
- [ ] Rodar o reasoner (HermiT via Owlready2 `sync_reasoner`) sem inconsistência.

---

## 3. Fase 2 — Povoamento (ABox, 100+ indivíduos)

**Objetivo:** Criar ≥100 indivíduos coerentes. Arquivo: `scripts/02_povoar_ontologia.py`.

Plano de geração (use `random.seed(42)` para reprodutibilidade):

1. **Espécies (~15 indivíduos):** capivaras, aves aquáticas, jacarés, etc., com `nomeComum`, `nomeCientifico`, `pesoMedioKg`, e `habita` apontando para habitats.
2. **Habitats (~10):** o `banhado_do_taim`, lagoas (Lagoa Mangueira, Mirim), canais, áreas alagadas, vegetação — com lat/long aproximados da região real do Taim.
3. **Rodovia e trechos (~12):** `BR_471` + trechos `trecho_BR471_kmXX` cobrindo a faixa que corta o Taim (aprox. km 500–545 da BR-471/perímetro), cada um com `kmInicio`/`kmFim`, lat/long, `proximoA` algum habitat, `temFatorRisco`.
4. **Condições ambientais (~12):** instâncias de clima (chuva_forte, neblina, sol, garoa), períodos (madrugada/manhã/tarde/noite), estações (4).
5. **Fatores de risco (~8):** instâncias de tráfego, visibilidade, proximidade da água.
6. **Eventos de atropelamento (~50):** cada um com `dataHora` (datas/horas plausíveis distribuídas ao longo do ano), `envolveAnimal`, `ocorreEmTrecho`, `ocorreSobClima`, `ocorreNoPeriodo`, `ocorreNaEstacao`, `numeroAnimaisEnvolvidos`. Encadeie alguns com `precedeEvento` para exercitar a relação temporal.

> Soma: ~15 + 10 + 12 + 12 + 8 + 50 = **107 indivíduos** ✔. Ajustar quantidades para garantir folga acima de 100.

**Geração dos dados:** sintética porém **plausível**, ancorada em fatos reais (a BR-471 atravessa a ESEC Taim; capivara é a espécie emblemática de atropelamento; chuva/neblina reduzem visibilidade). Documentar isso no relatório.

**Critérios de aceitação da Fase 2:**
- [ ] `len(list(onto.individuals())) >= 100` (imprimir contagem no fim do script).
- [ ] Todo `EventoAtropelamento` tem `dataHora`, ≥1 animal e exatamente 1 trecho (consistente com as restrições).
- [ ] Reasoner roda sem inconsistência após o povoamento.
- [ ] Script imprime ≥3 exemplos concretos de indivíduos criados (para colar no relatório).

---

## 4. Fase 3 — Fontes de informação e extração de conhecimento

**Objetivo:** Atender à seção 4 do enunciado. Arquivo: `scripts/03_extrair_conhecimento.py`. Salvar brutos em `dados_brutos/`.

Implementar **pelo menos duas** das estratégias abaixo (mais = melhor nota):

1. **Wikipedia (scraping):** baixar a página "Banhado do Taim" e/ou "Estação Ecológica do Taim", extrair termos/espécies citadas (requests + BeautifulSoup). Salvar lista em `dados_brutos/wikipedia_termos.json`.
2. **Wikidata (SPARQL):** consultar o endpoint `https://query.wikidata.org/sparql` por táxons/espécies associadas à região ou fauna de áreas úmidas; mapear `nomeCientifico`. Salvar JSON.
3. **LLM (extração estruturada):** usar um prompt que recebe um trecho de texto ecológico e devolve triplas `(entidade1, relacao, entidade2)`. Documentar o **protocolo** (prompt exato + resposta) em `protocolos_llm/`. **Obrigatório revisar a saída** — anotar no relatório o que foi corrigido.
4. (Opcional) **NLP com spaCy** (`pt_core_news_sm`) para extração de dependências/relações.

**Integração:** converter os dados extraídos em indivíduos/propriedades da ontologia (ex.: criar `Ave` a partir de espécie da Wikidata, ligar `viveEm` lagoa). Mostrar exemplo no estilo `ave_001 rdf:type Ave; ave_001 viveEm lagoa_01`.

> **Atenção a rede:** o ambiente do Claude Code pode ter domínios bloqueados. Se Wikipedia/Wikidata não estiverem acessíveis, **documentar a tentativa** e usar um texto-fonte salvo localmente como fallback, deixando claro no relatório que a extração foi demonstrada sobre amostra local.

**Critérios de aceitação da Fase 3:**
- [ ] ≥2 estratégias de extração implementadas e executadas.
- [ ] `relatorio/FONTES_E_EXTRACAO.md` lista: quais fontes, como foram extraídas, limitações e erros do processo.
- [ ] Pelo menos alguns indivíduos da ontologia derivam de dados extraídos (rastreável).

---

## 5. Fase 4 — Consultas SPARQL (30+)

**Objetivo:** ≥30 consultas. Arquivo: `scripts/04_consultas_sparql.py` (carrega `taim.owl` com **rdflib** para SPARQL 1.1 completo, incluindo agregação). Saída em `consultas/resultados_sparql.md`.

Distribuição obrigatória (mínimos por categoria — somar ≥30):

| Categoria | Quantidade sugerida | Exemplos |
|---|---|---|
| Simples (por classe) | 6 | listar todos os `EventoAtropelamento`; todas as `Capivara`; todos os `TrechoRodovia` |
| Múltiplas relações | 8 | atropelamentos de capivara em trecho próximo a banhado (o exemplo do enunciado) |
| Com filtros | 6 | eventos sob `chuva_forte`; eventos no período `madrugada`; eventos com `temperaturaC < 15` (FILTER) |
| Agregação | 6 | nº de atropelamentos por trecho (COUNT + GROUP BY); por estação; por espécie; média de `nivelRisco` |
| Cenário do domínio | 4 | trechos críticos (proximidade de banhado + chuva + alto tráfego) que concentram atropelamentos noturnos |

Para **cada** consulta gerar no `.md`: (1) descrição em português, (2) bloco de código SPARQL, (3) resultado obtido (tabela). Reaproveite os dois exemplos SPARQL do enunciado como consultas 1 e 2.

> Defina o `PREFIX ex: <http://www.taim.org/ontologia#>` e `rdf:` em todas. Para agregação use `SELECT (COUNT(?e) AS ?total) ... GROUP BY ?trecho`.

**Critérios de aceitação da Fase 4:**
- [ ] ≥30 consultas, cobrindo as 5 categorias.
- [ ] Cada consulta executa e retorna resultado (ou resultado vazio justificado).
- [ ] `consultas/resultados_sparql.md` completo com descrição + código + resultado.

---

## 6. Fase 5 — Integração com ML (opcional, mas recomendada para nota)

**Objetivo:** Mostrar a ontologia como camada de **explicabilidade** de um modelo de risco. Arquivo: `scripts/05_ml_explicabilidade.py`.

Não precisa treinar um modelo de verdade — o enunciado permite **simular** predições:

1. Definir um "modelo" que recebe (km, horário, clima, proximidade de água, tipo de habitat) e devolve `risco ∈ [0,1]`. Pode ser uma `RandomForestRegressor` treinada em dados sintéticos OU uma função heurística — **explicar qual foi usada** no relatório.
2. Para um trecho com risco alto (ex.: 0.82 no km 32), **consultar a ontologia** (usar a classe definida `TrechoCritico` + SPARQL) e gerar a explicação em linguagem natural, no formato do enunciado:
   > "O risco é alto porque o trecho está próximo a um banhado, capivaras são comuns na região, frequentemente atravessam a rodovia, e a chuva forte reduz a visibilidade."
3. A explicação deve ser **derivada dos fatos da ontologia**, não hardcoded.

**Critérios de aceitação da Fase 5 (se feita):**
- [ ] Modelo descrito e predições de exemplo simuladas.
- [ ] Explicação gerada a partir de inferência/consulta na ontologia (rastreável aos fatos).

---

## 7. Fase 6 — Relatório (texto pronto para o PDF)

**Objetivo:** Produzir os textos em Markdown que o grupo monta no PDF final. O Claude Code escreve o conteúdo; o grupo adiciona prints do Protégé.

1. `relatorio/DECISOES_MODELAGEM.md` — **obrigatório**:
   - Justificar as principais decisões (por que reificar o atropelamento como classe `Evento`; por que condições ambientais são classes e não strings; escolha de relação temporal e espacial).
   - **Duas alternativas descartadas** com motivo. Sugestões:
     - *Alternativa A descartada:* modelar atropelamento como propriedade binária `atropela(Rodovia, Animal)` em vez de classe `Evento`. Descartada porque um evento tem múltiplos participantes e atributos (data/hora, clima, trecho) que uma relação binária não captura.
     - *Alternativa B descartada:* representar o clima como `data property` string direto no evento (`clima="chuva_forte"`) em vez de classe `CondicaoClimatica`. Descartada porque perde a capacidade de relacionar/consultar instâncias de clima e anexar atributos (temperatura, visibilidade).
2. `relatorio/FONTES_E_EXTRACAO.md` — gerado na Fase 3.
3. `relatorio/LIMITACOES_DADOS.md` — **obrigatório**: dados são sintéticos/plausíveis, possíveis vieses, coordenadas aproximadas, limitações da extração por LLM (alucinação) e do scraping.
4. `relatorio/CHECKLIST_RUBRICA.md` — tabela mapeando cada requisito do enunciado → onde foi atendido (arquivo/linha). Serve de autoavaliação.

**Critérios de aceitação da Fase 6:**
- [ ] Decisões justificadas + 2 alternativas descartadas explicadas.
- [ ] Limitações dos dados discutidas.
- [ ] Checklist da rubrica preenchido.

---

## 8. Fase 7 — Empacotamento final

1. Copiar para `entregas/`: `taim.owl`, todos os scripts, `resultados_sparql.md`, os `.md` do relatório, protocolos de LLM.
2. Verificação final automatizada (script `99_verificar.py` que imprime): contagem de classes, object props, data props, indivíduos, nº de consultas. Confirmar que batem com os mínimos.
3. Gerar um `README.md` na raiz explicando como rodar tudo na ordem (Fase 0 → 7).

**Itens manuais finais para o grupo (o Claude Code NÃO faz):**
- [ ] Abrir `taim.owl` no Protégé e capturar prints (aba *Classes*, *Object Properties*, *OntoGraf*, execução de uma consulta DL/SPARQL).
- [ ] Montar o PDF final juntando os `.md` + prints.
- [ ] Gravar o vídeo de até 5 minutos.

---

## 9. Resumo do mapa de requisitos → fase

| Requisito do enunciado | Atendido em |
|---|---|
| 15+ classes, hierarquia | Fase 1 |
| 10+ object properties | Fase 1 |
| 10+ data properties | Fase 1 |
| Restrições (domínio, alcance, cardinalidade) | Fase 1 |
| Relação temporal + espacial | Fases 1 e 2 |
| Justificar decisões + 2 alternativas descartadas | Fase 6 |
| 100+ indivíduos | Fase 2 |
| Script de povoamento + explicar geração + exemplos + limitações | Fases 2 e 6 |
| Fontes (Wikidata/Wikipedia/LLM/OSM) + extração + limitações | Fase 3 |
| 30+ consultas SPARQL nas 5 categorias | Fase 4 |
| Integração ML (opcional) | Fase 5 |
| Arquivo .owl, scripts, protocolos LLM, relatório | Fases 1–7 |

---

## 10. Notas técnicas para o Claude Code

- **Owlready2:** definir classes como subclasses de `Thing`; propriedades como subclasses de `ObjectProperty`/`DataProperty` com `domain`/`range` em listas. Restrições com `clase.is_a.append(propriedade.some(Classe))` ou `.min(1, Classe)`. Salvar com `onto.save(file="taim.owl", format="rdfxml")`.
- **Reasoner:** `with onto: sync_reasoner()` (precisa de Java instalado para HermiT — se Java faltar, registrar no relatório e validar consistência manualmente).
- **SPARQL:** preferir `rdflib` (`g = rdflib.Graph(); g.parse("taim.owl")`) para ter `COUNT`/`GROUP BY`/`FILTER` completos; Owlready2 tem SPARQL embutido mas é mais limitado para agregação.
- **Reprodutibilidade:** `random.seed(42)` no povoamento; fixar versões no `requirements.txt`.
- **Coordenadas reais aproximadas do Taim:** latitude ≈ -32.5, longitude ≈ -52.5 (região da ESEC Taim / Lagoa Mangueira). Usar como base para variar os pontos.
- **Se um domínio de rede estiver bloqueado:** não falhar silenciosamente — capturar a exceção, registrar em `dados_brutos/log_extracao.txt` e seguir com fallback local.
