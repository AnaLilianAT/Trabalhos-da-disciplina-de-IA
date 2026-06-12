"""
Fase 4 — Consultas SPARQL sobre a ontologia do Taim.
"""

import sys
from collections import Counter
from pathlib import Path

import rdflib
from rdflib import URIRef
from rdflib.namespace import RDF, RDFS, OWL

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

BASE_DIR = Path(__file__).resolve().parent.parent
OWL_PATH = BASE_DIR / "taim.owl"
OUT_MD = BASE_DIR / "consultas" / "resultados_sparql.md"

EX = "http://www.taim.org/ontologia#"

# Prefixos prefixados a todas as consultas.
PREFIXOS = f"""PREFIX ex:   <{EX}>
PREFIX rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX xsd:  <http://www.w3.org/2001/XMLSchema#>
"""


def short(term):
    """Encurta URIs para leitura (remove namespaces conhecidos)."""
    if isinstance(term, URIRef):
        s = str(term)
        if s.startswith(EX):
            return "ex:" + s[len(EX):]
        for pref, ns in (("rdf:", str(RDF)), ("rdfs:", str(RDFS)), ("owl:", str(OWL))):
            if s.startswith(ns):
                return pref + s[len(ns):]
        return s.rsplit("/", 1)[-1].rsplit("#", 1)[-1]
    if term is None:
        return ""
    return str(term)


# ===========================================================================
# DEFINIÇÃO DAS CONSULTAS
# Cada item: (categoria, descrição, sparql_sem_prefixos)
# ===========================================================================
CONSULTAS = []


def q(categoria, descricao, sparql):
    CONSULTAS.append({"categoria": categoria, "descricao": descricao,
                      "sparql": sparql.strip()})


# ---- (A) SIMPLES — por classe (6) -----------------------------------------
q("Simples", "Listar todos os eventos de atropelamento.",
  """
SELECT ?evento WHERE {
  ?evento rdf:type ex:EventoAtropelamento .
} ORDER BY ?evento
""")

q("Simples", "Listar todas as capivaras (indivíduos da classe Capivara).",
  """
SELECT ?capivara ?nome WHERE {
  ?capivara rdf:type ex:Capivara .
  OPTIONAL { ?capivara ex:nomeComum ?nome }
} ORDER BY ?capivara
""")

q("Simples", "Listar todos os trechos de rodovia com seus km de início e fim.",
  """
SELECT ?trecho ?kmInicio ?kmFim WHERE {
  ?trecho rdf:type ex:TrechoRodovia .
  ?trecho ex:kmInicio ?kmInicio .
  ?trecho ex:kmFim ?kmFim .
} ORDER BY ?kmInicio
""")

q("Simples", "Listar todos os animais (qualquer subclasse de Animal) e seu nome comum.",
  """
SELECT ?animal ?nomeComum WHERE {
  ?animal rdf:type/rdfs:subClassOf* ex:Animal .
  ?animal ex:nomeComum ?nomeComum .
} ORDER BY ?nomeComum
""")

q("Simples", "Listar todas as condições climáticas com sua descrição.",
  """
SELECT ?clima ?descricao WHERE {
  ?clima rdf:type ex:CondicaoClimatica .
  ?clima ex:descricaoClima ?descricao .
} ORDER BY ?clima
""")

q("Simples", "Listar todos os habitats (qualquer subclasse de Habitat).",
  """
SELECT DISTINCT ?habitat WHERE {
  ?habitat rdf:type/rdfs:subClassOf* ex:Habitat .
} ORDER BY ?habitat
""")

# ---- (B) MÚLTIPLAS RELAÇÕES (8) -------------------------------------------
q("Múltiplas relações",
  "Atropelamentos de capivara em trecho próximo a um banhado (exemplo do enunciado).",
  """
SELECT ?evento ?capivara ?trecho ?banhado WHERE {
  ?evento rdf:type ex:EventoAtropelamento ;
          ex:envolveAnimal ?capivara ;
          ex:ocorreEmTrecho ?trecho .
  ?capivara rdf:type ex:Capivara .
  ?trecho ex:proximoA ?banhado .
  ?banhado rdf:type ex:Banhado .
} ORDER BY ?evento
""")

q("Múltiplas relações", "Cada animal e os habitats que ele habita.",
  """
SELECT ?animal ?nome ?habitat WHERE {
  ?animal rdf:type/rdfs:subClassOf* ex:Animal ;
          ex:habita ?habitat .
  OPTIONAL { ?animal ex:nomeComum ?nome }
} ORDER BY ?animal
""")

q("Múltiplas relações",
  "Para cada evento, o trecho onde ocorreu e a rodovia a que o trecho pertence.",
  """
SELECT ?evento ?trecho ?rodovia WHERE {
  ?evento ex:ocorreEmTrecho ?trecho .
  ?trecho ex:pertenceARodovia ?rodovia .
} ORDER BY ?evento
""")

q("Múltiplas relações", "Eventos com o animal envolvido e o clima sob o qual ocorreram.",
  """
SELECT ?evento ?animal ?clima WHERE {
  ?evento ex:envolveAnimal ?animal ;
          ex:ocorreSobClima ?clima .
} ORDER BY ?evento
""")

q("Múltiplas relações", "Trechos e os habitats aos quais estão próximos.",
  """
SELECT ?trecho ?habitat WHERE {
  ?trecho rdf:type ex:TrechoRodovia ;
          ex:proximoA ?habitat .
} ORDER BY ?trecho
""")

q("Múltiplas relações",
  "Atropelamentos noturnos (madrugada ou noite) com a espécie do animal envolvido.",
  """
SELECT ?evento ?periodo ?especie WHERE {
  ?evento ex:ocorreNoPeriodo ?periodo ;
          ex:envolveAnimal ?animal .
  ?animal rdf:type ?especie .
  ?especie rdfs:subClassOf* ex:Animal .
  FILTER (?periodo IN (ex:madrugada, ex:noite))
} ORDER BY ?evento
""")

q("Múltiplas relações", "Animais que atravessam a BR-471 e onde habitam.",
  """
SELECT ?animal ?nome ?habitat WHERE {
  ?animal ex:atravessa ex:BR_471 .
  OPTIONAL { ?animal ex:nomeComum ?nome }
  OPTIONAL { ?animal ex:habita ?habitat }
} ORDER BY ?animal
""")

q("Múltiplas relações",
  "Eventos e os fatores de risco do trecho em que ocorreram.",
  """
SELECT ?evento ?trecho ?fator WHERE {
  ?evento ex:ocorreEmTrecho ?trecho .
  ?trecho ex:temFatorRisco ?fator .
} ORDER BY ?trecho ?evento
""")

# ---- (C) COM FILTROS (6) --------------------------------------------------
q("Com filtros", "Eventos ocorridos sob chuva forte.",
  """
SELECT ?evento WHERE {
  ?evento ex:ocorreSobClima ex:chuva_forte .
} ORDER BY ?evento
""")

q("Com filtros", "Eventos ocorridos no período da madrugada.",
  """
SELECT ?evento WHERE {
  ?evento ex:ocorreNoPeriodo ex:madrugada .
} ORDER BY ?evento
""")

q("Com filtros",
  "Eventos sob clima com temperatura abaixo de 15 °C (FILTER em data property).",
  """
SELECT ?evento ?clima ?temp WHERE {
  ?evento ex:ocorreSobClima ?clima .
  ?clima ex:temperaturaC ?temp .
  FILTER (?temp < 15.0)
} ORDER BY ?temp
""")

q("Com filtros", "Trechos com nível de risco acima de 0,7 (FILTER).",
  """
SELECT ?trecho ?nivelRisco WHERE {
  ?trecho ex:nivelRisco ?nivelRisco .
  FILTER (?nivelRisco > 0.7)
} ORDER BY DESC(?nivelRisco)
""")

q("Com filtros", "Animais de grande porte: peso médio acima de 20 kg (FILTER).",
  """
SELECT ?animal ?nome ?peso WHERE {
  ?animal ex:pesoMedioKg ?peso .
  OPTIONAL { ?animal ex:nomeComum ?nome }
  FILTER (?peso > 20.0)
} ORDER BY DESC(?peso)
""")

q("Com filtros", "Atropelamentos com 2 ou mais animais envolvidos (FILTER).",
  """
SELECT ?evento ?n WHERE {
  ?evento ex:numeroAnimaisEnvolvidos ?n .
  FILTER (?n >= 2)
} ORDER BY DESC(?n)
""")

# ---- (D) AGREGAÇÃO (6) ----------------------------------------------------
q("Agregação", "Número de atropelamentos por trecho (COUNT + GROUP BY).",
  """
SELECT ?trecho (COUNT(?evento) AS ?total) WHERE {
  ?evento rdf:type ex:EventoAtropelamento ;
          ex:ocorreEmTrecho ?trecho .
} GROUP BY ?trecho ORDER BY DESC(?total)
""")

q("Agregação", "Número de atropelamentos por estação do ano.",
  """
SELECT ?estacao (COUNT(?evento) AS ?total) WHERE {
  ?evento ex:ocorreNaEstacao ?estacao .
} GROUP BY ?estacao ORDER BY DESC(?total)
""")

q("Agregação", "Número de atropelamentos por espécie (classe do animal envolvido).",
  """
SELECT ?especie (COUNT(DISTINCT ?evento) AS ?total) WHERE {
  ?evento rdf:type ex:EventoAtropelamento ;
          ex:envolveAnimal ?animal .
  ?animal rdf:type ?especie .
  ?especie rdfs:subClassOf* ex:Animal .
} GROUP BY ?especie ORDER BY DESC(?total)
""")

q("Agregação", "Média do nível de risco dos trechos (AVG).",
  """
SELECT (AVG(?nivelRisco) AS ?mediaRisco) (COUNT(?trecho) AS ?nTrechos) WHERE {
  ?trecho ex:nivelRisco ?nivelRisco .
}
""")

q("Agregação", "Número de atropelamentos por período do dia.",
  """
SELECT ?periodo (COUNT(?evento) AS ?total) WHERE {
  ?evento ex:ocorreNoPeriodo ?periodo .
} GROUP BY ?periodo ORDER BY DESC(?total)
""")

q("Agregação",
  "Total de animais envolvidos (SUM de numeroAnimaisEnvolvidos) por clima.",
  """
SELECT ?clima (SUM(?n) AS ?somaAnimais) (COUNT(?evento) AS ?nEventos) WHERE {
  ?evento ex:ocorreSobClima ?clima ;
          ex:numeroAnimaisEnvolvidos ?n .
} GROUP BY ?clima ORDER BY DESC(?somaAnimais)
""")

# ---- (E) CENÁRIO DO DOMÍNIO (4) -------------------------------------------
q("Cenário do domínio",
  "Trechos críticos: próximos a um banhado E com fator de proximidade da água "
  "(replica a definição da classe TrechoCritico via SPARQL).",
  """
SELECT DISTINCT ?trecho WHERE {
  ?trecho rdf:type ex:TrechoRodovia ;
          ex:proximoA ?banhado ;
          ex:temFatorRisco ?fator .
  ?banhado rdf:type ex:Banhado .
  ?fator rdf:type ex:FatorProximidadeAgua .
} ORDER BY ?trecho
""")

q("Cenário do domínio",
  "Trechos críticos (proximidade de banhado + fator água) e quantos atropelamentos "
  "NOTURNOS concentram.",
  """
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
""")

q("Cenário do domínio",
  "Espécie mais atropelada à noite/madrugada sob baixa visibilidade "
  "(neblina ou chuva forte).",
  """
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
""")

q("Cenário do domínio",
  "Trechos de alto tráfego (volume diário >= 4000) e quantos atropelamentos têm.",
  """
SELECT ?trecho ?volume (COUNT(?evento) AS ?atropelamentos) WHERE {
  ?trecho rdf:type ex:TrechoRodovia ;
          ex:volumeTrafegoDiario ?volume .
  FILTER (?volume >= 4000)
  OPTIONAL { ?evento ex:ocorreEmTrecho ?trecho }
} GROUP BY ?trecho ?volume ORDER BY DESC(?atropelamentos)
""")

# ---- Extras ------------------------------------------
q("Múltiplas relações",
  "Pares de eventos encadeados pela relação temporal precedeEvento.",
  """
SELECT ?anterior ?posterior WHERE {
  ?anterior ex:precedeEvento ?posterior .
} ORDER BY ?anterior
""")

q("Simples",
  "Indivíduos derivados de extração externa (possuem rdfs:comment de proveniência).",
  """
SELECT ?individuo ?proveniencia WHERE {
  ?individuo rdfs:comment ?proveniencia .
} ORDER BY ?individuo
""")


# ===========================================================================
# EXECUÇÃO E GERAÇÃO DO MARKDOWN
# ===========================================================================
def tabela_md(res):
    """Formata um resultado rdflib como tabela Markdown (limita a 25 linhas)."""
    cols = [str(v) for v in res.vars]
    linhas = list(res)
    cabecalho = "| " + " | ".join(cols) + " |"
    separador = "| " + " | ".join("---" for _ in cols) + " |"
    corpo = []
    LIMITE = 25
    for row in linhas[:LIMITE]:
        celulas = [short(row[v]) for v in res.vars]
        corpo.append("| " + " | ".join(celulas) + " |")
    texto = "\n".join([cabecalho, separador] + corpo)
    if len(linhas) > LIMITE:
        texto += f"\n\n_({len(linhas)} linhas no total; exibindo as primeiras {LIMITE}.)_"
    elif not linhas:
        texto = "_(consulta executou e retornou 0 linhas)_"
    return texto, len(linhas)


def main():
    print("=" * 60)
    print("FASE 4 — Consultas SPARQL")
    print("=" * 60)

    g = rdflib.Graph()
    g.parse(str(OWL_PATH), format="xml")
    print(f"Grafo carregado: {len(g)} triplas de {OWL_PATH.name}")

    # Contagem por categoria.
    cont = Counter(c["categoria"] for c in CONSULTAS)
    print(f"Total de consultas: {len(CONSULTAS)}")
    for cat, n in cont.items():
        print(f"  - {cat}: {n}")
    assert len(CONSULTAS) >= 30, "FALHA: menos de 30 consultas."

    # Monta o Markdown.
    linhas_md = []
    linhas_md.append("# Resultados das Consultas SPARQL — Ontologia do Taim\n")
    linhas_md.append(
        f"Geradas por [`scripts/04_consultas_sparql.py`](../scripts/04_consultas_sparql.py) "
        f"sobre `taim.owl` ({len(g)} triplas), via **rdflib** (SPARQL 1.1).\n")
    linhas_md.append(f"**Total: {len(CONSULTAS)} consultas** — "
                     + ", ".join(f"{cat}: {n}" for cat, n in cont.items()) + ".\n")
    linhas_md.append("Todas usam os prefixos:\n")
    linhas_md.append("```sparql\n" + PREFIXOS + "```\n")
    linhas_md.append("---\n")

    falhas = 0
    for i, c in enumerate(CONSULTAS, start=1):
        print(f"Executando consulta {i:02d} [{c['categoria']}]...")
        bloco = [f"## Consulta {i} — {c['categoria']}\n",
                 f"**Descrição:** {c['descricao']}\n",
                 "```sparql\n" + c["sparql"] + "\n```\n",
                 "**Resultado:**\n"]
        try:
            res = g.query(PREFIXOS + c["sparql"])
            tab, n = tabela_md(res)
            bloco.append(tab + "\n")
        except Exception as exc:
            falhas += 1
            bloco.append(f"> ERRO ao executar: {exc}\n")
        bloco.append("\n---\n")
        linhas_md.append("\n".join(bloco))

    OUT_MD.write_text("\n".join(linhas_md), encoding="utf-8")
    print(f"\nArquivo gerado: {OUT_MD}")
    print(f"Consultas com erro: {falhas}")
    assert falhas == 0, "FALHA: alguma consulta não executou."
    print("OK: todas as 32 consultas executaram.")


if __name__ == "__main__":
    main()
