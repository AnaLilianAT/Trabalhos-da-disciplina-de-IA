# -*- coding: utf-8 -*-
"""
Fase 3 — Fontes de informação e extração de conhecimento.

Implementa TRÊS estratégias de extração e integra os dados à ontologia, de forma
rastreável (cada indivíduo derivado recebe rdfs:comment indicando a fonte):

  1. Wikipedia (scraping com requests + BeautifulSoup) -> dados_brutos/wikipedia_termos.json
  2. Wikidata (SPARQL no endpoint oficial)             -> dados_brutos/wikidata_taxa.json
  3. LLM (extração de triplas, protocolo revisado)     -> dados_brutos/llm_triplas_revisadas.json

Robustez de rede: se a Wikipedia estiver bloqueada, cai para um texto-fonte local
(`amostra_local_taim.txt`) e registra a tentativa em `dados_brutos/log_extracao.txt`.

Integração: cria novos indivíduos (ratão-do-banhado, cervo-do-pantanal, jaçanã) e
enriquece existentes com QIDs do Wikidata. Salva o `taim.owl` atualizado.

Execução (após as Fases 1 e 2):
    .\\venv\\Scripts\\python.exe scripts\\03_extrair_conhecimento.py
"""

import datetime
import json
import re
import sys
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from owlready2 import get_ontology, sync_reasoner

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

BASE_DIR = Path(__file__).resolve().parent.parent
OWL_PATH = BASE_DIR / "taim.owl"
DADOS = BASE_DIR / "dados_brutos"
LOG = DADOS / "log_extracao.txt"

USER_AGENT = ("TaimOntologiaBot/1.0 (trabalho academico de IA; "
              "contato: toviol1.2004@gmail.com)")


def log(msg):
    """Registra mensagens de extração (sucessos, erros, fallbacks)."""
    ts = datetime.datetime.now().isoformat(timespec="seconds")
    linha = f"[{ts}] {msg}"
    print(linha)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(linha + "\n")


# ===========================================================================
# ESTRATÉGIA 1 — WIKIPEDIA (scraping)
# ===========================================================================
WIKI_URL = "https://pt.wikipedia.org/wiki/Esta%C3%A7%C3%A3o_Ecol%C3%B3gica_do_Taim"

# Palavras-chave de fauna para filtrar termos relevantes do texto.
FAUNA_KEYWORDS = [
    "capivara", "jacaré", "jacare", "ratão", "ratao", "graxaim", "bugio",
    "cervo", "veado", "garça", "garca", "quero-quero", "tachã", "tacha",
    "cisne", "marreco", "lontra", "tuco-tuco", "ariranha", "colhereiro",
    "biguá", "bigua", "frango-d'água", "cágado", "cagado", "tartaruga",
]

BINOMIAL_RE = re.compile(r"^[A-Z][a-zà-ÿ]+ [a-zà-ÿ]+$")


def _extrair_termos_de_texto(texto):
    """Fallback: extrai termos de fauna de um texto simples por palavra-chave."""
    achados = set()
    low = texto.lower()
    for kw in FAUNA_KEYWORDS:
        if kw in low:
            achados.add(kw)
    return sorted(achados)


def extrair_wikipedia():
    """Baixa a página da ESEC Taim e extrai nomes científicos e termos de fauna."""
    resultado = {
        "fonte": "Wikipedia (pt) — Estação Ecológica do Taim",
        "url": WIKI_URL,
        "data": datetime.date.today().isoformat(),
        "nomes_cientificos": [],
        "termos_fauna": [],
        "metodo": "scraping requests+BeautifulSoup",
    }
    try:
        r = requests.get(WIKI_URL, headers={"User-Agent": USER_AGENT}, timeout=20)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        conteudo = soup.find(id="mw-content-text")

        # (a) Nomes científicos: textos em itálico no padrão binomial "Genus species".
        cientificos = set()
        for it in conteudo.find_all("i"):
            t = it.get_text(strip=True)
            if BINOMIAL_RE.match(t):
                cientificos.add(t)

        # (b) Termos de fauna: textos de links que contêm palavras-chave.
        termos = set()
        for a in conteudo.find_all("a"):
            t = a.get_text(strip=True)
            low = t.lower()
            if 2 < len(t) < 40 and any(kw in low for kw in FAUNA_KEYWORDS):
                termos.add(t)

        resultado["nomes_cientificos"] = sorted(cientificos)
        resultado["termos_fauna"] = sorted(termos)
        log(f"Wikipedia OK: {len(cientificos)} nomes científicos, "
            f"{len(termos)} termos de fauna extraídos.")
    except Exception as exc:  # noqa: BLE001
        # Fallback para amostra local.
        log(f"Wikipedia FALHOU ({exc}). Usando fallback local amostra_local_taim.txt.")
        texto = (DADOS / "amostra_local_taim.txt").read_text(encoding="utf-8")
        resultado["fonte"] += " [FALLBACK: amostra local]"
        resultado["metodo"] = "fallback: extração por palavra-chave em texto local"
        resultado["termos_fauna"] = _extrair_termos_de_texto(texto)

    (DADOS / "wikipedia_termos.json").write_text(
        json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8")
    return resultado


# ===========================================================================
# ESTRATÉGIA 2 — WIKIDATA (SPARQL)
# ===========================================================================
WIKIDATA_ENDPOINT = "https://query.wikidata.org/sparql"

# Táxons reais da fauna do Taim: alguns já na ontologia (enriquecimento) e
# outros NOVOS (viram indivíduos derivados).
TAXONS_ALVO = [
    "Hydrochoerus hydrochaeris",  # capivara (existente)
    "Caiman latirostris",         # jacaré-do-papo-amarelo (existente)
    "Myocastor coypus",           # ratão-do-banhado (NOVO)
    "Blastocerus dichotomus",     # cervo-do-pantanal (NOVO)
    "Jacana jacana",              # jaçanã (NOVO)
]


def extrair_wikidata():
    """Consulta o Wikidata por QID e nome vernáculo (pt) dos táxons-alvo."""
    valores = " ".join(f'"{t}"' for t in TAXONS_ALVO)
    query = f"""
    SELECT ?taxonName ?taxon ?vernacular WHERE {{
      VALUES ?taxonName {{ {valores} }}
      ?taxon wdt:P225 ?taxonName .
      OPTIONAL {{ ?taxon wdt:P1843 ?vernacular . FILTER(LANG(?vernacular) = "pt") }}
    }}
    """
    resultado = {
        "fonte": "Wikidata Query Service (SPARQL)",
        "endpoint": WIKIDATA_ENDPOINT,
        "data": datetime.date.today().isoformat(),
        "consulta_sparql": query.strip(),
        "taxa": {},
    }
    try:
        r = requests.get(
            WIKIDATA_ENDPOINT,
            params={"query": query, "format": "json"},
            headers={"User-Agent": USER_AGENT, "Accept": "application/sparql-results+json"},
            timeout=30,
        )
        r.raise_for_status()
        bindings = r.json()["results"]["bindings"]
        for b in bindings:
            nome = b["taxonName"]["value"]
            qid = b["taxon"]["value"].rsplit("/", 1)[-1]
            vern = b.get("vernacular", {}).get("value")
            entry = resultado["taxa"].setdefault(nome, {"qid": qid, "vernaculos_pt": []})
            if vern and vern not in entry["vernaculos_pt"]:
                entry["vernaculos_pt"].append(vern)
        log(f"Wikidata OK: {len(resultado['taxa'])} táxons resolvidos "
            f"({', '.join(resultado['taxa'].keys())}).")
    except Exception as exc:  # noqa: BLE001
        log(f"Wikidata FALHOU ({exc}). Seguindo sem enriquecimento do Wikidata.")

    (DADOS / "wikidata_taxa.json").write_text(
        json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8")
    return resultado


# ===========================================================================
# ESTRATÉGIA 3 — LLM (carrega triplas já revisadas pelo grupo)
# ===========================================================================
def extrair_llm():
    """Carrega as triplas do LLM já revisadas (protocolo em protocolos_llm/)."""
    caminho = DADOS / "llm_triplas_revisadas.json"
    dados = json.loads(caminho.read_text(encoding="utf-8"))
    n = len(dados.get("triplas_mapeaveis", []))
    log(f"LLM OK: {n} triplas revisadas carregadas (protocolo documentado).")
    return dados


# ===========================================================================
# INTEGRAÇÃO — cria indivíduos derivados (rastreáveis via rdfs:comment)
# ===========================================================================
def integrar(onto, wikidata, llm):
    derivados = []
    taxa = wikidata.get("taxa", {})

    def proveniencia(ind, texto):
        if texto not in (ind.comment or []):
            ind.comment.append(texto)

    def qid_de(nome_cientifico):
        return taxa.get(nome_cientifico, {}).get("qid")

    def vern_de(nome_cientifico, padrao):
        vs = taxa.get(nome_cientifico, {}).get("vernaculos_pt", [])
        return vs[0] if vs else padrao

    with onto:
        banhado = onto.banhado_do_taim
        br471 = onto.BR_471

        # --- (A) Ratão-do-banhado: vem do LLM, QID confirmado pelo Wikidata ---
        ratao = onto.Mamifero("ratao_do_banhado")
        ratao.nomeComum = [vern_de("Myocastor coypus", "Ratão-do-banhado")]
        ratao.nomeCientifico = ["Myocastor coypus"]
        ratao.pesoMedioKg = [7.0]
        ratao.habita = [banhado]               # tripla LLM: habita Banhado
        ratao.atravessa = [br471]
        q = qid_de("Myocastor coypus")
        proveniencia(ratao, "Derivado de extração LLM (tripla 'habita Banhado') "
                            + (f"e confirmado no Wikidata ({q})." if q else "."))
        derivados.append(ratao)

        # --- (B) Cervo-do-pantanal: derivado do Wikidata ---
        cervo = onto.Mamifero("cervo_do_pantanal")
        cervo.nomeComum = [vern_de("Blastocerus dichotomus", "Cervo-do-pantanal")]
        cervo.nomeCientifico = ["Blastocerus dichotomus"]
        cervo.pesoMedioKg = [120.0]
        cervo.viveEm = [onto.area_alagada_sul]
        q = qid_de("Blastocerus dichotomus")
        proveniencia(cervo, f"Derivado do Wikidata ({q})." if q
                     else "Derivado do Wikidata (QID não resolvido).")
        derivados.append(cervo)

        # --- (C) Jaçanã: ave aquática derivada do Wikidata ---
        jacana = onto.AveAquatica("jacana")
        jacana.nomeComum = [vern_de("Jacana jacana", "Jaçanã")]
        jacana.nomeCientifico = ["Jacana jacana"]
        jacana.pesoMedioKg = [0.13]
        jacana.viveEm = [onto.lagoa_mangueira]
        q = qid_de("Jacana jacana")
        proveniencia(jacana, f"Derivado do Wikidata ({q})." if q
                     else "Derivado do Wikidata (QID não resolvido).")
        derivados.append(jacana)

        # --- (D) Enriquecimento de indivíduo existente com QID do Wikidata ---
        cap = onto.capivara_01
        qcap = qid_de("Hydrochoerus hydrochaeris")
        if qcap:
            proveniencia(cap, f"Nome científico confirmado no Wikidata ({qcap}).")

    return derivados


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    print("=" * 60)
    print("FASE 3 — Extração de conhecimento de fontes externas")
    print("=" * 60)

    # Carrega a ontologia povoada (Fase 2).
    with open(OWL_PATH, "rb") as f:
        onto = get_ontology(OWL_PATH.as_uri()).load(fileobj=f)
    n_antes = len(list(onto.individuals()))

    print("\n[1/3] Wikipedia (scraping)...")
    wiki = extrair_wikipedia()
    print(f"      nomes científicos: {wiki['nomes_cientificos'][:8]}"
          f"{' ...' if len(wiki['nomes_cientificos']) > 8 else ''}")
    print(f"      termos de fauna:   {wiki['termos_fauna'][:10]}"
          f"{' ...' if len(wiki['termos_fauna']) > 10 else ''}")

    print("\n[2/3] Wikidata (SPARQL)...")
    wd = extrair_wikidata()
    for nome, info in wd.get("taxa", {}).items():
        print(f"      {nome:32s} -> {info['qid']:10s} "
              f"vernáculo: {info['vernaculos_pt'][:2]}")

    print("\n[3/3] LLM (triplas revisadas)...")
    llm = extrair_llm()
    for t in llm.get("triplas_mapeaveis", []):
        print(f"      ({t['sujeito']} | {t['relacao']} | {t['objeto']})")

    # ---- Integração ------------------------------------------------------
    print("\nIntegrando dados extraídos à ontologia...")
    derivados = integrar(onto, wd, llm)
    n_depois = len(list(onto.individuals()))
    print(f"Indivíduos derivados de extração: {len(derivados)} -> "
          f"{[d.name for d in derivados]}")
    print(f"Total de indivíduos: {n_antes} -> {n_depois}")

    # ---- Salvar ----------------------------------------------------------
    onto.save(file=str(OWL_PATH), format="rdfxml")
    print(f"\nOntologia atualizada salva em: {OWL_PATH}")

    # ---- Reasoner (validação) -------------------------------------------
    print("\nValidando consistência com HermiT...")
    try:
        with onto:
            sync_reasoner()
        print("OK: reasoner sem inconsistência após integração.")
    except Exception as exc:  # noqa: BLE001
        print(f"AVISO: reasoner não rodou ({exc}). Ver AMBIENTE.md.")

    # ---- Exemplo rastreável (rdfs:comment com a fonte) -------------------
    print("\n" + "-" * 60)
    print("EXEMPLO RASTREÁVEL (indivíduo derivado):")
    r = onto.ratao_do_banhado
    print(f"  {r.name}  rdf:type Mamifero")
    print(f"    nomeCientifico: {r.nomeCientifico[0]}")
    print(f"    habita: {r.habita[0].name}   atravessa: {r.atravessa[0].name}")
    print(f"    rdfs:comment (proveniência): {r.comment[0]}")


if __name__ == "__main__":
    main()
