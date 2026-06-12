"""
Fase 2 — Povoamento (ABox) da ontologia do Banhado do Taim.
"""

import datetime
import random
import sys
from pathlib import Path

from owlready2 import get_ontology, sync_reasoner

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

# Reprodutibilidade.
random.seed(42)

# ---------------------------------------------------------------------------
# Carrega a TBox salva na Fase 1.
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
OWL_PATH = BASE_DIR / "taim.owl"
with open(OWL_PATH, "rb") as _f:
    onto = get_ontology(OWL_PATH.as_uri()).load(fileobj=_f)

# Coordenadas reais aproximadas da região da ESEC Taim / Lagoa Mangueira.
LAT_BASE, LON_BASE = -32.5, -52.5


def coord():
    """Gera (lat, lon) plausível variando em torno da base do Taim."""
    return (round(LAT_BASE + random.uniform(-0.20, 0.20), 5),
            round(LON_BASE + random.uniform(-0.20, 0.20), 5))


with onto:

    # =======================================================================
    # HABITATS
    # =======================================================================
    banhado = onto.Banhado("banhado_do_taim")
    lat, lon = coord(); banhado.latitude = [lat]; banhado.longitude = [lon]

    veg_palustre = onto.Vegetacao("vegetacao_palustre")
    veg_ciliar = onto.Vegetacao("vegetacao_mata_ciliar")
    # Satisfaz a restrição existencial: Banhado contemVegetacao some Vegetacao.
    banhado.contemVegetacao = [veg_palustre, veg_ciliar]

    lagoa_mangueira = onto.Lagoa("lagoa_mangueira")
    lagoa_mirim = onto.Lagoa("lagoa_mirim")
    lagoa_nicola = onto.Lagoa("lagoa_nicola")
    canal_sg = onto.Canal("canal_sao_goncalo")
    canal_taim = onto.Canal("canal_taim")
    area_alagada_1 = onto.AreaAlagada("area_alagada_norte")
    area_alagada_2 = onto.AreaAlagada("area_alagada_sul")

    habitats_aquaticos = [banhado, lagoa_mangueira, lagoa_mirim, lagoa_nicola,
                          canal_sg, canal_taim, area_alagada_1, area_alagada_2]
    for h in habitats_aquaticos:
        la, lo = coord(); h.latitude = [la]; h.longitude = [lo]

    habitats = habitats_aquaticos + [veg_palustre, veg_ciliar]  # 10 habitats

    # =======================================================================
    # ESPÉCIES / ANIMAIS
    # =======================================================================
    # (classe, nome_individuo, nomeComum, nomeCientifico, pesoMedioKg)
    especies_def = [
        (onto.Capivara, "capivara_01", "Capivara", "Hydrochoerus hydrochaeris", 55.0),
        (onto.Capivara, "capivara_02", "Capivara", "Hydrochoerus hydrochaeris", 48.5),
        (onto.Capivara, "capivara_03", "Capivara", "Hydrochoerus hydrochaeris", 61.2),
        (onto.Bugio, "bugio_01", "Bugio-ruivo", "Alouatta guariba", 6.5),
        (onto.Bugio, "bugio_02", "Bugio-ruivo", "Alouatta guariba", 5.8),
        (onto.GraxaimDoMato, "graxaim_01", "Graxaim-do-mato", "Cerdocyon thous", 6.0),
        (onto.GraxaimDoMato, "graxaim_02", "Graxaim-do-mato", "Cerdocyon thous", 7.1),
        (onto.AveAquatica, "garca_branca_01", "Garça-branca-grande", "Ardea alba", 1.0),
        (onto.AveAquatica, "marreco_01", "Marreco-pardo", "Anas georgica", 0.6),
        (onto.AveAquatica, "frango_dagua_01", "Frango-d'água", "Gallinula galeata", 0.35),
        (onto.AveAquatica, "colhereiro_01", "Colhereiro", "Platalea ajaja", 1.4),
        (onto.Jacare, "jacare_01", "Jacaré-do-papo-amarelo", "Caiman latirostris", 30.0),
        (onto.Jacare, "jacare_02", "Jacaré-do-papo-amarelo", "Caiman latirostris", 24.5),
        (onto.Cagado, "cagado_01", "Cágado-de-barbicha", "Phrynops hilarii", 3.5),
        (onto.Cagado, "cagado_02", "Cágado-de-barbicha", "Phrynops hilarii", 4.2),
    ]
    animais = []
    for cls, nome, comum, cientifico, peso in especies_def:
        a = cls(nome)
        a.nomeComum = [comum]
        a.nomeCientifico = [cientifico]
        a.pesoMedioKg = [peso]
        a.habita = [random.choice(habitats_aquaticos)]
        animais.append(a)

    # =======================================================================
    # RODOVIA + TRECHOS
    # =======================================================================
    br471 = onto.Rodovia("BR_471")

    # ---- FATORES DE RISCO -------------------------------------------
    trafego_alto = onto.FatorTrafego("trafego_alto")
    trafego_alto.volumeTrafegoDiario = [4200]
    trafego_medio = onto.FatorTrafego("trafego_medio")
    trafego_medio.volumeTrafegoDiario = [1800]
    trafego_baixo = onto.FatorTrafego("trafego_baixo")
    trafego_baixo.volumeTrafegoDiario = [600]

    vis_baixa = onto.FatorVisibilidade("visibilidade_baixa")
    vis_media = onto.FatorVisibilidade("visibilidade_media")

    prox_agua_alta = onto.FatorProximidadeAgua("proximidade_agua_alta")
    prox_agua_media = onto.FatorProximidadeAgua("proximidade_agua_media")
    prox_agua_baixa = onto.FatorProximidadeAgua("proximidade_agua_baixa")

    fatores_trafego = [trafego_alto, trafego_medio, trafego_baixo]
    fatores_vis = [vis_baixa, vis_media]
    fatores_prox = [prox_agua_alta, prox_agua_media, prox_agua_baixa]

    # ---- TRECHOS ----------------------------------------------------------
    kms = [498, 502, 506, 510, 514, 518, 522, 526, 530, 534, 538, 542]
    trechos = []
    for km in kms:
        t = onto.TrechoRodovia(f"trecho_BR471_km{km}")
        t.kmInicio = [float(km)]
        t.kmFim = [float(km + 4)]
        la, lo = coord(); t.latitude = [la]; t.longitude = [lo]
        t.pertenceARodovia = br471         
        t.nivelRisco = [round(random.uniform(0.1, 0.95), 2)]
        t.volumeTrafegoDiario = [random.choice([600, 1800, 4200])]
        # Proximidade a habitat: metade dos trechos perto do banhado.
        t.proximoA = [random.choice(habitats_aquaticos)]
        # Fator de tráfego sempre; visibilidade às vezes.
        t.temFatorRisco = [random.choice(fatores_trafego)]
        if random.random() < 0.5:
            t.temFatorRisco.append(random.choice(fatores_vis))
        trechos.append(t)

    # Garante que ALGUNS trechos sejam TrechoCritico
    for t in [trechos[2], trechos[5], trechos[8], trechos[10]]:
        t.proximoA = [banhado]
        t.temFatorRisco.append(prox_agua_alta)

    # Relação espacial simétrica: encadeia trechos vizinhos com adjacenteA.
    for i in range(len(trechos) - 1):
        trechos[i].adjacenteA.append(trechos[i + 1])

    # Animais atravessam a BR-471 (relação ecológica/comportamental).
    for a in random.sample(animais, k=10):
        a.atravessa = [br471]

    # =======================================================================
    # CONDIÇÕES AMBIENTAIS
    # =======================================================================
    # Clima com descrição, temperatura e visibilidade.
    chuva_forte = onto.CondicaoClimatica("chuva_forte")
    chuva_forte.descricaoClima = ["Chuva forte com redução de visibilidade"]
    chuva_forte.temperaturaC = [18.0]; chuva_forte.visibilidadeMetros = [80.0]

    neblina = onto.CondicaoClimatica("neblina")
    neblina.descricaoClima = ["Neblina densa, comum nas madrugadas do Taim"]
    neblina.temperaturaC = [14.0]; neblina.visibilidadeMetros = [40.0]

    garoa = onto.CondicaoClimatica("garoa")
    garoa.descricaoClima = ["Garoa/chuvisco"]
    garoa.temperaturaC = [16.0]; garoa.visibilidadeMetros = [300.0]

    sol = onto.CondicaoClimatica("sol")
    sol.descricaoClima = ["Tempo bom, céu claro"]
    sol.temperaturaC = [26.0]; sol.visibilidadeMetros = [2000.0]
    climas = [chuva_forte, neblina, garoa, sol]

    # Período do dia.
    madrugada = onto.PeriodoDia("madrugada")
    manha = onto.PeriodoDia("manha")
    tarde = onto.PeriodoDia("tarde")
    noite = onto.PeriodoDia("noite")
    periodos = [madrugada, manha, tarde, noite]

    # Estações.
    verao = onto.Estacao("verao")
    outono = onto.Estacao("outono")
    inverno = onto.Estacao("inverno")
    primavera = onto.Estacao("primavera")
    estacoes = [verao, outono, inverno, primavera]

    # =======================================================================
    # EVENTOS DE ATROPELAMENTO
    # =======================================================================
    def estacao_de(mes):
        """Estação do hemisfério sul a partir do mês."""
        if mes in (12, 1, 2):
            return verao
        if mes in (3, 4, 5):
            return outono
        if mes in (6, 7, 8):
            return inverno
        return primavera

    # Período mais provável conforme o horário sorteado.
    def periodo_de(hora):
        if 0 <= hora < 6:
            return madrugada
        if 6 <= hora < 12:
            return manha
        if 12 <= hora < 18:
            return tarde
        return noite

    N_EVENTOS = 50
    eventos = []
    for i in range(1, N_EVENTOS + 1):
        ev = onto.EventoAtropelamento(f"atropelamento_{i:03d}")

        # Data/hora plausível (atropelamentos concentram-se à
        # noite/madrugada, então enviesamos o sorteio do horário).
        mes = random.randint(1, 12)
        dia = random.randint(1, 28)
        hora = random.choice([0, 1, 2, 3, 4, 5, 19, 20, 21, 22, 23,  # noturno (peso maior)
                              0, 2, 4, 20, 22, 8, 14])               # alguns diurnos
        minuto = random.randint(0, 59)
        ev.dataHora = datetime.datetime(2024, mes, dia, hora, minuto)

        # 1 a 3 animais por evento (capivaras tendem a aparecer em grupo).
        k = random.choices([1, 2, 3], weights=[60, 30, 10])[0]
        envolvidos = random.sample(animais, k=k)
        ev.envolveAnimal = envolvidos
        ev.numeroAnimaisEnvolvidos = [len(envolvidos)]

        # Exatamente 1 trecho.
        ev.ocorreEmTrecho = [random.choice(trechos)]

        # Clima enviesado para condições de baixa visibilidade à noite.
        if hora in (0, 1, 2, 3, 4, 5):
            ev.ocorreSobClima = [random.choice([neblina, chuva_forte, garoa, sol])]
        else:
            ev.ocorreSobClima = [random.choice(climas)]

        ev.ocorreNoPeriodo = [periodo_de(hora)]
        ev.ocorreNaEstacao = [estacao_de(mes)]
        eventos.append(ev)

    # Relação temporal TRANSITIVA: encadeia eventos por ordem cronológica
    # dentro de um mesmo trecho (subconjunto), exercitando precedeEvento.
    eventos_ord = sorted(eventos, key=lambda e: e.dataHora)
    for i in range(0, 20, 2):
        eventos_ord[i].precedeEvento.append(eventos_ord[i + 1])


# ===========================================================================
# REASONER, CONTAGEM, EXEMPLOS E SALVAMENTO
# ===========================================================================
def main():
    print("=" * 60)
    print("FASE 2 — Povoamento (ABox) da ontologia do Taim")
    print("=" * 60)

    individuos = list(onto.individuals())
    total = len(individuos)
    print(f"Total de indivíduos criados: {total}")

    # Contagem por categoria (para o relatório).
    print(f"  - Animais ................ {len(list(onto.Animal.instances()))}")
    print(f"  - Habitats ............... {len(list(onto.Habitat.instances()))}")
    print(f"  - Rodovias ............... {len(list(onto.Rodovia.instances()))}")
    print(f"  - Trechos ................ {len(list(onto.TrechoRodovia.instances()))}")
    print(f"  - Cond. ambientais ....... {len(list(onto.CondicaoAmbiental.instances()))}")
    print(f"  - Fatores de risco ....... {len(list(onto.FatorRisco.instances()))}")
    print(f"  - Eventos atropelamento .. {len(list(onto.EventoAtropelamento.instances()))}")

    # Validação das restrições no nível dos dados.
    sem_animal = [e for e in onto.EventoAtropelamento.instances() if not e.envolveAnimal]
    sem_trecho = [e for e in onto.EventoAtropelamento.instances()
                  if len(e.ocorreEmTrecho) != 1]
    sem_data = [e for e in onto.EventoAtropelamento.instances() if e.dataHora is None]
    assert not sem_animal, "FALHA: evento sem animal."
    assert not sem_trecho, "FALHA: evento sem exatamente 1 trecho."
    assert not sem_data, "FALHA: evento sem dataHora."
    print("OK: todo atropelamento tem dataHora, >=1 animal e exatamente 1 trecho.")

    # ---- Salvar (TBox + ABox) ANTES do reasoner --------------------------
    onto.save(file=str(OWL_PATH), format="rdfxml")
    print(f"\nOntologia (com ABox afirmada) salva em: {OWL_PATH}")

    # ---- Reasoner (validação de consistência + classificação) ------------
    print("\nRodando o reasoner HermiT (sync_reasoner) para validar...")
    try:
        with onto:
            sync_reasoner()
        print("OK: reasoner executou sem reportar inconsistência.")
        criticos = list(onto.TrechoCritico.instances())
        print(f"Trechos inferidos como TrechoCritico pelo reasoner: "
              f"{[t.name for t in criticos]}")
    except Exception as exc:
        print(f"AVISO: reasoner não rodou ({exc}). Ver AMBIENTE.md.")

    # ---- 3 exemplos concretos ------------------
    print("\n" + "-" * 60)
    print("EXEMPLOS DE INDIVÍDUOS CRIADOS")
    print("-" * 60)
    for ev in eventos[:3]:
        animais_str = ", ".join(
            f"{a.name} ({a.nomeComum[0]})" for a in ev.envolveAnimal)
        trecho = ev.ocorreEmTrecho[0]
        print(f"\n• {ev.name}")
        print(f"    dataHora ............. {ev.dataHora}")
        print(f"    envolveAnimal ........ {animais_str}")
        print(f"    numeroAnimaisEnvolvidos {ev.numeroAnimaisEnvolvidos[0]}")
        print(f"    ocorreEmTrecho ....... {trecho.name} "
              f"(km {trecho.kmInicio[0]}–{trecho.kmFim[0]})")
        print(f"    ocorreSobClima ....... {ev.ocorreSobClima[0].name}")
        print(f"    ocorreNoPeriodo ...... {ev.ocorreNoPeriodo[0].name}")
        print(f"    ocorreNaEstacao ...... {ev.ocorreNaEstacao[0].name}")

    # Exemplo de animal e de trecho.
    cap = onto.capivara_01
    print(f"\n• {cap.name}: {cap.nomeComum[0]} / {cap.nomeCientifico[0]} / "
          f"{cap.pesoMedioKg[0]} kg / habita {cap.habita[0].name}")


if __name__ == "__main__":
    main()
