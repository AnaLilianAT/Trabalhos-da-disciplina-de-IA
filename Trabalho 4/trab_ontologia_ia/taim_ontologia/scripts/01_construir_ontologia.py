# -*- coding: utf-8 -*-
"""
Fase 1 — Construção da ontologia (TBox) do Banhado do Taim.

Define a TBox da ontologia de atropelamentos de fauna no Banhado do Taim (RS):
classes e hierarquia, propriedades de objeto e de dados (sempre com domain/range),
características OWL (simétrica, transitiva, funcional), restrições e uma classe
definida por equivalência. Ao final salva `taim.owl` em RDF/XML e tenta rodar o
reasoner HermiT (se houver Java).

Mínimos atendidos: 15+ classes, 10+ object properties, 10+ data properties,
relação temporal, relação espacial e restrições.

Execução:
    .\\venv\\Scripts\\python.exe scripts\\01_construir_ontologia.py
"""

import datetime
import sys
from pathlib import Path

# Console do Windows pode usar cp1252; forçamos UTF-8 para imprimir acentos e ≡.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

from owlready2 import (
    Thing,
    ObjectProperty,
    DataProperty,
    FunctionalProperty,
    SymmetricProperty,
    TransitiveProperty,
    AllDisjoint,
    Or,
    get_ontology,
    sync_reasoner,
    types,
)

# ---------------------------------------------------------------------------
# Caminhos: o .owl é salvo na raiz do projeto (taim_ontologia/taim.owl).
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
OWL_PATH = BASE_DIR / "taim.owl"

# IRI base do enunciado.
IRI_BASE = "http://www.taim.org/ontologia#"
onto = get_ontology(IRI_BASE)

# ===========================================================================
# 2.1 — CLASSES E HIERARQUIA (mínimo 15 — modelamos ~31)
# ===========================================================================
with onto:

    # ---- Animais ----------------------------------------------------------
    class Animal(Thing):
        """Qualquer animal da fauna do Taim sujeito a atropelamento."""

    class Mamifero(Animal):
        pass

    class Capivara(Mamifero):
        """Hydrochoerus hydrochaeris — espécie emblemática de atropelamento no Taim."""

    class Bugio(Mamifero):
        pass

    class GraxaimDoMato(Mamifero):
        pass

    class Ave(Animal):
        pass

    class AveAquatica(Ave):
        pass

    class Reptil(Animal):
        pass

    class Jacare(Reptil):
        pass

    class Cagado(Reptil):
        pass

    # ---- Habitats ---------------------------------------------------------
    class Habitat(Thing):
        """Ambiente onde a fauna vive (banhado, corpos d'água, vegetação)."""

    class Banhado(Habitat):
        pass

    class CorpoDagua(Habitat):
        pass

    class Lagoa(CorpoDagua):
        pass

    class Canal(CorpoDagua):
        pass

    class AreaAlagada(Habitat):
        pass

    class Vegetacao(Habitat):
        pass

    # ---- Infraestrutura viária -------------------------------------------
    class InfraestruturaViaria(Thing):
        pass

    class Rodovia(InfraestruturaViaria):
        pass

    class TrechoRodovia(InfraestruturaViaria):
        """Segmento de rodovia (delimitado por km início/fim) que corta o Taim."""

    # ---- Eventos (reificação do atropelamento/travessia) ------------------
    class Evento(Thing):
        """Evento reificado: tem participantes e atributos (data/hora, clima...)."""

    class EventoAtropelamento(Evento):
        pass

    class EventoTravessia(Evento):
        pass

    # ---- Condições ambientais --------------------------------------------
    class CondicaoAmbiental(Thing):
        pass

    class CondicaoClimatica(CondicaoAmbiental):
        """Ex.: chuva_forte, sol, neblina, garoa."""

    class PeriodoDia(CondicaoAmbiental):
        """Ex.: madrugada, manhã, tarde, noite."""

    class Estacao(CondicaoAmbiental):
        """Ex.: verão, outono, inverno, primavera."""

    # ---- Fatores de risco -------------------------------------------------
    class FatorRisco(Thing):
        pass

    class FatorTrafego(FatorRisco):
        pass

    class FatorVisibilidade(FatorRisco):
        pass

    class FatorProximidadeAgua(FatorRisco):
        pass

# ===========================================================================
# 2.2 — PROPRIEDADES DE OBJETO (mínimo 10 — modelamos 14)
# Sempre com domain e range. Características OWL marcadas onde couber.
# ===========================================================================
with onto:

    class envolveAnimal(ObjectProperty):
        """Liga um Evento aos animais que dele participam."""
        domain = [Evento]
        range = [Animal]

    class ocorreEmTrecho(ObjectProperty):
        domain = [Evento]
        range = [TrechoRodovia]

    class ocorreSobClima(ObjectProperty):
        domain = [Evento]
        range = [CondicaoClimatica]

    class ocorreNoPeriodo(ObjectProperty):
        """Relação TEMPORAL: período do dia em que o evento ocorreu."""
        domain = [Evento]
        range = [PeriodoDia]

    class ocorreNaEstacao(ObjectProperty):
        """Relação TEMPORAL: estação do ano do evento."""
        domain = [Evento]
        range = [Estacao]

    class proximoA(ObjectProperty):
        """Relação ESPACIAL: trecho próximo a um habitat."""
        domain = [TrechoRodovia]
        range = [Habitat]

    class pertenceARodovia(ObjectProperty, FunctionalProperty):
        """FUNCIONAL: cada trecho pertence a exatamente uma rodovia."""
        domain = [TrechoRodovia]
        range = [Rodovia]

    class habita(ObjectProperty):
        """Relação ecológica: animal habita um habitat."""
        domain = [Animal]
        range = [Habitat]

    class viveEm(ObjectProperty):
        """Sinônimo de `habita` (usado pela extração da Fase 3)."""
        domain = [Animal]
        range = [Habitat]
        equivalent_to = [habita]

    class atravessa(ObjectProperty):
        domain = [Animal]
        range = [Rodovia]

    class temFatorRisco(ObjectProperty):
        domain = [TrechoRodovia]
        range = [FatorRisco]

    class contemVegetacao(ObjectProperty):
        domain = [Habitat]
        range = [Vegetacao]

    class adjacenteA(ObjectProperty, SymmetricProperty):
        """ESPACIAL e SIMÉTRICA: trechos vizinhos na rodovia."""
        domain = [TrechoRodovia]
        range = [TrechoRodovia]

    class precedeEvento(ObjectProperty, TransitiveProperty):
        """TEMPORAL e TRANSITIVA: ordena eventos no tempo."""
        domain = [Evento]
        range = [Evento]

# ===========================================================================
# 2.3 — PROPRIEDADES DE DADOS (mínimo 10 — modelamos 14)
# ===========================================================================
with onto:

    class nomeComum(DataProperty):
        domain = [Animal]
        range = [str]

    class nomeCientifico(DataProperty):
        domain = [Animal]
        range = [str]

    class pesoMedioKg(DataProperty):
        domain = [Animal]
        range = [float]

    class kmInicio(DataProperty):
        domain = [TrechoRodovia]
        range = [float]

    class kmFim(DataProperty):
        domain = [TrechoRodovia]
        range = [float]

    class latitude(DataProperty):
        """ESPACIAL: vale para trechos e habitats (domínio = união)."""
        domain = [Or([TrechoRodovia, Habitat])]
        range = [float]

    class longitude(DataProperty):
        """ESPACIAL: vale para trechos e habitats (domínio = união)."""
        domain = [Or([TrechoRodovia, Habitat])]
        range = [float]

    class dataHora(DataProperty, FunctionalProperty):
        """TEMPORAL e FUNCIONAL: cada atropelamento tem uma única data/hora."""
        domain = [EventoAtropelamento]
        range = [datetime.datetime]

    class numeroAnimaisEnvolvidos(DataProperty):
        domain = [EventoAtropelamento]
        range = [int]

    class descricaoClima(DataProperty):
        domain = [CondicaoClimatica]
        range = [str]

    class temperaturaC(DataProperty):
        domain = [CondicaoClimatica]
        range = [float]

    class visibilidadeMetros(DataProperty):
        domain = [CondicaoClimatica]
        range = [float]

    class volumeTrafegoDiario(DataProperty):
        domain = [Or([FatorTrafego, TrechoRodovia])]
        range = [int]

    class nivelRisco(DataProperty):
        """Nível de risco do trecho, em [0, 1]."""
        domain = [TrechoRodovia]
        range = [float]

# ===========================================================================
# 2.4 — RESTRIÇÕES (obrigatório) + classes disjuntas
# ===========================================================================
with onto:

    # Cardinalidade mínima: todo atropelamento envolve >= 1 animal.
    EventoAtropelamento.is_a.append(envolveAnimal.min(1, Animal))

    # Cardinalidade exata: exatamente 1 trecho por atropelamento.
    EventoAtropelamento.is_a.append(ocorreEmTrecho.exactly(1, TrechoRodovia))

    # Existencial (some): todo Banhado contém alguma Vegetação.
    Banhado.is_a.append(contemVegetacao.some(Vegetacao))

    # Universal (only): travessias só envolvem Animal.
    EventoTravessia.is_a.append(envolveAnimal.only(Animal))

    # Classe DEFINIDA por equivalência (útil ao reasoner e à explicabilidade):
    # trecho próximo a um banhado E com fator de proximidade da água.
    class TrechoCritico(TrechoRodovia):
        """Trecho crítico: próximo a banhado e com fator de proximidade da água."""
        equivalent_to = [
            TrechoRodovia
            & proximoA.some(Banhado)
            & temFatorRisco.some(FatorProximidadeAgua)
        ]

    # Classes mutuamente disjuntas (permitem o reasoner detectar inconsistências).
    AllDisjoint([Animal, Habitat, Evento, InfraestruturaViaria,
                 CondicaoAmbiental, FatorRisco])


# ===========================================================================
# CONTAGENS E SALVAMENTO
# ===========================================================================
def _contar():
    """Conta classes, object properties e data properties da ontologia."""
    n_classes = len(list(onto.classes()))
    n_obj = len(list(onto.object_properties()))
    n_data = len(list(onto.data_properties()))
    return n_classes, n_obj, n_data


def main():
    n_classes, n_obj, n_data = _contar()

    print("=" * 60)
    print("FASE 1 — Construção da TBox (ontologia do Taim)")
    print("=" * 60)
    print(f"Classes ...................... {n_classes}  (mínimo 15)")
    print(f"Object properties ............ {n_obj}  (mínimo 10)")
    print(f"Data properties .............. {n_data}  (mínimo 10)")

    # Verificação dos mínimos da rubrica.
    assert n_classes >= 15, "FALHA: menos de 15 classes."
    assert n_obj >= 10, "FALHA: menos de 10 object properties."
    assert n_data >= 10, "FALHA: menos de 10 data properties."
    print("OK: mínimos da rubrica atendidos.")

    # Relação temporal e espacial declaradas (para o relatório).
    print("\nRelações TEMPORAIS: ocorreNoPeriodo, ocorreNaEstacao, "
          "precedeEvento, dataHora")
    print("Relações ESPACIAIS: proximoA, adjacenteA, latitude, longitude")

    print("\nRestrições definidas em EventoAtropelamento:")
    for r in EventoAtropelamento.is_a:
        print(f"  - {r}")
    print("Classe definida TrechoCritico ≡", TrechoCritico.equivalent_to[0])

    # ---- Salvar em RDF/XML (abrível no Protégé) --------------------------
    onto.save(file=str(OWL_PATH), format="rdfxml")
    print(f"\nOntologia salva em: {OWL_PATH}")

    # ---- Reasoner (HermiT precisa de Java) -------------------------------
    print("\nTentando rodar o reasoner HermiT (sync_reasoner)...")
    try:
        with onto:
            sync_reasoner()
        print("OK: reasoner executou sem reportar inconsistência.")
    except Exception as exc:  # noqa: BLE001 — queremos seguir mesmo sem Java.
        print("AVISO: não foi possível rodar o reasoner via código.")
        print(f"       Motivo: {exc}")
        print("       (Java/HermiT ausente — validar consistência no Protégé. "
              "Ver AMBIENTE.md.)")


if __name__ == "__main__":
    main()
