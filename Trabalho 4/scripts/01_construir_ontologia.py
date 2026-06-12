"""
Fase 1 — Construção da ontologia (TBox) do Banhado do Taim.
"""

import datetime
import sys
from pathlib import Path

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

BASE_DIR = Path(__file__).resolve().parent.parent
OWL_PATH = BASE_DIR / "taim.owl"

# IRI base do enunciado.
IRI_BASE = "http://www.taim.org/ontologia#"
onto = get_ontology(IRI_BASE)

# ===========================================================================
# CLASSES E HIERARQUIA
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
# PROPRIEDADES DE OBJETO
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
# PROPRIEDADES DE DADOS
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
# RESTRIÇÕES + classes disjuntas
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

    # Classe DEFINIDA por equivalência:
    # trecho próximo a um banhado E com fator de proximidade da água.
    class TrechoCritico(TrechoRodovia):
        """Trecho crítico: próximo a banhado e com fator de proximidade da água."""
        equivalent_to = [
            TrechoRodovia
            & proximoA.some(Banhado)
            & temFatorRisco.some(FatorProximidadeAgua)
        ]

    # Classes mutuamente disjuntas.
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
    print(f"Classes ...................... {n_classes}")
    print(f"Object properties ............ {n_obj}")
    print(f"Data properties .............. {n_data}")

    # Relação temporal e espacial declaradas.
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

    # ---- Reasoner -------------------------------
    print("\nTentando rodar o reasoner HermiT (sync_reasoner)...")
    try:
        with onto:
            sync_reasoner()
        print("OK: reasoner executou sem reportar inconsistência.")
    except Exception as exc: 
        print("AVISO: não foi possível rodar o reasoner via código.")
        print(f"       Motivo: {exc}")
        print("       (Java/HermiT ausente — validar consistência no Protégé. "
              "Ver AMBIENTE.md.)")


if __name__ == "__main__":
    main()
