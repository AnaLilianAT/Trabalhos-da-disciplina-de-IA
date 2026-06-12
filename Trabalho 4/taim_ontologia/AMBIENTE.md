# Ambiente de Execução — Ontologia do Banhado do Taim

Registro do ambiente reprodutível usado no trabalho (Fase 0).

## Sistema Operacional

- Windows 11 Home Single Language (10.0.26200)
- Shell: PowerShell

## Python e ambiente virtual

- **Python:** 3.14.5
- Ambiente virtual isolado em `taim_ontologia/venv/` (criado com `python -m venv venv`).
- Todos os scripts devem ser executados com o interpretador do venv:
  - Windows/PowerShell: `.\venv\Scripts\python.exe scripts\01_construir_ontologia.py`

## Dependências instaladas (versões exatas)

| Pacote | Versão |
|---|---|
| owlready2 | 0.50 |
| rdflib | 7.6.0 |
| requests | 2.34.2 |
| beautifulsoup4 | 4.14.3 |

Dependências transitivas relevantes: pyparsing 3.3.2, urllib3 2.7.0, certifi 2026.5.20,
charset-normalizer 3.4.7, idna 3.18, soupsieve 2.8.4, typing-extensions 4.15.0.

> spaCy e scikit-learn são **opcionais** (só necessários se as Fases 3/5 usarem NLP/ML).
> Estão comentados no `requirements.txt`.

## Java / Reasoner HermiT

- **Java instalado:** Java SE 26.0.1 (HotSpot 64-Bit Server VM), no PATH.
- O reasoner **HermiT** (acionado por `owlready2.sync_reasoner()`) **roda via código**:
  executa em ~1,4 s e classifica a ontologia sem reportar inconsistência.
  Ele inclusive infere a equivalência `habita ≡ viveEm`.
- Como alternativa/conferência, o grupo também pode rodar o reasoner no **Protégé**
  (que traz o HermiT embutido) ao abrir o `taim.owl`.

## Como reproduzir do zero

```powershell
cd taim_ontologia
python -m venv venv
.\venv\Scripts\python.exe -m pip install --upgrade pip
.\venv\Scripts\python.exe -m pip install -r requirements.txt
# Critério de aceitação da Fase 0:
.\venv\Scripts\python.exe -c "import owlready2, rdflib"   # roda sem erro
```
