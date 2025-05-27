# AutoGen Variante des InvestmentAgent

Dieses Dokument beschreibt, wie das Konzept aus `investment_agents.py` mit dem Multi-Agent-Framework **Microsoft AutoGen** umgesetzt werden kann. Die Implementierung befindet sich in `autogen_agents.py` und orientiert sich eng an der bestehenden Architektur, nutzt jedoch AutoGen-spezifische Features, insbesondere eine integrierte Gesprächsmemory.

## Wichtige Unterschiede zur Agents-SDK-Version

1. **Agentenklassen**: Spezial- und Supervisor-Agenten werden als `AssistantAgent` erzeugt. Werkzeuge wie PDF-Parsing oder Websuche lassen sich mit `@autogen.tool` registrieren und stehen den Agenten zur Verfügung.
2. **Gedächtnis**: Jeder Agent erhält eine `SummaryBufferMemory` mit Tokenlimit, sodass lange Gespräche automatisch zusammengefasst werden und weniger Kontexttoken verbrauchen【F:autogen_agents.py†L133-L140】.
3. **Orchestrierung**: Die Funktion `evaluate` startet alle Spezialagenten parallel mittels `asyncio` und ruft anschließend Supervisor und Report-Agent auf. Das Ergebnis wird wie gewohnt als Markdown abgelegt【F:autogen_agents.py†L243-L285】.
4. **Vektor-Gedächtnis**: Die bestehende FAISS-basierte Historie bleibt erhalten und wird über das Tool `vector_memory` eingebunden【F:autogen_agents.py†L114-L126】【F:autogen_agents.py†L176-L179】.

## Ausführen der Pipeline

```bash
python autogen_agents.py <PDF-Pfad> <Projektname>
```

Installiere zuerst die Abhängigkeiten aus `requirements-autogen.txt`:

```bash
pip install -r requirements-autogen.txt
```

Die Umgebung benötigt wie gewohnt die Variablen `OPENAI_API_KEY` und `SERPAPI_API_KEY`. Optional kann `AGENT_MODEL` (z. B. `gpt-4o`) gesetzt werden. Berichte werden im Verzeichnis `data/reports` abgelegt.

## Hinweise zur Speicheroptimierung

- Durch die `SummaryBufferMemory` werden Dialoge automatisch gekürzt, wodurch weniger Tokens an das Modell gesendet werden.
- Historische Projektdaten bleiben über FAISS verfügbar und können mit dem Tool `vector_memory` abgefragt werden.
- Sollten weitere AutoGen-Features wie längerfristige Memories benötigt werden, lassen sich diese über die `memory`-Parameter der Agenten ergänzen.
