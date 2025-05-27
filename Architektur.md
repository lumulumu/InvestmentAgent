# Architekturübersicht des InvestmentAgent-Projekts

Diese Anleitung beschreibt den Aufbau des Skripts `investment_agents.py` und erläutert die Abläufe, so dass ein ähnliches Konzept mit einem anderen Multi‑Agent‑Framework (z.B. Microsoft AutoGen) umgesetzt werden kann.

## 1. Abhängigkeiten
Die Datei `requirements.txt` listet alle benutzten Pakete. Wichtig sind insbesondere:

- `openai` – Zugriff auf das OpenAI API.
- `openai-agents` – Agents SDK zur Definition und Ausführung der Agenten.
- `faiss-cpu` – Vektorspeicher zur Ablage von Embeddings.
- `PyPDF2` – Einlesen und Aufteilen von PDF‑Dokumenten.
- `requests` und `urllib3` – HTTP‑Requests inkl. Retry‑Policy für SerpAPI.
- `numpy` – Verarbeitung numerischer Daten und Embeddings.
- `Jinja2` – Rendering des finalen HTML‑Berichts.
- `nest_asyncio` – Ermöglicht verschachtelte Event‑Loops (z.B. in Jupyter).

## 2. Konfiguration und Initialisierung
Umgebungvariablen definieren API‑Keys und optionale Einstellungen. Aus dem `README.md`:
```
OPENAI_API_KEY – OpenAI API key
SERPAPI_API_KEY – SerpAPI key used for Google search
AGENT_MODEL – optional model name, defaults to gpt-4o
AGENT_DATA_DIR – optional directory for persistent data (defaults to ./data)
```
Der Code liest diese Variablen in `investment_agents.py` ein und setzt eine `Retry`‑Konfiguration für HTTP‑Aufrufe【F:investment_agents.py†L57-L82】.
Es wird ein OpenAI‑Client erstellt und eine `requests`‑Session mit Retry‑Adapter konfiguriert.【F:investment_agents.py†L76-L81】.

Lokale Daten (Reports, FAISS‑Index und Metadaten) werden im Verzeichnis `AGENT_DATA_DIR` abgelegt. Existiert dort bereits ein Index, wird er geladen, andernfalls neu erstellt.【F:investment_agents.py†L128-L143】.

## 3. Hilfsfunktionen
- `_embed(text)` erzeugt ein normalisiertes Embedding via OpenAI und nutzt LRU‑Caching, um wiederholte Anfragen zu vermeiden.【F:investment_agents.py†L149-L158】
- `_extract_pdf(path)` lädt ein PDF seitenweise hoch, ruft GPT‑Vision zur Textextraktion auf und fügt die Ergebnisse zusammen.【F:investment_agents.py†L161-L202】
- `web_search(query)` führt eine Google‑Suche über SerpAPI durch und gibt die Top‑Ergebnisse zurück.【F:investment_agents.py†L208-L217】
- `vector_memory_impl(...)` stellt einen persistenten Vektor‑Speicher bereit: Embeddings können hinzugefügt, abgefragt oder gelistet werden.【F:investment_agents.py†L220-L243】

Diese Funktionen werden mithilfe von `function_tool` als Tools für die Agenten registriert (z.B. `parse_pdf`, `web_search`, `vector_memory_tool`).

## 4. Agenten
Es existieren mehrere spezialisierte Agenten, definiert über das Agents SDK:

- `FinancialHealthAgent` – extrahiert Finanzkennzahlen.
- `MarketOpportunityAgent` – analysiert Markt und Wettbewerb.
- `RiskAssessmentAgent` – identifiziert finanzielle und operative Risiken.
- `AlveusFitAgent` – prüft projektspezifische Kriterien.
- `ReportAgent` – fasst die Resultate in einem schlanken JSON zusammen.
- `SupervisorAgent` – bewertet die Ergebnisse und entscheidet YES/NO bzw. fordert einen RETRY an.

Jeder Agent erhält eine Instruktion und je nach Bedarf Zugriff auf bestimmte Tools (PDF‑Parser, Websuche, Vektorspeicher). Die Definitionen sind in den Zeilen 252‑271 zu finden.【F:investment_agents.py†L252-L271】

## 5. Orchestrierung
Die zentrale Funktion `evaluate(pdf, project)` steuert den Ablauf:
1. Das PDF wird mit `_extract_pdf` in reinen Text umgewandelt.
2. Alle Spezialagenten werden parallel über `Runner.run` auf diesen Text angewendet.【F:investment_agents.py†L317-L320】
3. Anschließend wird das zusammengesetzte Ergebnis vom `ReportAgent` in ein JSON‑Objekt verwandelt. Falls das JSON nicht direkt geparst werden kann, wird per Regex nachgeholfen.【F:investment_agents.py†L325-L340】
4. Der `SupervisorAgent` erhält die aktuelle Zusammenfassung sowie vergangene Ergebnisse aus dem Vektorspeicher, um eine Entscheidung zu treffen (YES/NO) und ggf. eine Begründung auszugeben.【F:investment_agents.py†L342-L349】
5. Das Resultat wird im FAISS‑Vektorstore gespeichert und ein HTML‑Bericht über Jinja2 erzeugt. Der Pfad zum Bericht wird im Rückgabewert festgehalten.【F:investment_agents.py†L355-L367】

Diese Funktion dient ebenfalls als CLI‑Entry‑Point und kann direkt mit `python investment_agents.py <pdf> <projektname>` aufgerufen werden. Ein Fallback sorgt dafür, dass auch in Umgebungen mit bereits laufendem Event‑Loop (z.B. Jupyter) ausgeführt werden kann.【F:investment_agents.py†L372-L389】

## 6. Tests
Im Ordner `tests` befindet sich ein Pytest‑Skript, das Kernfunktionen prüft. Dort werden die Agentenaufrufe durch Dummy‑Funktionen ersetzt, um die Pipeline ohne API‑Zugriff zu testen. Zudem wird der Vektorspeicher getestet (Add, Query, List).【F:tests/test_evaluate.py†L16-L60】

## 7. Übertragbarkeit auf andere Frameworks
Um das Konzept z.B. in Microsoft AutoGen nachzubilden, sollte man folgende Punkte berücksichtigen:

1. **Tools implementieren:** PDF‑Parsing, Websuche und Vektor‑Speicher müssen als aufrufbare Funktionen bereitstehen. AutoGen bietet ebenfalls die Möglichkeit, Tools oder Skills zu registrieren.
2. **Agentenrollen definieren:** Jede Rolle (Finanzen, Markt, Risiko, Projekt‑Fit, Report, Supervisor) erhält eine präzise Instruktion und ggf. Zugriff auf einzelne Tools. Die Parallelisierung der Agenten lässt sich in AutoGen mit asynchronen Calls oder Multi‑Agent‑Sessions realisieren.
3. **Speicherung und Gedächtnis:** Der FAISS‑basierte Speicher dient dazu, frühere Projekte als Kontext zu verwenden. Im anderen Framework müsste ein vergleichbarer Vektorstore eingebunden werden.
4. **Ablauf:** Zuerst Text extrahieren, anschließend Spezialagenten parallel ausführen, Resultat aggregieren, von einer übergeordneten Instanz bewerten lassen und schließlich Bericht erzeugen und Daten persistieren.
5. **Fehlerbehandlung & Retries:** Sowohl OpenAI‑ als auch SerpAPI‑Aufrufe sind mit Retry‑Logik versehen. Ein alternatives Framework sollte ähnliche Mechanismen besitzen.

Mit dieser Übersicht lässt sich die Struktur des InvestmentAgent-Projekts in ein beliebiges Multi‑Agent‑Framework übertragen.
