# Detaillierte Architektur des InvestmentAgent-Projekts

Diese Anleitung fasst den gesamten Aufbau des Skripts `investment_agents.py` zusammen. Sie soll es einem anderen Bot ermöglichen, das Konzept mit einem beliebigen Multi-Agent-Framework (z.B. Microsoft AutoGen) nachzubauen. Die aktuelle Umsetzung verwendet das OpenAI Agents SDK, lässt sich aber leicht übertragen.

## 1. Verwendete Pakete

Die Datei `requirements.txt` enthält alle Abhängigkeiten. Wichtig sind insbesondere:

- **openai** – Zugriff auf das OpenAI API.
- **openai-agents** – Agents SDK zur Definition der Agenten.
- **faiss-cpu** – Vektorspeicher für Embeddings.
- **PyPDF2** – Aufteilen und Lesen von PDF-Dateien.
- **requests** und **urllib3** – HTTP-Anfragen mit Retry-Logik für SerpAPI.
- **numpy** – Berechnung und Speicherung der Embedding-Vektoren.
- **nest_asyncio** – optional, um verschachtelte Event-Loops zu erlauben.

Die Liste findet sich in `requirements.txt`【F:requirements.txt†L1-L20】.

## 2. Konfiguration und Initialisierung

`investment_agents.py` liest mehrere Umgebungsvariablen ein und richtet eine globale Retry-Policy ein. Die wichtigsten Variablen sind `OPENAI_API_KEY`, `SERPAPI_API_KEY`, ein optionales `AGENT_MODEL` und `AGENT_DATA_DIR` für persistente Daten. Die Initialisierung erfolgt in den Zeilen 57–71 und richtet sowohl den OpenAI-Client als auch eine Requests-Session mit Retry-Adapter ein【F:investment_agents.py†L57-L82】.

Anschließend werden die Pfade zum Datenverzeichnis, dem FAISS-Index und den Metadaten gesetzt. Der Vektorspeicher wird über `VectorStore` geladen bzw. neu erstellt【F:investment_agents.py†L126-L139】【F:vector_store.py†L9-L26】.

## 3. Hilfsfunktionen und Tools

Mehrere Hilfsfunktionen dienen später als Tools für die Agenten:

1. **_embed(text)** erzeugt ein normiertes Embedding mit OpenAI und ist mit LRU-Cache versehen【F:investment_agents.py†L145-L154】.
2. **_extract_pdf(path)** schneidet das PDF in Seitenblöcke, lädt diese hoch und lässt GPT-Vision den Text extrahieren【F:investment_agents.py†L157-L198】.
3. **web_search(query)** ruft die SerpAPI mit einer Retry-Logik auf und sammelt die ersten Treffer【F:investment_agents.py†L204-L213】.
4. **vector_memory_impl(...)** verwaltet die Einträge im FAISS-Index und kann Daten hinzufügen, abfragen oder auflisten【F:investment_agents.py†L216-L244】.

Diese Funktionen werden als Tools (über `function_tool`) registriert und stehen den Agenten je nach Rolle zur Verfügung.

## 4. Agentendefinitionen

Das Skript definiert sechs spezialisierte Agenten. Jeder erhält eine feste Instruktion und Zugriff auf ausgewählte Tools:

- **FinancialHealthAgent** – analysiert Finanzkennzahlen.
- **MarketOpportunityAgent** – bewertet Markt und Wettbewerb.
- **RiskAssessmentAgent** – identifiziert wesentliche Risiken.
- **AlveusFitAgent** – prüft projektspezifische Kriterien des Investors.
- **ReportAgent** – erzeugt einen Markdown-Bericht aus den Zwischenergebnissen.
- **SupervisorAgent** – beurteilt die Gesamtergebnisse (YES/NO bzw. RETRY) und greift auf frühere Projekte im Vektorspeicher zu.

Die Definitionen befinden sich in den Zeilen 253–280 des Skripts【F:investment_agents.py†L253-L280】.

## 5. Ablauf der Funktion `evaluate`

Die Orchestrierung geschieht in `evaluate(pdf, project)`:

1. Das PDF wird per `_extract_pdf` in Text umgewandelt.
2. Vier Spezialagenten (Financial, Market, Risk, AlveusFit) werden parallel auf diesen Text angesetzt【F:investment_agents.py†L324-L331】.
3. Deren Ergebnisse und frühere Projektdaten bilden den Input für den Supervisor. Dieser entscheidet und liefert eine rationale Begründung【F:investment_agents.py†L333-L344】.
4. Mit allen gesammelten Informationen erzeugt der ReportAgent den finalen Markdown-Bericht【F:investment_agents.py†L346-L355】.
5. Anschließend werden Summary, Keywords und Metrics aus dem Markdown extrahiert und der FAISS-Store aktualisiert. Der Bericht wird im Ordner `data/reports` abgelegt【F:investment_agents.py†L357-L372】.

`evaluate` gibt ein `EvaluationResult` zurück, das den Pfad zum gespeicherten Bericht enthält.

## 6. Kommandozeilen-Aufruf

Wird `investment_agents.py` direkt ausgeführt, erwartet es ein PDF und einen Projektnamen. Das Skript startet dann die asynchrone Pipeline. Falls bereits ein Event-Loop läuft (z.B. in Jupyter), wird `nest_asyncio` genutzt, um den bestehenden Loop zu verwenden【F:investment_agents.py†L375-L396】.

## 7. Tests

`tests/test_evaluate.py` simuliert die gesamte Pipeline ohne Netzwerkzugriff. Embeddings und Agentenaufrufe werden durch Dummy-Funktionen ersetzt, um die Speicherung und Verarbeitung zu testen. Ebenso wird der Vektorspeicher verifiziert (Hinzufügen, Abfragen, Auflisten)【F:tests/test_evaluate.py†L16-L60】.

## 8. Übertragbarkeit auf andere Frameworks

Um die gleiche Logik in einem anderen Framework wie Microsoft AutoGen zu implementieren, sollten folgende Konzepte berücksichtigt werden:

1. **Tools implementieren:** PDF-Parsing, Websuche und Vektor-Gedächtnis müssen als aufrufbare Funktionen bereitstehen. In AutoGen können diese als "Skills" registriert werden.
2. **Agentenrollen nachbilden:** Jede im Skript definierte Agentenrolle benötigt eine entsprechende Instruktion und Zugriff auf genau die Tools, die hier genutzt werden.
3. **Parallele Ausführung:** Die Spezialagenten laufen zeitgleich. In AutoGen ließe sich dies durch parallele Aufrufe oder Multi-Agent-Sessions umsetzen.
4. **Speicher für frühere Projekte:** Das Beispiel verwendet FAISS für Embeddings. In anderen Frameworks sollte ein kompatibler Vektorstore angebunden werden.
5. **Supervisor-Logik:** Nach Auswertung der Spezialagenten wird eine zentrale Instanz benötigt, die entscheidet, ob das Ergebnis ausreichend ist oder erneut versucht werden soll. Diese Instanz sollte frühere Projekte in die Entscheidung einbeziehen.
6. **Markdown-Bericht generieren:** Der finale Bericht wird direkt als Markdown erstellt. Ein anderes Framework kann dieses Format einfach übernehmen.

Mit dieser Beschreibung lässt sich die Architektur des InvestmentAgent in einer beliebigen Multi-Agent-Umgebung reproduzieren.
