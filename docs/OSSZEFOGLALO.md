# 🥞 shiny-pancake: összefoglaló

## Mi ez?

A shiny-pancake egy **multi-agent LLM hub**. Azt mutatja meg, hogyan kell nagy nyelvi modelleket
valódi rendszerekbe integrálni: gyártófüggetlen provider réteg, guardrailekkel védett toolok,
több ágens összehangolása, párhuzamos summary workerek, tartós job queue, élő megfigyelhetőség,
kiértékelés (evals) és MCP mindkét irányban.

Háromféleképpen használható:

1. **Szolgáltatásként**: HTTP API, web UI (Pancake Hub), Docker és docker-compose skálázható workerekkel.
2. **CLI-ként**: `shiny-pancake run | summarize | serve | worker | eval | mcp-serve | ...`
3. **Könyvtárként / harnessként**: saját modell, saját toolok, saját ágensek és saját eval suite köthető be.

API kulcs nélkül is teljesen működik: egy determinisztikus offline provider hajtja meg a valódi
agent loopot, tool registryt, worker poolt és tracinget. Ha megvan az `ANTHROPIC_API_KEY`, Claude-ra
vált, és bármilyen OpenAI-kompatibilis végpontra (Ollama, vLLM, LM Studio, OpenAI) is ráköthető.

## Mit bizonyít a repo (LLM-integrációs szempontból)

| Terület | Mit mutat meg |
|---|---|
| Gyártófüggetlenség | Egy `LLMProvider` interfész, egységes (kanonikus) üzenetformátum; Claude, OpenAI-kompatibilis és offline provider |
| Claude korrekt használata | Prompt caching (statikus prefix + beszélgetés vége), szerveroldali refusal fallback, thinking blokkok változatlan visszaküldése (append-only history), `effort`, nincs `temperature` (API 2026-10-21), típusos hibakezelés |
| Model routing | Olcsó, gyors modell a summary workereknek, erős modell az orchestratornak és a végső szerkesztésnek |
| Multi-agent orchestration | Orchestrator–workers minta: a specialisták **toolként** érhetők el, így a delegálás ugyanazt a validációt, timeoutot, tracinget és párhuzamosságot kapja |
| Több summary worker | `asyncio.Queue` alapú worker pool: map-reduce fa-redukcióval, refine, és *ensemble* (több nézőpont, amit egy editor modell fésül össze) |
| Toolok és biztonság | Séma-validáció, timeout, kimenet-limit, jóváhagyási policy mellékhatásos toolokra, sandboxolt fájlelérés, untrusted tartalom keretezése, prompt injection jelzés |
| Külső rendszerek | `MCPBridge`: bármely MCP szerver toolja futásidőben regisztrálódik; `mcp-serve`: a hub maga is MCP szerver |
| Hub és skálázás | Tartós SQLite job queue (WAL, atomikus claim, crash recovery), horizontálisan skálázható worker konténerek, élő SSE eseményfolyam |
| Rugalmasság és költség | Korlátozott párhuzamosság, retry jitteres backoffal, vendor-fallback, lépés- és tokenkeretek, ágensenkénti/workerenkénti token- és költségszámítás |
| Megfigyelhetőség | Minden LLM hívás, tool hívás és pipeline lépés trace event: tárolva, élőben streamelve, JSONL-be exportálva |
| Kiértékelés | JSONL eval suite determinisztikus ellenőrzésekkel + opcionális LLM judge; CI-ban fut |

## Architektúra dióhéjban

```text
CLI / HTTP API + web UI / MCP szerver
                │
               Hub  (composition root: itt drótozódik össze minden)
        ┌───────┴─────────────────────────┐
  Agent team                       Summary orchestrator
  orchestrator                      chunker → worker pool (N db worker)
   ├─ researcher (fájlok, memória, MCP)   → fa-redukció → (ensemble) editor
   ├─ analyst   (kalkulátor, idő)
   └─ critic    (review, tool nélkül)
        │                                 │
   Tool registry (validáció, timeout, jóváhagyás, untrusted keretezés)
        │
   LLM providerek (Claude · OpenAI-kompatibilis · offline) + retry / concurrency / fallback
        │
   Tracer → SQLite store · JSONL · SSE a böngészőnek

Job queue (SQLite, WAL) ◄── API enqueue        Worker processzek / konténerek ──► claim · futtatás · eredmény
```

Részletek, szekvenciadiagramok és design döntések: [ARCHITECTURE.md](ARCHITECTURE.md).

## Az összefoglaló motor (a „több summary worker” rész)

1. **Chunking**: bekezdések → mondatok → kemény vágás, mondathatárra igazított átfedéssel.
2. **Map**: minden (profil × chunk) pár egy független job; a poolban az a worker veszi fel, amelyik éppen szabad.
3. **Reduce**: a részösszefoglalók csoportokban (a chunk-kereten belül) párhuzamosan redukálódnak, szintről szintre, amíg profilonként egy marad.
4. **Edit** (ensemble): az editor (fő modell) a profilokat (kulcspontok, teendők, kockázatok…) egy Markdown riporttá fésüli össze.

Hibatűrés: a sikertelen map job újra sorba kerül (másik worker is felveheti), a sikertelen reduce
nem dönti be a futást, csak egyszerűsített összefűzésre vált.

| Stratégia | Mikor jó |
|---|---|
| `stuff` | rövid szöveg, egy hívás |
| `map_reduce` | hosszú dokumentum, párhuzamos feldolgozás |
| `refine` | ahol a sorrend és a narratíva számít |
| `ensemble` | döntés-előkészítő anyagok, riportok, több nézőpontból |
| `auto` | méret és profilok alapján választ |

## Gyors kipróbálás

```bash
pip install -e ".[dev]"
```

```bash
shiny-pancake art
```

```bash
shiny-pancake run "Mennyi 1250 * 0.27 + 99? Ellenőrizd is." --offline --show-trace
```

```bash
shiny-pancake summarize workspace/notes/kickoff-meeting.md --strategy ensemble --offline
```

```bash
shiny-pancake serve --offline
```

Dockerben (hub + 2 worker konténer közös volume-on):

```bash
docker compose up --build
```

Teszt, lint, eval:

```bash
make lint test eval
```

## Demó forgatókönyv (kb. 5 perc, pl. interjúhoz)

1. **Pancake Hub UI** (`shiny-pancake serve --offline`): *Summarize* fül, „ai-integration-playbook.md”,
   `ensemble`. Élőben látszik, ahogy a worker-1..4 párhuzamosan dolgozik, a reduce szintek, majd az editor.
2. **Ask the team**: „What is 12 * (3 + 4)? Check it.” Az orchestrator az analystnek delegál
   (kalkulátor), majd a critic ellenőrzi. A trace-ben minden lépés, token és becsült költség látszik.
3. **Biztonság**: „Read the file ../../etc/passwd …” A sandbox blokkolja; ezt az eval suite is ellenőrzi.
4. **Skálázás**: `docker compose up --scale worker=4`. A jobokat külön konténerek dolgozzák fel (a `worker` oszlop mutatja a hostnevet).
5. **Harness**: `examples/custom_harness.py`: saját tool (`@function_tool`), saját specialista, provider-wrapper, eval.
6. **MCP**: `shiny-pancake mcp-serve`. Claude Desktop vagy Claude Code toolként hívhatja az ágenscsapatot.
7. **Claude**: `ANTHROPIC_API_KEY` beállítva ugyanez valódi modellel; a worker Haiku, az editor Sonnet. A trace-ben látszik a cache találat és a költség.

## Fontos design döntések röviden

* **Ágensek mint toolok**: nincs külön delegálási protokoll, minden ugyanazon a validált, tracelt úton megy.
* **Least privilege**: a critic-nek nincs toolja, az analyst nem olvas fájlt, csak a researcher látja az MCP toolokat.
* **A hiba is adat**: a tool hibák `is_error` eredményként mennek vissza a modellnek, nem omlik össze a folyamat.
* **Append-only history**: a provider-specifikus blokkok (thinking, fallback) bájtra pontosan visszamennek, így a cache és a thinking érvényes marad.
* **Költségtudatosság**: model routing, prompt caching, step/token budget, költségbecslés futásonként, ágensenként és workerenként.
* **Determinisztikus tesztelés**: a teljes rendszer (API, SSE, workerek, MCP, CLI, evals) API kulcs és hálózat nélkül tesztelhető.

## Mi jöhet még

OpenTelemetry export · vektor memória (pgvector) · token streaming a UI-ba · interaktív jóváhagyási
felület mellékhatásos toolokhoz · Postgres/Redis queue több hostra · Message Batches a nem sürgős
összefoglaló jobokhoz.
