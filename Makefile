.PHONY: install test lint format eval demo serve worker docker docker-up docker-down clean

install:            ## editable install with dev extras
	pip install -e ".[dev]"

test:               ## unit + integration tests (offline, no API key)
	pytest -q

lint:               ## ruff lint + format check
	ruff check src tests examples
	ruff format --check src tests examples

format:
	ruff format src tests examples
	ruff check --fix src tests examples

eval:               ## offline eval suite (what CI runs)
	shiny-pancake eval evals/smoke.jsonl --offline --json eval-report.json

demo:               ## a quick tour in the terminal
	shiny-pancake art
	shiny-pancake run "What is 12 * (3 + 4)? Check it." --offline --show-trace
	shiny-pancake summarize workspace/ai-integration-playbook.md --strategy ensemble --offline

serve:              ## API + web UI on http://127.0.0.1:8000
	SP_OFFLINE_LATENCY=0.15 shiny-pancake serve

worker:
	shiny-pancake worker --concurrency 2

docker:
	docker build -t shiny-pancake:latest .

docker-up:          ## hub + 2 scaled workers
	docker compose up --build

docker-down:
	docker compose down

clean:
	rm -rf data traces .pytest_cache .ruff_cache eval-report.json
