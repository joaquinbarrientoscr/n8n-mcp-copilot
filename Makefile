.PHONY: help setup db-up db-down ingest ingest-semantic reset ui eval eval-fast test clean

help:
	@echo "n8n + MCP Copilot — common tasks"
	@echo ""
	@echo "  make setup            Create venv and install deps"
	@echo "  make db-up            Start Postgres+pgvector (docker compose)"
	@echo "  make db-down          Stop the DB"
	@echo "  make ingest           Run ingestion with fixed_size chunking"
	@echo "  make ingest-semantic  Run ingestion with semantic chunking"
	@echo "  make reset            Wipe DB and re-ingest"
	@echo "  make ui               Launch Streamlit UI"
	@echo "  make eval             Run full eval (retrieval + LLM judge)"
	@echo "  make eval-fast        Run retrieval-only eval (no judge)"
	@echo "  make test             Run smoke tests"
	@echo "  make clean            Wipe venv and DB volume"

setup:
	python -m venv .venv
	./.venv/bin/pip install -U pip
	./.venv/bin/pip install -r requirements.txt
	@test -f .env || cp .env.example .env
	@echo ""
	@echo "✓ Setup complete. Edit .env with your API keys, then run 'make db-up'."

db-up:
	docker compose up -d
	@echo "✓ Postgres + pgvector running on localhost:5433"

db-down:
	docker compose down

ingest:
	./.venv/bin/python -m ingestion.ingest

ingest-semantic:
	./.venv/bin/python -m ingestion.ingest --strategy semantic

reset:
	./.venv/bin/python -m ingestion.ingest --reset

ui:
	./.venv/bin/streamlit run ui/app.py

eval:
	./.venv/bin/python -m eval.run_eval

eval-fast:
	./.venv/bin/python -m eval.run_eval --skip-judge

test:
	./.venv/bin/pytest tests/ -v

clean:
	docker compose down -v
	rm -rf .venv eval/runs
