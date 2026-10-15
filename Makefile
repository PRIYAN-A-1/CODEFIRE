test:
	python -m pytest -q
smoke:
	python scripts/smoke_test.py
api:
	python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
ui:
	cd frontend && npm run dev
