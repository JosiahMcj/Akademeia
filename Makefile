PY := .venv/bin/python
setup:
	python3 -m venv .venv && .venv/bin/pip install -q -r requirements.txt
test:
	$(PY) -m pytest tests/ -q
