.PHONY: setup test figures
PY := .venv/bin/python
setup:
	python3 -m venv .venv && .venv/bin/pip install -q -r requirements.txt
test:
	$(PY) -m pytest tests/ -q
figures:
	$(PY) scripts/make_figures.py && $(PY) scripts/make_length_figures.py
