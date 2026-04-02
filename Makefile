.PHONY: help install test run clean

help:
	@echo "F1 Simulation - Python Version"
	@echo ""
	@echo "Available targets:"
	@echo "  make install  - Install dependencies"
	@echo "  make test     - Run tests"
	@echo "  make run      - Run simulation"
	@echo "  make clean    - Remove logs and cache"

install:
	pip install -r requirements.txt

test:
	python3 test.py

run:
	python3 main.py

clean:
	rm -rf logs/ __pycache__ *.pyc .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
