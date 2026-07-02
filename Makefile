.PHONY: clean clean-build clean-pyc lint format typecheck artifact install
.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

clean: clean-build clean-pyc ## remove all build and Python artifacts

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

lint: ## check style with ruff
	uv run ruff check emb2heights scripts artifacts
	uv run ruff format --check emb2heights scripts artifacts

format: ## format code with ruff
	uv run ruff check --fix emb2heights scripts artifacts
	uv run ruff format emb2heights scripts artifacts

typecheck: ## type check with ty
	uv run ty check emb2heights

artifact: ## run example artifact script
	uv run python artifacts/example_artifact.py

artifact-light: ## run example artifact script in light mode
	uv run python artifacts/example_artifact.py --light

release: dist ## package and upload a release
	uv run --group release twine upload dist/*

dist: clean ## builds source and wheel package
	uv run --group release python -m build
	ls -l dist

install: ## sync the project environment with uv
	uv sync
