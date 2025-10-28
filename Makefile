# https://stackoverflow.com/questions/38878088/activate-anaconda-python-environment-from-makefile
# By default make uses sh to execute commands, and sh doesn't know `source`
SHELL=/bin/bash

install:
	pip install --upgrade pip
	@echo "installing all specified dependencies..."
	# NOTE: root package needs to be installed for CLI tests to work!
	@uv sync --all-groups

update:
	@echo "updating and pinning the dependencies specified in 'pyproject.toml':"
	@uv lock --upgrade

lock:
	@echo "locking the dependencies specified in 'pyproject.toml':"
	@uv lock


# when dependency resolving gets stuck:
force_update:
	@echo "force updating the requirements. removing lock file"
	@uv cache clean
	@rm -f uv.lock
	@echo "pinning the dependencies specified in 'pyproject.toml':"
	@uv sync --refresh

outdated:
	@uv pip list --outdated


hook:
	@uv run pre-commit install
	@uv run pre-commit run --all-files

hookup:
	@uv run pre-commit autoupdate

hook3:
	@uv run pre-commit clean

clean:
	rm -rf .pytest_cache .coverage coverage.xml tests/__pycache__ .mypyp_cache/ .tox

parallel-demo:  ## Run parallel coordinates demo
	uv run python scripts/parallel_demo.py

test:  ## Run all tests
	uv run pytest tests/ -v

test-parallel:  ## Test parallel coordinates module
	uv run pytest tests/test_parallelplot.py -v

parallel-dev: test-parallel  ## Full parallel plot development cycle

mypy:
	uv run mypy src/


.PHONY: clean test build docs parallel-demo test-parallel parallel-dev
