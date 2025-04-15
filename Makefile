PYTHON_VERSION := 3.10.12

POETRY := PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring poetry

.PHONY: install install-dev clean test check-pyenv check-poetry check-nvcc


install: check-pyenv check-poetry check-nvcc
	pyenv install -s $(PYTHON_VERSION)
	pyenv local $(PYTHON_VERSION)
	$(POETRY) env use $$(pyenv which python)
	$(POETRY) install --only main

install-dev: install
	$(POETRY) install --only dev
	$(POETRY) run pre-commit install

clean: check-poetry
	$(POETRY) env remove --all
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -delete

test: install-dev
	$(POETRY) run pytest

check-pyenv:
	@if ! command -v pyenv --version >/dev/null 2>&1; then \
		echo "Error: pyenv is not installed. Please install it first:"; \
		exit 1; \
	fi

check-poetry:
	@if ! command -v poetry --version >/dev/null 2>&1; then \
		echo "Error: poetry is not installed. Please install it first:"; \
		exit 1; \
	fi

check-nvcc:
	@if ! command -v nvcc --version >/dev/null 2>&1; then \
		echo "Error: nvcc is not installed. Please install it first:"; \
		exit 1; \
	fi
