#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = copy-paste-aug
PYTHON_VERSION = 3.11
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	uv sync


## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	uv run ruff format --check
	uv run ruff check

## Format source code with ruff
.PHONY: format
format:
	uv run ruff check --fix
	uv run ruff format


## Run tests
.PHONY: test
test:
	uv run python -m pytest tests


## Set up Python interpreter environment
.PHONY: create-env
create-env:
	uv venv --python $(PYTHON_VERSION)
	@echo ">>> New uv virtual environment created. Activate with:"
	@echo ">>> Windows: .\\\\.venv\\\\Scripts\\\\activate"
	@echo ">>> Unix/macOS: source ./.venv/bin/activate"




#################################################################################
# PROJECT RUNS                                                                 #
#################################################################################

## Run training
.PHONY: train
train:
	uv run python -m cpa.training $(ARGS)

.PHONY: download-coco
download-coco:
ifeq ($(OUTPUT),)
	$(error OUTPUT is required. Usage: make download-coco OUTPUT=/path/to/file.zip)
endif
	curl -L -o $(OUTPUT) \
		https://www.kaggle.com/api/v1/datasets/download/awsaf49/coco-2017-dataset


## Run evaluation
.PHONY: eval
eval:
	uv run python -m cpa.training training.mode=eval $(ARGS)

## Generate a premade COCO2017 copy-paste dataset
.PHONY: premade-coco
premade-coco:
	uv run python -m cpa.premade_datasets.coco2017 $(ARGS)

## Generate config dataclasses from YAML configs
.PHONY: gen-configs
gen-configs:
	uv run python scripts/generate_configs.py

## Make dataset
.PHONY: data
data: requirements
	$(PYTHON_INTERPRETER) cpa/dataset.py


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
