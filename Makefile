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

TINYRFD_DATA ?= data.nosync/processed/coco2017_simple_cp_seed42_sub50
TINYRFD_VAL_DATA ?=
TINYRFD_VARIANT ?= n
TINYRFD_TRAIN_IMAGE_SET ?= augmented
TINYRFD_ARGS ?=
WANDB_MODE ?= offline

## Train tiny RF-DETR segmentation on a premade COCO2017 dataset
.PHONY: tinyrfdeter-train
tinyrfdeter-train:
	WANDB_MODE=$(WANDB_MODE) uv run python -m cpa.tinyrfdeter.lightning \
		--data-root $(TINYRFD_DATA) $(if $(TINYRFD_VAL_DATA),--val-data-root $(TINYRFD_VAL_DATA),) \
		--train-image-set $(TINYRFD_TRAIN_IMAGE_SET) \
		--variant $(TINYRFD_VARIANT) \
		--wandb \
		$(TINYRFD_ARGS)

.PHONY: tinyrfdeter-train-n
tinyrfdeter-train-n:
	$(MAKE) tinyrfdeter-train TINYRFD_VARIANT=n

.PHONY: tinyrfdeter-train-s
tinyrfdeter-train-s:
	$(MAKE) tinyrfdeter-train TINYRFD_VARIANT=s

.PHONY: tinyrfdeter-train-m
tinyrfdeter-train-m:
	$(MAKE) tinyrfdeter-train TINYRFD_VARIANT=m

.PHONY: tinyrfdeter-smoke
tinyrfdeter-smoke:
	WANDB_MODE=offline uv run python -m cpa.tinyrfdeter.lightning \
		--data-root $(TINYRFD_DATA) $(if $(TINYRFD_VAL_DATA),--val-data-root $(TINYRFD_VAL_DATA),) \
		--train-image-set $(TINYRFD_TRAIN_IMAGE_SET) \
		--variant n \
		--image-size 96 \
		--batch-size 1 \
		--num-workers 0 \
		--train-subset-percent 0.1 \
		--val-subset-percent 0.1 \
		--wandb \
		--fast-dev-run

.PHONY: download-coco
download-coco:
ifeq ($(OUTPUT),)
	$(error OUTPUT is required. Usage: make download-coco OUTPUT=/path/to/file.zip)
endif
	curl -L -o $(OUTPUT) \
		https://www.kaggle.com/api/v1/datasets/download/awsaf49/coco-2017-dataset

.PHONY: download-coco-aria2
download-coco-aria2:
ifeq ($(OUTPUT),)
	$(error OUTPUT is required. Usage: make download-coco OUTPUT=/path/to/file.zip)
endif
	aria2c \
		--max-connection-per-server=16 \
		--split=16 \
		--min-split-size=1M \
		--continue \
		--console-log-level=error \
		-o $(OUTPUT) \
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
