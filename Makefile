#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = copy-paste-aug
PYTHON_VERSION = 3.11
PYTHON_INTERPRETER = python
UV ?= uv

# Dataset roots. Override these from the command line when your paths differ.
# Example:
#   make eltseg-train-simple COCO_ROOT=/data/coco2017 SUBSETS_ROOT=/data/processed
COCO_ROOT ?= data.nosync/raw/coco2017
SUBSETS_ROOT ?= data.nosync/processed
SIMPLE_CP_DATA ?= $(SUBSETS_ROOT)/coco2017_simple_cp_seed42_sub50
PCT_CP_DATA ?= $(SUBSETS_ROOT)/coco2017_harmonized_pct_seed42_sub50
LBM_CP_DATA ?= $(SUBSETS_ROOT)/coco2017_harmonized_lbm_seed42_sub50

WANDB_MODE ?= offline
WANDB_PROJECT ?= $(PROJECT_NAME)
WANDB_ENTITY ?=
RUNS_ROOT ?= runs

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	$(UV) sync


## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	$(UV) run ruff format --check
	$(UV) run ruff check


## Format source code with ruff
.PHONY: format
format:
	$(UV) run ruff check --fix
	$(UV) run ruff format


## Run tests
.PHONY: test
test:
	$(UV) run python -m pytest tests


## Set up Python interpreter environment
.PHONY: create-env
create-env:
	$(UV) venv --python $(PYTHON_VERSION)
	@echo ">>> New uv virtual environment created. Activate with:"
	@echo ">>> Windows: .\\.venv\\Scripts\\activate"
	@echo ">>> Unix/macOS: source ./.venv/bin/activate"


#################################################################################
# PROJECT RUNS                                                                  #
#################################################################################

## Run training
.PHONY: train
train:
	$(UV) run python -m cpa.training $(ARGS)


# -----------------------------------------------------------------------------
# Shared copy-paste dataset shortcuts
# -----------------------------------------------------------------------------

## Print the resolved dataset paths used by the training targets
.PHONY: print-datasets
print-datasets:
	@echo "COCO_ROOT      = $(COCO_ROOT)"
	@echo "SIMPLE_CP_DATA = $(SIMPLE_CP_DATA)"
	@echo "PCT_CP_DATA    = $(PCT_CP_DATA)"
	@echo "LBM_CP_DATA    = $(LBM_CP_DATA)"


# -----------------------------------------------------------------------------
# Tiny RF-DETR segmentation runs
# -----------------------------------------------------------------------------

TINYRFD_DATA ?= $(SIMPLE_CP_DATA)
TINYRFD_VAL_DATA ?= $(COCO_ROOT)
TINYRFD_VARIANT ?= n
TINYRFD_TRAIN_IMAGE_SET ?= augmented
TINYRFD_ARGS ?=

## Train tiny RF-DETR segmentation on a premade COCO2017 dataset
.PHONY: tinyrfdeter-train
tinyrfdeter-train:
	WANDB_MODE=$(WANDB_MODE) $(UV) run python -m cpa.tinyrfdeter.lightning \
		--data-root "$(TINYRFD_DATA)" $(if $(TINYRFD_VAL_DATA),--val-data-root "$(TINYRFD_VAL_DATA)",) \
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

## Train tiny RF-DETR on the Simple Copy-Paste subset and validate on original COCO2017
.PHONY: tinyrfdeter-train-simple
tinyrfdeter-train-simple:
	$(MAKE) tinyrfdeter-train \
		TINYRFD_DATA="$(SIMPLE_CP_DATA)" \
		TINYRFD_VAL_DATA="$(COCO_ROOT)"

## Train tiny RF-DETR on the PCTNet-harmonized subset and validate on original COCO2017
.PHONY: tinyrfdeter-train-pct
tinyrfdeter-train-pct:
	$(MAKE) tinyrfdeter-train \
		TINYRFD_DATA="$(PCT_CP_DATA)" \
		TINYRFD_VAL_DATA="$(COCO_ROOT)"

## Train tiny RF-DETR on the LBM-harmonized subset and validate on original COCO2017
.PHONY: tinyrfdeter-train-lbm
tinyrfdeter-train-lbm:
	$(MAKE) tinyrfdeter-train \
		TINYRFD_DATA="$(LBM_CP_DATA)" \
		TINYRFD_VAL_DATA="$(COCO_ROOT)"

.PHONY: tinyrfdeter-smoke
tinyrfdeter-smoke:
	WANDB_MODE=offline $(UV) run python -m cpa.tinyrfdeter.lightning \
		--data-root "$(TINYRFD_DATA)" $(if $(TINYRFD_VAL_DATA),--val-data-root "$(TINYRFD_VAL_DATA)",) \
		--train-image-set $(TINYRFD_TRAIN_IMAGE_SET) \
		--variant n \
		--image-size 96 \
		--batch-size 1 \
		--num-workers 0 \
		--train-subset-percent 0.1 \
		--val-subset-percent 0.1 \
		--wandb \
		--fast-dev-run


# -----------------------------------------------------------------------------
# ELT instance-segmentation runs
# -----------------------------------------------------------------------------
# The ELT trainer trains on one premade copy-paste subset and validates on the
# untouched COCO2017 root. By default it uses lists/train_augmented.txt, so only
# generated copy-paste images are used for training.
#
# Override examples:
#   make eltseg-train ELT_DATA=/data/coco2017_harmonized_lbm_seed42_sub50 COCO_ROOT=/data/coco2017
#   make eltseg-train-simple ELT_DEVICE=cuda WANDB_MODE=online
#   make eltseg-train-pct ELT_TRAIN_IMAGE_SET=all
#
# ELT_ENTRYPOINT defaults to scripts/train_elt_instance_seg.py. If you put the
# trainer at project root, call with ELT_ENTRYPOINT=train_elt_instance_seg.py.
# If you package it as a module, call with ELT_ENTRYPOINT="-m cpa.elt_instance_seg.training".

ELT_ENTRYPOINT ?= cpa/elt_train_inst_seg.py
ELT_DATA ?= $(SIMPLE_CP_DATA)
ELT_DATA_NAME ?= $(notdir $(patsubst %/,%,$(ELT_DATA)))
ELT_COCO_ROOT ?= $(COCO_ROOT)
ELT_TRAIN_IMAGE_SET ?= augmented
ELT_TRAIN_IMAGE_DIR ?= $(ELT_DATA)/train2017
ELT_TRAIN_JSON ?= $(ELT_DATA)/annotations/instances_train2017.json
ELT_TRAIN_LIST ?= $(ELT_DATA)/lists/train_$(ELT_TRAIN_IMAGE_SET).txt
ELT_VAL_IMAGE_DIR ?= $(ELT_COCO_ROOT)/val2017
ELT_VAL_JSON ?= $(ELT_COCO_ROOT)/annotations/instances_val2017.json
ELT_OUTPUT_ROOT ?= $(RUNS_ROOT)/elt_instance_seg
ELT_OUTPUT_DIR ?= $(ELT_OUTPUT_ROOT)/$(ELT_DATA_NAME)_$(ELT_TRAIN_IMAGE_SET)
ELT_RUN_NAME ?= $(ELT_DATA_NAME)_$(ELT_TRAIN_IMAGE_SET)_val_original_coco2017
ELT_MODEL ?= maskrcnn_resnet50_fpn_v2
ELT_INIT ?= none
ELT_DEVICE ?= cuda
ELT_EPOCHS ?= 12
ELT_BATCH_SIZE ?= 2
ELT_NUM_WORKERS ?= 4
ELT_LR ?= 2.5e-4
ELT_WEIGHT_DECAY ?= 1e-4
ELT_MIN_LOOPS ?= 1
ELT_MAX_LOOPS ?= 4
ELT_EVAL_LOOPS ?= 1,2,4
ELT_WANDB_PROJECT ?= elt-coco-copy-paste
ELT_WANDB_LOG_IMAGES ?= 8
ELT_WANDB_TOP_K ?= 15
ELT_RESUME ?=
ELT_LIMIT_TRAIN_BATCHES ?= 0
ELT_LIMIT_VAL_BATCHES ?= 0
ELT_ARGS ?=

ELT_COMMON_ARGS = \
	--train-root "$(ELT_DATA)" \
	--train-image-dir "$(ELT_TRAIN_IMAGE_DIR)" \
	--train-json "$(ELT_TRAIN_JSON)" \
	--train-list "$(ELT_TRAIN_LIST)" \
	--train-only $(ELT_TRAIN_IMAGE_SET) \
	--val-root "$(ELT_COCO_ROOT)" \
	--val-image-dir "$(ELT_VAL_IMAGE_DIR)" \
	--val-json "$(ELT_VAL_JSON)" \
	--model $(ELT_MODEL) \
	--init $(ELT_INIT) \
	--device $(ELT_DEVICE) \
	--epochs $(ELT_EPOCHS) \
	--batch-size $(ELT_BATCH_SIZE) \
	--num-workers $(ELT_NUM_WORKERS) \
	--lr $(ELT_LR) \
	--weight-decay $(ELT_WEIGHT_DECAY) \
	--use-elt \
	--ilsd \
	--elt-min-loops $(ELT_MIN_LOOPS) \
	--elt-max-loops $(ELT_MAX_LOOPS) \
	--eval-loop-budgets $(ELT_EVAL_LOOPS) \
	--output-dir "$(ELT_OUTPUT_DIR)" \
	--wandb \
	--wandb-project $(ELT_WANDB_PROJECT) \
	$(if $(WANDB_ENTITY),--wandb-entity "$(WANDB_ENTITY)",) \
	--wandb-run-name "$(ELT_RUN_NAME)" \
	--wandb-tags elt,instance-segmentation,$(ELT_TRAIN_IMAGE_SET),original-coco2017-val,$(ELT_DATA_NAME) \
	--wandb-log-images $(ELT_WANDB_LOG_IMAGES) \
	--wandb-top-k $(ELT_WANDB_TOP_K) \
	$(if $(ELT_RESUME),--resume "$(ELT_RESUME)",) \
	$(if $(filter-out 0,$(ELT_LIMIT_TRAIN_BATCHES)),--limit-train-batches $(ELT_LIMIT_TRAIN_BATCHES),) \
	$(if $(filter-out 0,$(ELT_LIMIT_VAL_BATCHES)),--limit-val-batches $(ELT_LIMIT_VAL_BATCHES),) \
	$(ELT_ARGS)

## Check that the selected ELT training subset and original COCO validation paths exist
.PHONY: eltseg-check-data
eltseg-check-data:
	@test -d "$(ELT_DATA)" || (echo "Missing ELT_DATA: $(ELT_DATA)" >&2; exit 2)
	@test -d "$(ELT_TRAIN_IMAGE_DIR)" || (echo "Missing train images: $(ELT_TRAIN_IMAGE_DIR)" >&2; exit 2)
	@test -f "$(ELT_TRAIN_JSON)" || (echo "Missing train annotations: $(ELT_TRAIN_JSON)" >&2; exit 2)
	@test -f "$(ELT_TRAIN_LIST)" || (echo "Missing train list: $(ELT_TRAIN_LIST)" >&2; exit 2)
	@test -d "$(ELT_COCO_ROOT)" || (echo "Missing ELT_COCO_ROOT: $(ELT_COCO_ROOT)" >&2; exit 2)
	@test -d "$(ELT_VAL_IMAGE_DIR)" || (echo "Missing original COCO val images: $(ELT_VAL_IMAGE_DIR)" >&2; exit 2)
	@test -f "$(ELT_VAL_JSON)" || (echo "Missing original COCO val annotations: $(ELT_VAL_JSON)" >&2; exit 2)

## Train ELT instance segmentation on ELT_DATA and validate on original COCO2017
.PHONY: eltseg-train
eltseg-train: eltseg-check-data
	WANDB_MODE=$(WANDB_MODE) $(UV) run python $(ELT_ENTRYPOINT) \
		$(ELT_COMMON_ARGS)

## Train ELT on the Simple Copy-Paste subset and validate on original COCO2017
.PHONY: eltseg-train-simple
eltseg-train-simple:
	$(MAKE) eltseg-train \
		ELT_DATA="$(SIMPLE_CP_DATA)" \
		ELT_RUN_NAME=coco2017_simple_cp_seed42_sub50_augmented_val_original_coco2017 \
		ELT_OUTPUT_DIR="$(ELT_OUTPUT_ROOT)/coco2017_simple_cp_seed42_sub50_augmented"

## Train ELT on the PCTNet-harmonized subset and validate on original COCO2017
.PHONY: eltseg-train-pct
eltseg-train-pct:
	$(MAKE) eltseg-train \
		ELT_DATA="$(PCT_CP_DATA)" \
		ELT_RUN_NAME=coco2017_harmonized_pct_seed42_sub50_augmented_val_original_coco2017 \
		ELT_OUTPUT_DIR="$(ELT_OUTPUT_ROOT)/coco2017_harmonized_pct_seed42_sub50_augmented"

## Train ELT on the LBM-harmonized subset and validate on original COCO2017
.PHONY: eltseg-train-lbm
eltseg-train-lbm:
	$(MAKE) eltseg-train \
		ELT_DATA="$(LBM_CP_DATA)" \
		ELT_RUN_NAME=coco2017_harmonized_lbm_seed42_sub50_augmented_val_original_coco2017 \
		ELT_OUTPUT_DIR="$(ELT_OUTPUT_ROOT)/coco2017_harmonized_lbm_seed42_sub50_augmented"

## Train ELT sequentially on Simple, PCTNet, and LBM subsets
.PHONY: eltseg-train-all-subsets
eltseg-train-all-subsets:
	$(MAKE) eltseg-train-simple
	$(MAKE) eltseg-train-pct
	$(MAKE) eltseg-train-lbm

## Offline smoke-test for the selected ELT_DATA subset
.PHONY: eltseg-smoke
eltseg-smoke:
	$(MAKE) eltseg-train \
		WANDB_MODE=offline \
		ELT_EPOCHS=1 \
		ELT_BATCH_SIZE=1 \
		ELT_NUM_WORKERS=0 \
		ELT_WANDB_LOG_IMAGES=2 \
		ELT_LIMIT_TRAIN_BATCHES=2 \
		ELT_LIMIT_VAL_BATCHES=2 \
		ELT_ARGS="--no-progress"

.PHONY: eltseg-smoke-simple
eltseg-smoke-simple:
	$(MAKE) eltseg-smoke ELT_DATA="$(SIMPLE_CP_DATA)" ELT_RUN_NAME=smoke_coco2017_simple_cp_seed42_sub50

.PHONY: eltseg-smoke-pct
eltseg-smoke-pct:
	$(MAKE) eltseg-smoke ELT_DATA="$(PCT_CP_DATA)" ELT_RUN_NAME=smoke_coco2017_harmonized_pct_seed42_sub50

.PHONY: eltseg-smoke-lbm
eltseg-smoke-lbm:
	$(MAKE) eltseg-smoke ELT_DATA="$(LBM_CP_DATA)" ELT_RUN_NAME=smoke_coco2017_harmonized_lbm_seed42_sub50

## Evaluate an ELT checkpoint on original COCO2017 validation
.PHONY: eltseg-eval
eltseg-eval: eltseg-check-data
	@test -n "$(ELT_RESUME)" || (echo "ELT_RESUME is required. Example: make eltseg-eval ELT_RESUME=runs/elt_instance_seg/coco2017_simple_cp_seed42_sub50_augmented/best_segm_ap.pt" >&2; exit 2)
	WANDB_MODE=$(WANDB_MODE) $(UV) run python $(ELT_ENTRYPOINT) \
		$(ELT_COMMON_ARGS) \
		--eval-only

.PHONY: eltseg-eval-simple
eltseg-eval-simple:
	$(MAKE) eltseg-eval \
		ELT_DATA="$(SIMPLE_CP_DATA)" \
		ELT_RUN_NAME=eval_coco2017_simple_cp_seed42_sub50 \
		ELT_OUTPUT_DIR="$(ELT_OUTPUT_ROOT)/eval_coco2017_simple_cp_seed42_sub50"

.PHONY: eltseg-eval-pct
eltseg-eval-pct:
	$(MAKE) eltseg-eval \
		ELT_DATA="$(PCT_CP_DATA)" \
		ELT_RUN_NAME=eval_coco2017_harmonized_pct_seed42_sub50 \
		ELT_OUTPUT_DIR="$(ELT_OUTPUT_ROOT)/eval_coco2017_harmonized_pct_seed42_sub50"

.PHONY: eltseg-eval-lbm
eltseg-eval-lbm:
	$(MAKE) eltseg-eval \
		ELT_DATA="$(LBM_CP_DATA)" \
		ELT_RUN_NAME=eval_coco2017_harmonized_lbm_seed42_sub50 \
		ELT_OUTPUT_DIR="$(ELT_OUTPUT_ROOT)/eval_coco2017_harmonized_lbm_seed42_sub50"


.PHONY: download-coco
download-coco:
	@test -n "$(OUTPUT)" || (echo "OUTPUT is required. Usage: make download-coco OUTPUT=/path/to/file.zip" >&2; exit 2)
	curl -L -o $(OUTPUT) \
		https://www.kaggle.com/api/v1/datasets/download/awsaf49/coco-2017-dataset

.PHONY: download-coco-aria2
download-coco-aria2:
	@test -n "$(OUTPUT)" || (echo "OUTPUT is required. Usage: make download-coco-aria2 OUTPUT=/path/to/file.zip" >&2; exit 2)
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
	$(UV) run python -m cpa.training training.mode=eval $(ARGS)

## Generate a premade COCO2017 copy-paste dataset
.PHONY: premade-coco
premade-coco:
	$(UV) run python -m cpa.premade_datasets.coco2017 $(ARGS)

## Generate config dataclasses from YAML configs
.PHONY: gen-configs
gen-configs:
	$(UV) run python scripts/generate_configs.py

## Make dataset
.PHONY: data
data: requirements
	$(PYTHON_INTERPRETER) cpa/dataset.py


#################################################################################
# Self Documenting Commands                                                      #
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
