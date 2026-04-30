# ===========================================================================
# Makefile — NAO YOLO Camera Controller
# Run all targets from the root of the repository.
# ===========================================================================

# --- Paths ------------------------------------------------------------------
WEBOTS_CONTROLLER := $(WEBOTS_HOME)/Contents/MacOS/webots-controller
MODELS_DIR        := src/models
SCRIPT            := src/nao_cam.py
ROBOT_NAME        := NAO
PORT              ?= 1234

# YOLO model URLs
WEIGHTS_URL := https://pjreddie.com/media/files/yolov3.weights
CFG_URL     := https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg
NAMES_URL   := https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names

# Model file paths
WEIGHTS := $(MODELS_DIR)/yolov3.weights
CFG     := $(MODELS_DIR)/yolov3.cfg
NAMES   := $(MODELS_DIR)/coco.names

# Ollama settings
OLLAMA_MODEL ?= llama3.2-vision

# ---------------------------------------------------------------------------
.PHONY: all run models install pull-model serve clean help

# Default target
all: help

# ---------------------------------------------------------------------------
## run         — download models (if needed) then launch the NAO controller
run: models
	@echo ">>> Starting NAO YOLO controller (robot=$(ROBOT_NAME), port=$(PORT))…"
	$(WEBOTS_CONTROLLER) --robot-name=$(ROBOT_NAME) --port=$(PORT) $(SCRIPT)

# ---------------------------------------------------------------------------
## models      — download YOLO model files into src/models/ (skips if present)
models: $(WEIGHTS) $(CFG) $(NAMES)

$(MODELS_DIR):
	mkdir -p $(MODELS_DIR)

$(WEIGHTS): | $(MODELS_DIR)
	@echo ">>> Downloading yolov3.weights (~237 MB)…"
	curl -L --progress-bar -o $@ $(WEIGHTS_URL)

$(CFG): | $(MODELS_DIR)
	@echo ">>> Downloading yolov3.cfg…"
	curl -L --silent -o $@ $(CFG_URL)

$(NAMES): | $(MODELS_DIR)
	@echo ">>> Downloading coco.names…"
	curl -L --silent -o $@ $(NAMES_URL)

# ---------------------------------------------------------------------------
## install     — install Python dependencies from requirements.txt
install:
	@echo ">>> Installing Python dependencies…"
	pip install -r requirements.txt

# ---------------------------------------------------------------------------
## pull-model  — download the Llama 3 model via Ollama (~5 GB for vision)
pull-model:
	@echo ">>> Pulling $(OLLAMA_MODEL) via Ollama…"
	ollama pull $(OLLAMA_MODEL)

# ---------------------------------------------------------------------------
## serve       — start the Ollama server in the background
serve:
	@echo ">>> Starting Ollama server…"
	ollama serve

# ---------------------------------------------------------------------------
## clean       — remove downloaded model files (keeps src/models/ directory)
clean:
	@echo ">>> Removing downloaded model files…"
	rm -f $(WEIGHTS) $(CFG) $(NAMES)

# ---------------------------------------------------------------------------
## help        — show available targets (default)
help:
	@echo ""
	@echo "  NAO YOLO Camera — available targets:"
	@echo ""
	@grep -E '^## ' Makefile | sed 's/## /    make /'
	@echo ""
	@echo "  Typical first-time setup:"
	@echo "    make install   # install Python deps"
	@echo "    make run       # downloads models then launches Webots controller"
	@echo "    make run PORT=1235 # run on a specific port"
	@echo ""
