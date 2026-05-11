# Webots NAO + YOLOv8 + LLM 
WEBOTS_CONTROLLER := $(WEBOTS_HOME)/Contents/MacOS/webots-controller
MODELS_DIR  := src/models
ROBOT_NAME := NAO
PORT           ?= 1234
OLLAMA_MODEL   ?= llama3.2-vision
GEMINI_MODEL   ?= gemini-2.5-flash

.PHONY: all help simple models install pull-model clean

all: help

help:
	@echo ""
	@echo "  make simple   - run simple_controller.py (needs WEBOTS_HOME)"
	@echo "  make models   - mkdir for Ultralytics weights (yolov8n.pt auto-downloads)"
	@echo "  make install  - pip install -r requirements.txt"
	@echo "  make pull-model - ollama pull $(OLLAMA_MODEL)"
	@echo ""

models:
	mkdir -p $(MODELS_DIR)

simple: models
	$(WEBOTS_CONTROLLER) --robot-name=$(ROBOT_NAME) --port=$(PORT) src/simple_controller.py

install:
	pip install -r requirements.txt

pull-model:
	ollama pull $(OLLAMA_MODEL)

clean:
	rm -f $(MODELS_DIR)/yolov3.weights $(MODELS_DIR)/yolov3.cfg $(MODELS_DIR)/coco.names
