This project implements an autonomous task-planning system for a simulated NAO robot in Webots. It uses **YOLOv8** for real-time object detection and a local **Llama 3.2 Vision** model (via Ollama) to analyze the scene and generate multi-step action plans.

---

## 🚀 Quick Start (Scratch Setup)

Follow these steps to get the environment running on your local machine.

### 1. Prerequisites
Ensure you have the following installed:
*   **Webots**: [Download here](https://cyberbotics.com/) (The robot simulator).
*   **Python 3.10+**: Core language for the controller.
*   **Ollama**: [Download here](https://ollama.com/) (For running the LLM locally).

### 2. Clone & Install Dependencies
```bash
git clone <your-repo-url>
cd llm-robot-cs424
pip install -r requirements.txt
```

### 3. Environment Configuration
The project uses a `.env` file to manage paths and model settings. 
1. Copy the example file: `cp .env.example .env`
2. Open `.env` and ensure the following variables are set:

```env
# Path to your Webots installation (REQUIRED)
WEBOTS_HOME=/Applications/Webots.app

# Ollama settings (Local LLM)
OLLAMA_MODEL=llama3.2-vision
OLLAMA_HOST=http://localhost:11434
```

### 4. Setup the LLM (Ollama)
The robot's "brain" runs locally. You must pull the vision model before running:
```bash
# In a new terminal:
ollama serve

# In your main terminal:
make pull-model
```

### 5. Launch the Simulation
The `make run` command will automatically download the YOLO weights (if missing) and start the controller.
```bash
make run
```

---

## 🛠 Project Components

### **YOLO Object Detection**
The robot uses YOLOv8 to identify objects in its field of view. 
*   **Logic**: Located in `src/yolo_detection.py`.
*   **Data**: The model identifies 80 classes (from the COCO dataset) and estimates distance based on perceived object height.

### **Scene State Extractor**
Every few frames, the robot "packages" its entire world-view into a JSON payload. This includes:
*   **Visuals**: YOLO detections with distance and horizontal angles.
*   **Proprioception**: 25 joint angles (arms, legs, head).
*   **Spatial**: GPS coordinates and IMU orientation (Roll/Pitch/Yaw).
*   **Sensors**: Sonar (proximity) and touch bumpers.

### **Llama 3 Task Planner**
When you give the robot a goal (e.g., *"Find the bottle and move toward it"*), the system:
1. Sends the **Scene JSON** and a **Camera Snapshot** to Llama 3.2 Vision.
2. The LLM reasons about the scene and returns a JSON action plan.
3. The robot executes the plan step-by-step.

---

## ⌨️ Useful Commands

| Command | Description |
| :--- | :--- |
| `make run` | Starts the Webots controller and YOLO loop. |
| `make models` | Downloads only the YOLOv3 configuration and weights. |
| `make pull-model` | Downloads the 5GB Llama 3.2 Vision model to your machine. |
| `make clean` | Removes YOLO weights and temporary snapshots. |

---

## ❓ Troubleshooting
*   **"No such file or directory" for webots-controller**: Ensure `WEBOTS_HOME` in your `.env` points to the actual folder where Webots is installed.
*   **Ollama connection refused**: Make sure the Ollama application is open and you have run `ollama serve`.
*   **Laggy Performance**: Running YOLO and a Vision LLM locally is resource-intensive. If the simulation is too slow, try closing other heavy applications.
