# YOLO Model Files

Place the following three files in this directory before running `nao_cam.py`:

| File | Size | Download |
|---|---|---|
| `yolov3.weights` | ~237 MB | https://pjreddie.com/media/files/yolov3.weights |
| `yolov3.cfg` | ~8 KB | https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg |
| `coco.names` | ~625 B | https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names |

## Quick download (from the repo root)

```bash
mkdir -p src/models && cd src/models
curl -O https://pjreddie.com/media/files/yolov3.weights
curl -O https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg
curl -O https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
```

> **Note:** `yolov3.weights` is 237 MB and is excluded from git via `.gitignore`.
