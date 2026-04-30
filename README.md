# Project Overview
This project connects a simulated NAO robot in Webots to a Python controller that streams camera images and runs YOLOv3 object detection. Users can observe the camera feed and press `q` to quit.

## Working Components
- **Webots Robot Connection**: The controller successfully connects to the `NAO` robot in the simulator.
- **Camera Streaming**: The `CameraTop` device provides a real-time video stream.
- **Image Processing**: OpenCV is used to display the raw camera feed.

## Downloaded YOLO Files
The following files have been downloaded to `src/models/`:
- `yolov3.cfg`: YOLOv3 network configuration
- `yolov3.weights`: YOLOv3 pre-trained weights (~237 MB)
- `coco.names`: COCO dataset class names

## Running the Application

To run the controller with the YOLO model, use the following command:
```bash
$WEBOTS_HOME/Contents/MacOS/webots-controller --robot-name=NAO src/nao_cam.py
```

**Expected Output**:
1. A Webots window opens showing the simulated environment.
2. A separate window titled "NAO Camera" displays the video feed from the robot.
3. Press `q` in the camera window to close the application.

### Troubleshooting
If you encounter issues:
1. Ensure Webots is installed and `WEBOTS_HOME` is set correctly.
2. Verify the model files exist in `src/models/`.
3. Check that the camera name in the script matches the one in your Webots world file.

## Future Work
The next step is to integrate the `YOLODetector` class into `nao_cam.py` to enable real-time object detection on the camera feed.
