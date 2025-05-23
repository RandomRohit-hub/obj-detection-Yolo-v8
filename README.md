
---

# YOLOv8 Object Detection

This repository demonstrates object detection using the YOLOv8 model. It includes a Python script that leverages a pre-trained YOLOv8 model in ONNX format to detect objects in images.([Medium][1])

## Table of Contents

* [Installation](#installation)
* [Usage](#usage)
* [Dependencies](#dependencies)
* [Model](#model)
* [License](#license)

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/RandomRohit-hub/obj-detection-Yolo-v8.git
   cd obj-detection-Yolo-v8
   ```



2. **Create and activate a virtual environment (optional but recommended):**

   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Unix or MacOS:
   source venv/bin/activate
   ```



3. **Install the required dependencies:**

   ```bash
   pip install -r requirements.txt
   ```



## Usage

To run object detection on an image using the provided `detection.py` script:

```bash
python detection.py --image path_to_your_image.jpg
```



Replace `path_to_your_image.jpg` with the path to the image you want to process.

### Command-line Arguments

* `--image`: Path to the input image.
* `--model`: Path to the YOLOv8 ONNX model file (default: `yolov8n.onnx`).
* `--classes`: Path to the file containing class names (default: `class.names`).([Medium][1])

Example:

```bash
python detection.py --image sample.jpg --model yolov8n.onnx --classes class.names
```



## Dependencies

The project relies on the following Python libraries:([GitHub][2])

* `opencv-python`
* `numpy`

These are specified in the `requirements.txt` file.

## Model

The project uses the YOLOv8 model in ONNX format (`yolov8n.onnx`). Ensure that this file is present in the project directory. If you need to obtain or convert YOLOv8 models to ONNX format, refer to the [Ultralytics documentation](https://docs.ultralytics.com/usage/cli/) for guidance.([viso.ai][3], [Ultralytics Docs][4])



