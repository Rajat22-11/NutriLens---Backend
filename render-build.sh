#!/usr/bin/env bash
# Render build step for the NutriLens API.
set -o errexit
pip install --upgrade pip
pip install -r requirements.txt
# yolov5 imports a few helpers from `ultralytics`; install it without its heavy
# dependency tree (polars, full opencv, ...) since everything it needs is above.
pip install --no-deps "ultralytics==8.3.253" "ultralytics-thop==2.2.1"
python -c "import sys; sys.path.append('yolov5'); from models.common import DetectMultiBackend; print('yolov5 import OK')"
