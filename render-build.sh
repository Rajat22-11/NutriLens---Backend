#!/usr/bin/env bash
# Render build step for the NutriLens API.
set -o errexit
pip install --upgrade pip
pip install -r requirements.txt
python -c "import sys; sys.path.append('yolov5'); from models.common import DetectMultiBackend; print('yolov5 import OK')"
