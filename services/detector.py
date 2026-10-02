"""YOLOv5 food detector (Indian food classes), loaded once and shared across requests."""
import logging
import os
import sys
import threading

import numpy as np

from config.config import BASE_DIR, settings
from services.nutrition import FOOD_CLASSES

logger = logging.getLogger(__name__)

YOLOV5_PATH = os.path.join(BASE_DIR, "yolov5")


class FoodDetector:
    def __init__(self, weights_path):
        self.weights_path = weights_path
        self.model = None
        self.error = None
        self._lock = threading.Lock()
        self._load_attempted = False

    @property
    def ready(self):
        return self.model is not None

    def load(self):
        with self._lock:
            if self._load_attempted:
                return self.model is not None
            self._load_attempted = True
            if settings.YOLO_MODE == "off":
                self.error = "disabled"
                return False
            try:
                import torch

                torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
                if YOLOV5_PATH not in sys.path:
                    sys.path.append(YOLOV5_PATH)
                from models.common import DetectMultiBackend  # yolov5

                logger.info("Loading YOLOv5 weights from %s", self.weights_path)
                self.model = DetectMultiBackend(self.weights_path, device=torch.device("cpu"), fp16=False)
                self.model.eval()
                logger.info("YOLOv5 model ready (%d classes)", len(self.model.names))
                return True
            except Exception as exc:
                self.error = str(exc)
                logger.exception("YOLOv5 model failed to load; Gemini fallback only")
                return False

    def detect(self, rgb_image):
        """Run detection on an RGB numpy image. Returns list of dicts with boxes in image pixels."""
        if not self._load_attempted:
            self.load()
        if self.model is None:
            return []

        import torch
        from utils.augmentations import letterbox  # yolov5
        from utils.general import non_max_suppression, scale_boxes  # yolov5

        h, w = rgb_image.shape[:2]
        size = settings.YOLO_IMG_SIZE
        stride = int(getattr(self.model, "stride", 32))
        padded = letterbox(rgb_image, size, stride=stride, auto=False)[0]
        tensor = torch.from_numpy(np.ascontiguousarray(padded.transpose(2, 0, 1))).float() / 255.0
        tensor = tensor.unsqueeze(0)

        with self._lock, torch.inference_mode():
            pred = self.model(tensor)
        det = non_max_suppression(pred, settings.YOLO_CONF_THRESHOLD, 0.45, max_det=20)[0]
        if det is None or not len(det):
            return []
        det[:, :4] = scale_boxes(tensor.shape[2:], det[:, :4], (h, w)).round()

        results = []
        for x1, y1, x2, y2, conf, cls in det.tolist():
            cls = int(cls)
            name, emoji = FOOD_CLASSES.get(cls, (f"Food {cls}", "🍽️"))
            results.append(
                {
                    "class_id": cls,
                    "name": name,
                    "emoji": emoji,
                    "confidence": round(float(conf), 3),
                    "box": [int(x1), int(y1), int(x2), int(y2)],
                }
            )
        return results


detector = FoodDetector(settings.YOLO_WEIGHTS)
