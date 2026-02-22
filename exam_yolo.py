from ultralytics import YOLO
import torch

class YoloMobileDetector:
    def __init__(self, model_path="best.pt"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(model_path)
        self.model.to(self.device)

    def detect_mobile(self, frame):
        results = self.model(frame, verbose=False)[0]

        max_conf = 0.0

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            class_name = self.model.names[cls_id]

            if "mobile" in class_name.lower() or "phone" in class_name.lower():
                max_conf = max(max_conf, conf)

        return max_conf