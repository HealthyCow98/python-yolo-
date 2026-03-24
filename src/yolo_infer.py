from ultralytics import YOLO

model = YOLO("models/best.pt")

def predict_crop(crop):
    result = model(crop)
    return result[0].probs.top1  # 0: normal, 1: defect