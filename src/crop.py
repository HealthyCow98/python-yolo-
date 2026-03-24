import os
import cv2
from utils_crop import normalize_bbox

def save_crops(image, needles, base_name):
    os.makedirs("dataset/raw", exist_ok=True)
    print("켜졌따")
    img_h, img_w = image.shape[:2]

    for i, n in enumerate(needles):
        x, y, w, h = n["x"], n["y"], n["w"], n["h"]

        #1 crop = image[y:y+h, x:x+w]
        #2  crop = image[y:int(y + h * 0.7), x:x + w]
        #3
        x1, y1, x2, y2 = normalize_bbox(x, y, w, h, img_w, img_h)

        crop = image[y1:y2, x1:x2]

        filename = f"{base_name}_{i}.jpg"
        cv2.imwrite(f"dataset/raw/{filename}", crop)