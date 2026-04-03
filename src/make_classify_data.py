from ultralytics import YOLO
import cv2
import os

# ======================
# 설정
# ======================
DETECT_MODEL_PATH = "../models/needle_detect_v1.pt"

INPUT_DIR = "../input"
OUT_DIR = "../dataset/classify/train/tmp"

CONF_THRESHOLD = 0.5
PADDING = 5

# ======================
# 폴더 생성
# ======================
os.makedirs(OUT_DIR, exist_ok=True)

# ======================
# 모델 로드
# ======================
model = YOLO(DETECT_MODEL_PATH)

# ======================
# 실행
# ======================
for file in os.listdir(INPUT_DIR):
    if not file.lower().endswith((".bmp", ".jpg", ".png")):
        continue

    img_path = os.path.join(INPUT_DIR, file)
    img = cv2.imread(img_path)

    if img is None:
        print(f"이미지 로드 실패: {file}")
        continue

    results = model(img)[0]

    idx = 0

    for box in results.boxes:
        conf = float(box.conf[0])

        if conf < CONF_THRESHOLD:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # padding 적용
        x1 = max(0, x1 - PADDING)
        y1 = max(0, y1 - PADDING)
        x2 = min(img.shape[1], x2 + PADDING)
        y2 = min(img.shape[0], y2 + PADDING)

        # 너무 작은 박스 제거
        if (x2 - x1) < 10 or (y2 - y1) < 10:
            continue

        crop = img[y1:y2, x1:x2]

        save_name = f"{file.rsplit('.',1)[0]}_{idx}.jpg"
        save_path = os.path.join(OUT_DIR, save_name)

        cv2.imwrite(save_path, crop)

        idx += 1

    print(f"완료: {file} → {idx}개 crop")

print("🔥 전체 완료")