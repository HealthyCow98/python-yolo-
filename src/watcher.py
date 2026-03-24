import os
import time
from ultralytics import YOLO
import cv2
import json
import shutil
from utils_crop import normalize_bbox
from detect_cv import detect_needles
from crop import save_crops
INPUT_DIR = "input"
OUTPUT_DIR = "output"
KEEP_DIR = "keepFiles"

os.makedirs(KEEP_DIR, exist_ok=True)
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

#  YOLO 모델 로드
model = YOLO("runs/classify/train3/weights/best.pt")

processed_files = set()


#  YOLO 예측 함수
#     # 26.03.20 변경 - 불량 검출이 너무 안되서 추가

# def predict_crop(crop):
#     result = model(crop)[0]
#     cls = int(result.probs.top1)         # 0: normal, 1: defect
#     conf = float(result.probs.top1conf)  # 확률
# 
#     if cls == 0 and conf < 0.8:  # normal인데 확신 낮으면
#         cls = 1  # 불량으로 간주
#         
#     return cls, conf

def predict_crop(crop):
    result = model(crop)[0]
    probs = result.probs.data.tolist()

    normal_prob = probs[0]
    defect_prob = probs[1]

    #  defect 확률 기준으로 판단
    if defect_prob > 0.2:
        cls = 1
        conf = defect_prob
    else:
        cls = 0
        conf = normal_prob

    return cls, conf

def process_image(file_path):
    print(f"처리중: {file_path}")

    needles, image = detect_needles(file_path)

## crop 생성
    base = os.path.basename(file_path)
    name, _ = os.path.splitext(base)

    save_crops(image, needles, name)

    if image is None:
        print("이미지 로드 실패")
        return

    result_list = []
    defect_count = 0

    for n in needles:
        x, y, w, h = n["x"], n["y"], n["w"], n["h"]

        #1 crop = image[y:y+h, x:x+w]
        #2 crop = image[y:int(y + h * 0.7), x:x + w]
        #3
        img_h, img_w = image.shape[:2]

        x1, y1, x2, y2 = normalize_bbox(x, y, w, h, img_w, img_h)

        crop = image[y1:y2, x1:x2]
        # crop 파일 ( 학습시킨 파일 그대로 읽기 위해 주석처리하였음 ) 흑백 보정 및 히스토그램 보정
        # crop = cv2.resize(crop, (224, 224))
        # crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        # crop = cv2.equalizeHist(crop)

        #  안전장치 (이거 중요)
        if crop is None or crop.size == 0:
            continue

        pred, conf = predict_crop(crop)

        label = "normal"
        color = (0, 255, 0)

        #  불량 판단 기준
        if pred == 1 and conf > 0.7:
            label = "defect"
            color = (0, 0, 255)
            defect_count += 1

        # bbox + 라벨 표시
        #cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            image,
            f"{label}:{conf:.2f}",
            (x1, y1-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

        result_list.append({
            "bbox": [x1, y1, x2, y2],
            "class": label,
            "confidence": conf
        })

    #  JSON 저장
    base = os.path.basename(file_path)
    name, _ = os.path.splitext(base)

    with open(f"{OUTPUT_DIR}/{name}.json", "w") as f:
        json.dump(result_list, f, indent=2)

    #  결과 이미지 저장
    cv2.imwrite(f"{OUTPUT_DIR}/{name}_result.jpg", image)

    print(f"바늘 개수: {len(needles)}")

    #  최종 판정
    if defect_count > 0:
        print(f" 불량 (불량 바늘 {defect_count}개)")
    else:
        print(" 정상")


def watch_folder():
    print("폴더 감시 시작")

    while True:
        files = os.listdir(INPUT_DIR)
        print("현재 파일:", files)

        for file in files:
            file_path = os.path.join(INPUT_DIR, file)

            if file not in processed_files and file.lower().endswith((".jpg", ".png", ".bmp")):
                process_image(file_path)
                move_to_keep(file_path)
                processed_files.add(file)

        time.sleep(2)


def move_to_keep(file_path):
    base = os.path.basename(file_path)
    dest_path = os.path.join(KEEP_DIR, base)

    shutil.move(file_path, dest_path)
    print(f"파일 이동 완료 → {dest_path}")


if __name__ == "__main__":
    watch_folder()
