import os
import time
from ultralytics import YOLO
import cv2
import json
import shutil
import pymysql

INPUT_DIR = "../input"
OUTPUT_DIR = "../output"
KEEP_DIR = "../keepFiles"
AMB_DIR = "../ambiguous"

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(KEEP_DIR, exist_ok=True)
os.makedirs(AMB_DIR, exist_ok=True)

# YOLO 모델 로드
model = YOLO("../models/needle_detect_v1.pt")
cls_model = YOLO("../runs/classify/train7/weights/best.pt")

processed_files = set()

# DB 연결
conn = pymysql.connect(
    host="localhost",
    user="root",
    password="1234",
    database="inference_results",
    charset="utf8mb4"
)
cursor = conn.cursor()


def classify_crop(crop):
    result = cls_model(crop)[0]
    cls = int(result.probs.top1)
    conf = float(result.probs.top1conf)
    return cls, conf


def process_image(file_path):
    print(f"처리중: {file_path}")
    image = cv2.imread(file_path)

    if image is None:
        print("이미지 로드 실패")
        return

    base = os.path.basename(file_path)
    name, _ = os.path.splitext(base)

    results = model(image, conf=0.5, iou=0.3)[0]
    boxes = sorted(results.boxes, key=lambda b: b.xyxy[0][0])

    result_list = []
    defect_found = False

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])

        w = x2 - x1
        h = y2 - y1

        if w < 15 or h < 15:
            continue

        crop = image[y1:y2, x1:x2]

        cls, cls_conf = classify_crop(crop)

        if cls_conf < 0.7:
            cv2.imwrite(f"{AMB_DIR}/{name}_{len(result_list)}.jpg", crop)

        label_name = cls_model.names[cls]

        if label_name == "defect":
            label = "defect"
            color = (0, 0, 255)
            defect_found = True
        else:
            label = "normal"
            color = (0, 255, 0)

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        result_list.append({
            "bbox": [x1, y1, x2, y2],
            "confidence": conf,
            "class": label,
            "class_conf": cls_conf
        })

    final_result = "NG" if defect_found else "OK"

    if len(result_list) == 0:
        final_result = "NG"

    # DB 저장 (여기가 핵심)
    for idx, n in enumerate(result_list):
        x1, y1, x2, y2 = n["bbox"]

        cursor.execute("""
            INSERT INTO inference_results
            (file_name, work_num, infr_dy, batch_index,
             model_type, class_name, probability,
             x, y, width, height)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            name,
            name,  # work_num 없으면 일단 name 넣어
            time.strftime("%Y%m%d"),
            idx,
            "needle_cls",
            n["class"],
            n["class_conf"],
            x1,
            y1,
            x2 - x1,
            y2 - y1
        ))

    conn.commit()

    # 결과 텍스트
    cv2.putText(
        image,
        f"RESULT: {final_result}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 0, 255) if defect_found else (0, 255, 0),
        3
    )

    # JSON 저장
    with open(f"{OUTPUT_DIR}/{name}.json", "w") as f:
        json.dump({
            "result": final_result,
            "needle_count": len(result_list),
            "needles": result_list
        }, f, indent=2)

    cv2.imwrite(f"{OUTPUT_DIR}/{name}_result.jpg", image)

    print(f"검출 개수: {len(result_list)}")


def watch_folder():
    print("폴더 감시 시작")

    while True:
        files = os.listdir(INPUT_DIR)

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