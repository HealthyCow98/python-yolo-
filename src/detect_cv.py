import cv2
import numpy as np
import os
import csv
from datetime import datetime


# ============================================================
# 설정값
# ============================================================
INPUT_DIR = "data/incoming"
OUTPUT_DIR = "data/results"
REPORT_DIR = "data/reports"

# 바늘 탐지 파라미터
MIN_NEEDLE_AREA = 2000        # 바늘 최소 면적 (픽셀)
MIN_NEEDLE_WIDTH = 20         # 바늘 최소 너비 (픽셀) - 빗살 구조물 오탐 방지
MAX_SINGLE_WIDTH = 120         # 단일 바늘 최대 너비 - 이보다 넓으면 병합으로 판단
MORPH_KERNEL_SIZE = 5          # 모폴로지 커널 크기
MORPH_ITERATIONS = 2           # 모폴로지 반복 횟수

# CLAHE 대비 향상 파라미터
CLAHE_CLIP_LIMIT = 4.0
CLAHE_TILE_SIZE = (8, 8)


def split_merged_contour(thresh_roi, x_offset, y_offset, h):
    """
    수직 프로젝션으로 병합된 바늘 컨투어를 개별 바늘로 분리한다.
    thresh_roi: 병합된 영역의 이진화 이미지
    반환: 분리된 바늘들의 (x, y, w, h) 리스트
    """
    # 수직 프로젝션: 각 열의 흰색 픽셀 수
    proj = np.sum(thresh_roi, axis=0) / 255
    roi_w = thresh_roi.shape[1]

    # 골짜기(valley) 찾기 - 프로젝션이 낮은 지점이 바늘 경계
    # 최소값의 3배 이하인 지점을 경계로 판단
    threshold = max(np.min(proj) * 3, np.max(proj) * 0.15)
    is_valley = proj < threshold

    # 연속된 골짜기 영역의 중앙점을 분할점으로 사용
    split_points = [0]
    in_valley = False
    valley_start = 0

    for i in range(roi_w):
        if is_valley[i] and not in_valley:
            valley_start = i
            in_valley = True
        elif not is_valley[i] and in_valley:
            mid = (valley_start + i) // 2
            if mid > 30 and mid < roi_w - 30:  # 가장자리 무시
                split_points.append(mid)
            in_valley = False

    split_points.append(roi_w)

    # 분할된 영역을 개별 바늘로 변환
    result = []
    for i in range(len(split_points) - 1):
        sx = split_points[i]
        ex = split_points[i + 1]
        w = ex - sx
        if w > 50:  # 너무 작은 영역 무시
            result.append({
                "x": x_offset + sx,
                "y": y_offset,
                "w": w,
                "h": h,
            })

    return result


def detect_needles(image_path):
    """
    이미지에서 바늘을 탐지하고 각 바늘의 위치/크기를 반환한다.
    CLAHE 대비 향상 + OTSU 자동 이진화로 이미지 밝기에 무관하게 동작.
    병합된 바늘은 수직 프로젝션으로 자동 분리.
    """
    img_color = cv2.imread(image_path)
    if img_color is None:
        print(f"[오류] 이미지를 불러올 수 없습니다: {image_path}")
        return [], None

    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    # 1. CLAHE로 대비 향상
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT,
                            tileGridSize=CLAHE_TILE_SIZE)
    enhanced = clahe.apply(img_gray)

    # 2. OTSU 자동 이진화
    # _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 2. 이진화 개선 ( OTSU + adaptive 결합 )
    # 2-1 OTSU (전체 기준)
    _, thresh = cv2.threshold(
        enhanced,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # 2-2 adaptive (지역 기준 - 조명 대응)
    # thresh2 = cv2.adaptiveThreshold(
    #     enhanced,
    #     255,
    #     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #     cv2.THRESH_BINARY,
    #     11,  # 블록 크기 (홀수)
    #     2  # 보정값
    # )

    # 2-3 결합 (합침)
    #thresh = cv2.bitwise_or(thresh1, thresh2)
    # thresh = cv2.adaptiveThreshold(
    #     enhanced,
    #     255,
    #     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #     #cv2.THRESH_BINARY,
    #     cv2.THRESH_BINARY_INV,  # 중요 (반전)
    #     15,  # block size (11 → 15로 키움)
    #     3  # C값 (2 → 3)
    # )
    # 3. 모폴로지 연산
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE))
    # thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=MORPH_ITERATIONS)
    # thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=MORPH_ITERATIONS)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    # thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    # 4. 컨투어 검출
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print("contours 개수:", len(contours))
    # 5. 바늘 필터링 + 병합 분리
    needles = []
    img_h = img_color.shape[0]
    MAX_NEEDLE_HEIGHT = 220

    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)

        # 상단 오탐 제거
        cy = y + h // 2  # 중심 좌표
        if cy < img_h * 0.3:
            continue

        # 높이 필터
        if h < 80:
            continue

        if h > MAX_NEEDLE_HEIGHT:
            y = y + (h - MAX_NEEDLE_HEIGHT)
            h = MAX_NEEDLE_HEIGHT

        if area < MIN_NEEDLE_AREA or w < MIN_NEEDLE_WIDTH:
            continue

        aspect_ratio = w / h if h > 0 else 999

        if w <= MAX_SINGLE_WIDTH and aspect_ratio < 1.0:
            # 단일 바늘
            needles.append({"x": x, "y": y, "w": w, "h": h, "area": area})

        elif w > MAX_SINGLE_WIDTH and aspect_ratio < 1.0 and h > 200 and area < 200000:
            # 병합된 바늘
            roi = thresh[y:y + h, x:x + w]
            split = split_merged_contour(roi, x, y, h)
            needles.extend(split)

    # x좌표 기준 정렬 (왼쪽→오른쪽)
    needles.sort(key=lambda n: n["x"])
    for i, needle in enumerate(needles):
        needle["index"] = i + 1

    return needles, img_color


def draw_results(image, needles, filename):
    """탐지 결과를 이미지 위에 시각화한다."""
    result = image.copy()

    for needle in needles:
        x, y, w, h = needle["x"], needle["y"], needle["w"], needle["h"]
        idx = needle["index"]

        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(result, f"#{idx}", (x, y - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    # 상단에 요약 정보 표시
    total = len(needles)
    summary = f"File: {filename} | Needle Count: {total}"
    cv2.rectangle(result, (0, 0), (len(summary) * 25, 80), (0, 0, 0), -1)
    cv2.putText(result, summary, (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 3)

    return result


def process_all_images():
    """data/incoming 폴더의 모든 이미지를 처리한다."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    image_files = sorted([
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith((".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"))
    ])

    if not image_files:
        print(f"[오류] {INPUT_DIR} 폴더에 이미지가 없습니다.")
        return

    print(f"{'='*60}")
    print(f" 바늘 객체 탐지 시스템")
    print(f" 처리 대상: {len(image_files)}장")
    print(f" 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    # CSV 리포트 준비
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(REPORT_DIR, f"report_{timestamp}.csv")
    csv_file = open(csv_path, "w", newline="", encoding="utf-8-sig")
    writer = csv.writer(csv_file)
    writer.writerow(["파일명", "바늘 총 개수"])

    all_results = []

    for file_idx, filename in enumerate(image_files, 1):
        image_path = os.path.join(INPUT_DIR, filename)
        print(f"[{file_idx}/{len(image_files)}] {filename} 처리 중...")

        needles, image = detect_needles(image_path)
        if image is None:
            continue

        result_image = draw_results(image, needles, filename)
        output_name = os.path.splitext(filename)[0] + "_result.png"
        output_path = os.path.join(OUTPUT_DIR, output_name)
        cv2.imwrite(output_path, result_image)

        total = len(needles)
        writer.writerow([filename, total])

        print(f"  → 바늘 {total}개 탐지")
        all_results.append({"filename": filename, "count": total})

    csv_file.close()

    # 최종 요약
    print(f"\n{'='*60}")
    print(f" 처리 완료 요약")
    print(f"{'='*60}")
    print(f"{'파일명':<50} {'바늘 수':>6}")
    print(f"{'-'*60}")
    for r in all_results:
        print(f"{r['filename']:<50} {r['count']:>6}")
    print(f"{'-'*60}")
    total_needles = sum(r["count"] for r in all_results)
    print(f"{'합계':<50} {total_needles:>6}")
    print(f"\n결과 이미지: {OUTPUT_DIR}/")
    print(f"CSV 리포트:  {csv_path}")
    print(f"완료 시간:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    process_all_images()
