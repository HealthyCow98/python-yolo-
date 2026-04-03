import os
import cv2
import shutil

# ======================
# 설정
# ======================
# SRC_DIR = "../dataset/classify/train/tmp"
SRC_DIR = "../ambiguous"
NORMAL_DIR = "../dataset/classify/train/normal"
DEFECT_DIR = "../dataset/classify/train/defect"
AMBIGUOUS_DIR = "../dataset/classify/train/ambiguous"

os.makedirs(NORMAL_DIR, exist_ok=True)
os.makedirs(DEFECT_DIR, exist_ok=True)
os.makedirs(AMBIGUOUS_DIR, exist_ok=True)

files = [f for f in os.listdir(SRC_DIR) if f.lower().endswith((".jpg", ".png", ".bmp"))]

print(f"총 {len(files)}개 이미지")

idx = 0
cv2.namedWindow("Labeling Tool", cv2.WINDOW_NORMAL)



while idx < len(files):
    file = files[idx]
    path = os.path.join(SRC_DIR, file)

    img = cv2.imread(path)
    history = []
    if img is None:
        print(f"이미지 로드 실패: {file}")
        idx += 1
        continue

    # 이미지 표시
    display = img.copy()
    cv2.putText(display, f"{file}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.imshow("Labeling Tool", display)

    key = cv2.waitKey(0) & 0xFF

    # A → normal
    if key == ord('a'):
        shutil.move(path, os.path.join(NORMAL_DIR, file))
        history.append((file, NORMAL_DIR))
        print(f"[NORMAL] {file}")
        idx += 1

    # D → defect
    elif key == ord('d'):
        shutil.move(path, os.path.join(DEFECT_DIR, file))
        history.append((file, DEFECT_DIR))
        print(f"[DEFECT] {file}")
        idx += 1

    # F → ambiguous
    elif key == ord('f'):
        shutil.move(path, os.path.join(AMBIGUOUS_DIR, file))
        history.append((file, AMBIGUOUS_DIR))
        print(f"[AMBIGUOUS] {file}")
        idx += 1

    # S → skip
    elif key == ord('s'):
        print(f"[SKIP] {file}")
        history.append((file, NORMAL_DIR))
        idx += 1

    # Q → 종료
    elif key == ord('q'):
        print("종료")
        break

    # 뒤로가기 (옵션)
    elif key == ord('z'):
        if history:
            last_file, last_dir = history.pop()

            src = os.path.join(last_dir, last_file)
            dst = os.path.join(SRC_DIR, last_file)

            shutil.move(src, dst)

            idx = max(0, idx - 1)
            print(f"[UNDO] {last_file}")

cv2.destroyAllWindows()