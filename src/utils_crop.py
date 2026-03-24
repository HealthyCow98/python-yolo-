
# bbox 정규화 함수 -> watcher.py ( 실제 탐지 ) , crop.py ( 교육용 crop 생성 ) 동일화 하기 위함
def normalize_bbox(x, y, w, h, img_w, img_h):
    cx = x + w // 2

    # 바늘 구멍 위치 (대략 하단 70~80%)
    hole_y = y + int(h * 0.75)

    FIX_W = max(120, int(w * 1.3))
    FIX_H = max(260, int(h * 1.3))

    x1 = int(cx - FIX_W / 2)
    x2 = int(cx + FIX_W / 2)

    # 구멍 중심 기준으로 위아래 균형
    y1 = int(hole_y - FIX_H / 2)
    y2 = int(hole_y + FIX_H / 2)

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_w, x2)
    y2 = min(img_h, y2)

    return x1, y1, x2, y2