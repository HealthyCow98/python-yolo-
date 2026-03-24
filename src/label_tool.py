import cv2
import os

IMAGE_DIR = r"C:\Users\thrjs\OneDrive\Desktop\vision"

current_class = "good"

def draw_bbox(event, x, y, flags, param):
    global ix, iy, drawing, img, boxes

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            img_copy = img.copy()
            cv2.rectangle(img_copy, (ix, iy), (x, y), (0,255,0), 2)
            cv2.imshow("image", img_copy)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.rectangle(img, (ix, iy), (x, y), (0,255,0), 2)
        boxes.append((ix, iy, x, y, current_class))


def save_yolo_format(image_path, boxes):
    h, w, _ = img.shape
    txt_path = os.path.splitext(image_path)[0] + ".txt"

    with open(txt_path, "w") as f:
        for box in boxes:
            x1, y1, x2, y2, cls = box
            cx = ((x1 + x2) / 2) / w
            cy = ((y1 + y2) / 2) / h
            bw = abs(x2 - x1) / w
            bh = abs(y2 - y1) / h

            class_id = 0 if cls == "good" else 1
            f.write(f"{class_id} {cx} {cy} {bw} {bh}\n")


drawing = False
ix, iy = -1, -1

for file in os.listdir(IMAGE_DIR):
    if file.endswith(".bmp"):
        path = os.path.join(IMAGE_DIR, file)
        img = cv2.imread(path)
        boxes = []

        cv2.namedWindow("image")
        cv2.setMouseCallback("image", draw_bbox)

        while True:
            cv2.imshow("image", img)
            key = cv2.waitKey(1)

            if key == ord('g'):
                current_class = "good"
                print("class = good")

            elif key == ord('b'):
                current_class = "bad"
                print("class = bad")

            elif key == ord('s'):
                save_yolo_format(path, boxes)
                print("saved!")
                break

            elif key == 27:  # ESC
                break

        cv2.destroyAllWindows()