import cv2
import os
import shutil
from pathlib import Path
import numpy as np

FOLDER1 = Path("DATASET/Camera/RAWDATA")
FOLDER2 = Path("DATASET/Camera/PROCESSED DATA")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

last_used_points = []
selected_point_idx = -1

def is_image_file(p: Path) -> bool:
    return p.suffix.lower() in IMAGE_EXTS

def rotate_image(img, angle):
    if angle == 0:
        return img
    elif angle == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        raise ValueError(f"Unsupported rotation angle: {angle}")

def perspective_crop(img, points):
    if len(points) != 4:
        raise ValueError("Need 4 points for perspective crop.")
    pts_src = np.array(points, dtype="float32")

    width = max(np.linalg.norm(pts_src[0] - pts_src[1]),
                np.linalg.norm(pts_src[2] - pts_src[3]))
    height = max(np.linalg.norm(pts_src[0] - pts_src[3]),
                 np.linalg.norm(pts_src[1] - pts_src[2]))

    pts_dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype="float32")

    matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)
    warped = cv2.warpPerspective(img, matrix, (int(width), int(height)))
    return warped

def purge_output_dir():
    if FOLDER2.exists():
        shutil.rmtree(FOLDER2)
    FOLDER2.mkdir(parents=True, exist_ok=True)

def get_user_points(img):
    global last_used_points, selected_point_idx

    h, w = img.shape[:2]
    screen_res = (1920, 1080)
    max_w, max_h = screen_res[0] - 100, screen_res[1] - 150  # margins

    scale = 1.0
    if h > max_h or w > max_w:
        scale = min(max_w / w, max_h / h)

    disp_w, disp_h = int(w * scale), int(h * scale)
    display_img = cv2.resize(img, (disp_w, disp_h)) if scale != 1 else img.copy()

    # Reset points if outside new image
    clicked_points = []
    for pt in last_used_points:
        if 0 <= pt[0] < w and 0 <= pt[1] < h:
            clicked_points.append(pt)
    selected_point_idx = -1

    def inside(p1, p2, r=20):
        return np.linalg.norm(np.array(p1) - np.array(p2)) < r

    def mouse_callback(event, x, y, flags, param):
        global selected_point_idx
        nonlocal clicked_points

        if event == cv2.EVENT_LBUTTONDOWN:
            for i, pt in enumerate(clicked_points):
                px, py = int(pt[0] * scale), int(pt[1] * scale)
                if inside((x, y), (px, py)):
                    selected_point_idx = i
                    return
            if len(clicked_points) < 4:
                clicked_points.append((int(x / scale), int(y / scale)))
                print(f"Point {len(clicked_points)}: {clicked_points[-1]}")

        elif event == cv2.EVENT_MOUSEMOVE and selected_point_idx != -1:
            move_x = int(x / scale)
            move_y = int(y / scale)
            clicked_points[selected_point_idx] = (move_x, move_y)

        elif event == cv2.EVENT_LBUTTONUP:
            selected_point_idx = -1

    cv2.namedWindow("Select 4 corners (Enter to preview, ESC to skip)", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Select 4 corners (Enter to preview, ESC to skip)", mouse_callback)

    while True:
        preview = display_img.copy()
        for pt in clicked_points:
            px, py = int(pt[0] * scale), int(pt[1] * scale)
            cv2.circle(preview, (px, py), 15, (0, 0, 255), -1)
        cv2.imshow("Select 4 corners (Enter to preview, ESC to skip)", preview)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            cv2.destroyWindow("Select 4 corners (Enter to preview, ESC to skip)")
            return None
        elif key == 13 and len(clicked_points) == 4:
            cv2.destroyWindow("Select 4 corners (Enter to preview, ESC to skip)")
            last_used_points = clicked_points.copy()
            return clicked_points

def confirm_and_save(img, points, target_folder, stem, suffix):
    try:
        cropped = perspective_crop(img, points)
    except Exception as e:
        print(f"[ERROR] Cropping failed: {e}")
        return False

    while True:
        cv2.imshow("Cropped Preview", cropped)
        print("Press 's' to save, 'b' to go back and reselect, or ESC to skip.")
        key = cv2.waitKey(0) & 0xFF

        if key == ord('s'):
            for angle in [0, 90, 180, 270]:
                rotated = rotate_image(cropped, angle)
                out_name = f"{stem}_r{angle}{suffix}"
                out_path = target_folder / out_name
                cv2.imwrite(str(out_path), rotated)
                print(f"[SAVED] {out_path}")
            cv2.destroyWindow("Cropped Preview")
            return True

        elif key == ord('b'):
            cv2.destroyWindow("Cropped Preview")
            return None

        elif key == 27:
            print("Skipped after crop.")
            cv2.destroyWindow("Cropped Preview")
            return False

def process_all_images():
    for root, _, files in os.walk(FOLDER1):
        rel_path = Path(root).relative_to(FOLDER1)
        target_folder = FOLDER2 / rel_path
        target_folder.mkdir(parents=True, exist_ok=True)

        for file in files:
            in_path = Path(root) / file
            stem = in_path.stem
            suffix = in_path.suffix.lower()

            if not is_image_file(in_path):
                continue

            img = cv2.imread(str(in_path))
            if img is None:
                print(f"[WARN] Skipping unreadable image: {in_path}")
                continue

            print(f"\n=== {in_path} ===")

            while True:
                user_points = get_user_points(img)
                if user_points is None:
                    print("Skipped.")
                    break

                result = confirm_and_save(img, user_points, target_folder, stem, suffix)

                if result is True:
                    break
                elif result is False:
                    break
                else:
                    print("Redoing point selection...")

if __name__ == "__main__":
    print("[START] Image Crop Tool — Ratio Fixed, Dots Adjustable, Safe Reuse]")
    if not FOLDER1.exists():
        raise FileNotFoundError(f"Source folder not found: {FOLDER1}")
    purge_output_dir()
    process_all_images()
    print("[DONE]")
