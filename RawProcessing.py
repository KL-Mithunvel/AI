import cv2 as cv
import numpy as np
from pathlib import Path

# ---------- your existing function (unchanged) ----------
def highlight_checkerboard(img_bgr):
    h, w = img_bgr.shape[:2]

    # 1) Texture: gradient energy (checkerboard is very “edge-dense”)
    gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)
    gx = cv.Scharr(gray, cv.CV_32F, 1, 0)
    gy = cv.Scharr(gray, cv.CV_32F, 0, 1)
    grad_mag = cv.magnitude(gx, gy)

    # Local energy (box-filtered squared gradients)
    win = max(9, int(min(h, w) * 0.01) // 2 * 2 + 1)  # odd kernel ~1% of min dim
    energy = cv.boxFilter(grad_mag**2, ddepth=-1, ksize=(win, win))
    energy = cv.normalize(energy, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

    # 2) Threshold + morphology to consolidate the pattern
    _, th = cv.threshold(energy, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    mask = cv.morphologyEx(th, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_RECT,(15,15)))
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN,  cv.getStructuringElement(cv.MORPH_RECT,(5,5)))

    # 3) Keep big component(s) that touch the image border (checkerboard frame)
    num, labels, stats, _ = cv.connectedComponentsWithStats(mask, 8)
    keep = np.zeros_like(mask)
    area_min = 0.001 * h * w
    for i in range(1, num):
        x, y, ww, hh, area = stats[i]
        touches_border = (x == 0) or (y == 0) or (x + ww >= w - 1) or (y + hh >= h - 1)
        if area > area_min and touches_border:
            keep[labels == i] = 255

    # Clean edges a bit
    keep = cv.morphologyEx(keep, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_RECT,(31,31)))
    keep = cv.dilate(keep, cv.getStructuringElement(cv.MORPH_RECT,(9,9)), 1)

    # 4) Fit rotated rectangle and make overlay
    coords = np.column_stack(np.where(keep > 0))
    out = img_bgr.copy()
    if coords.size:
        rect = cv.minAreaRect(coords[:, ::-1].astype(np.float32))  # (center, (w,h), angle)
        box = cv.boxPoints(rect).astype(int)

        overlay = img_bgr.copy()
        cv.fillPoly(overlay, [box], (0, 255, 0))
        out = cv.addWeighted(overlay, 0.25, img_bgr, 0.75, 0)
        cv.polylines(out, [box], True, (0, 255, 0), 4)

    return out, keep, energy

# ---------- NEW: tile detector built on top of the checkerboard mask ----------
def highlight_tile(img_bgr, checker_mask, k=3):
    """
    Finds and highlights the tile (central terracotta patch) in blue.
    - Excludes checkerboard pixels using 'checker_mask'
    - Color clusters (Lab) -> chooses largest, rectangular, non-border region
    """
    h, w = img_bgr.shape[:2]

    # Exclude checkerboard region (dilate a bit to be safe)
    exclude = cv.dilate(checker_mask, cv.getStructuringElement(cv.MORPH_RECT, (21,21)), 1)

    # K-means in Lab color space (good for color separation)
    lab = cv.cvtColor(img_bgr, cv.COLOR_BGR2LAB)
    Z = lab.reshape((-1, 3)).astype(np.float32)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    _, labels, centers = cv.kmeans(Z, k, None, criteria, 3, cv.KMEANS_PP_CENTERS)
    labels = labels.reshape((h, w))

    best_score = -1
    best_contour = None

    for c in range(k):
        m = (labels == c).astype(np.uint8) * 255
        m[exclude > 0] = 0  # remove checkerboard zone

        # Clean up cluster mask
        m = cv.morphologyEx(m, cv.MORPH_OPEN,  cv.getStructuringElement(cv.MORPH_RECT, (7,7)))
        m = cv.morphologyEx(m, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_RECT, (31,31)))

        # Remove small / border-touching components
        num, lab2, stats, _ = cv.connectedComponentsWithStats(m, 8)
        keep = np.zeros_like(m)
        for i in range(1, num):
            x, y, ww, hh, area = stats[i]
            touches = (x == 0) or (y == 0) or (x + ww >= w-1) or (y + hh >= h-1)
            if (not touches) and area > 0.002 * h * w:
                keep[lab2 == i] = 255

        cnts, _ = cv.findContours(keep, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue

        # pick the biggest blob in this cluster
        c_big = max(cnts, key=cv.contourArea)
        area = cv.contourArea(c_big)
        rect = cv.minAreaRect(c_big)
        rect_area = rect[1][0] * rect[1][1]
        if rect_area <= 0:
            continue

        # Prefer large + rectangle-like blobs
        solidity = area / rect_area  # ~1 means very rectangle-like & filled
        score = area * solidity
        if score > best_score:
            best_score = score
            best_contour = c_big

    # Build overlay
    out = img_bgr.copy()
    tile_mask = np.zeros((h, w), np.uint8)
    if best_contour is not None:
        cv.drawContours(tile_mask, [best_contour], -1, 255, -1)
        box = cv.boxPoints(cv.minAreaRect(best_contour)).astype(int)

        overlay = out.copy()
        cv.fillPoly(overlay, [box], (255, 0, 0))         # blue fill
        out = cv.addWeighted(overlay, 0.25, out, 0.75, 0)
        cv.polylines(out, [box], True, (255, 0, 0), 4)   # blue border

    return out, tile_mask

# ---------- Convenience wrapper to highlight both ----------
def highlight_checkerboard_and_tile(img_bgr):
    out1, checker_mask, _ = highlight_checkerboard(img_bgr)
    out2, tile_mask = highlight_tile(out1, checker_mask)
    return out2, checker_mask, tile_mask


if __name__ == "__main__":
    img = cv.imread("DATASET/RAW DATA/A/20250912_083243.jpg")

    highlighted, checker_mask, tile_mask = highlight_checkerboard_and_tile(img)

    cv.imwrite("highlighted_both.png", highlighted)
    cv.imwrite("checker_mask.png", checker_mask)
    cv.imwrite("tile_mask.png", tile_mask)
