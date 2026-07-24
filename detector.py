import cv2
import numpy as np

GREEN_LOW = np.array([28, 30, 25], np.uint8)
GREEN_HIGH = np.array([100, 255, 255], np.uint8)


def detect_candidates(image, depth_m):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, GREEN_LOW, GREEN_HIGH)
    non_green = cv2.morphologyEx(
        cv2.bitwise_not(green),
        cv2.MORPH_OPEN,
        np.ones((3, 3), np.uint8),
    )
    contours, _ = cv2.findContours(
        non_green,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    output = []
    h_img, w_img = image.shape[:2]
    for contour in contours:
        area = cv2.contourArea(contour)
        if not 25 <= area <= 12000:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if x < 5 or y < 5 or x + w > w_img - 5 or y + h > h_img - 5:
            continue
        support = green[
            max(0, y - 4):min(h_img, y + h + 12),
            max(0, x - 12):min(w_img, x + w + 12),
        ]
        if support.size == 0 or np.count_nonzero(support) / support.size < 0.20:
            continue
        u, v = x + w // 2, y + h // 2
        roi = depth_m[
            max(0, v - 7):min(h_img, v + 8),
            max(0, u - 7):min(w_img, u + 8),
        ]
        valid = roi[np.isfinite(roi) & (roi >= 0.22) & (roi <= 1.20)]
        if valid.size < 20:
            continue
        median = float(np.median(valid))
        cluster = valid[np.abs(valid - median) < max(0.025, 0.07 * median)]
        if cluster.size < 12:
            continue
        output.append({
            'u': u,
            'v': v,
            'z': float(np.median(cluster)),
            'bbox': (x, y, w, h),
            'area': float(area),
        })
    return sorted(output, key=lambda item: item['z'])
