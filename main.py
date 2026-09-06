import cv2
import numpy as np
from typing import Optional, Tuple

blur_size = (13, 13)
canny_thresh_1 = 50
canny_thresh_2 = 150
hough_rho = 1
hough_theta = np.pi / 180
hough_threshold = 40
hough_min_line_length = 40
hough_max_line_gap = 100
min_abs_slope = 0.4

Line = Tuple[int, int, int, int]

capture = cv2.VideoCapture("dataset.mp4")
if not capture.isOpened():
    raise RuntimeError("Не удалось открыть видео")

success, frame = capture.read()
if success:
    cv2.imwrite("reference_frame.png", frame)
    print("reference_frame.png сохранен!")
else:
    print("Не удалось сохранить кадр")

mask = cv2.imread("roi_mask.png", cv2.IMREAD_GRAYSCALE)
if mask is None:
    raise RuntimeError("Не удалось открыть ROI-маску")

_, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

height, width = frame.shape[:2]
if mask.shape != (height, width):
    raise RuntimeError(f"Размер маски {mask.shape} не совпадает с размером кадра {(height, width)}")

capture.set(cv2.CAP_PROP_POS_FRAMES, 0)


def approximate_lane_lines(
        lines: Optional[np.ndarray], width: int, height: int
) -> Tuple[Optional[Line], Optional[Line]]:
    if lines is None:
        return None, None

    left_lines = []
    right_lines = []

    for line in lines:
        x1, y1, x2, y2 = line

        if x2 == x1:
            continue

        slope = (y2 - y1) / (x2 - x1)

        if abs(slope) < min_abs_slope:
            continue

        if x1 < width * 0.4 or x1 > width * 0.6:
            continue

        intercept = y1 - slope * x1
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        if slope < 0 and x1 < width / 2:
            left_lines.append((slope, intercept, length))
        elif slope > 0 and x1 > width / 2:
            right_lines.append((slope, intercept, length))

    def get_average_line(lines_group):
        if not lines_group:
            return None

        total_length = sum(l[2] for l in lines_group)
        k_avg = sum(l[0] * l[2] for l in lines_group) / total_length
        b_avg = sum(l[1] * l[2] for l in lines_group) / total_length

        y_bottom = height
        y_top = int(height * 0.6)

        if k_avg == 0:
            return None

        x_bottom = int((y_bottom - b_avg) / k_avg)
        x_top = int((y_top - b_avg) / k_avg)

        if x_bottom < -width or x_bottom > width * 2:
            return None
        if x_top < -width or x_top > width * 2:
            return None

        return (x_bottom, y_bottom, x_top, y_top)

    left_line = get_average_line(left_lines)
    right_line = get_average_line(right_lines)

    return left_line, right_line


try:
    fps = capture.get(cv2.CAP_PROP_FPS)
    delay_ms = max(1, round(1000 / fps))

    while True:
        success, frame = capture.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        adjusted = cv2.convertScaleAbs(gray, alpha=1.5, beta=-50)
        blurred = cv2.GaussianBlur(adjusted, blur_size, 0)
        edges = cv2.Canny(blurred, canny_thresh_1, canny_thresh_2)

# Можно еще по цвету полос отличить (из-за всяких теней деревьев или машин по яркости и тп труднее иногда)

        roi_edges = cv2.bitwise_and(edges, mask)

        lines = cv2.HoughLinesP(
            roi_edges,
            rho=hough_rho,
            theta=hough_theta,
            threshold=hough_threshold,
            minLineLength=hough_min_line_length,
            maxLineGap=hough_max_line_gap,
        )

        left_line, right_line = approximate_lane_lines(lines, width, height)

        debug_frame = np.zeros_like(frame)

        if left_line is not None:
            x1, y1, x2, y2 = left_line
            cv2.line(debug_frame, (x1, y1), (x2, y2), (0, 0, 255), 5)

        if right_line is not None:
            x1, y1, x2, y2 = right_line
            cv2.line(debug_frame, (x1, y1), (x2, y2), (0, 255, 0), 5)

        if left_line is not None and right_line is not None:
            center_x_bottom = int((left_line[0] + right_line[0]) / 2)
            center_x_top = int((left_line[2] + right_line[2]) / 2)
            cv2.line(debug_frame, (center_x_bottom, height), (center_x_top, int(height * 0.6)), (255, 0, 0), 5)

        cv2.imshow("Lane Lines", debug_frame)

        # cv2.imshow("Original", frame)
        # cv2.imshow("Adjusted", adjusted)
        # cv2.imshow("Edges", edges)
        # cv2.imshow("ROI Edges", roi_edges)

        if cv2.waitKey(delay_ms) & 0xFF == ord("q"):
            break

finally:
    capture.release()
    cv2.destroyAllWindows()