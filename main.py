import cv2

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

try:
    fps = capture.get(cv2.CAP_PROP_FPS)
    delay_ms = max(1, round(1000 / fps))

    while True:
        success, frame = capture.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        adjusted = cv2.convertScaleAbs(gray, alpha=1.5, beta=-50)
        blurred = cv2.GaussianBlur(adjusted, (13, 13), 0)
        edges = cv2.Canny(blurred, 50, 150)

# Можно еще по цвету полос отличить (из-за всяких теней деревьев или машин по яркости и тп труднее иногда)

        roi_edges = cv2.bitwise_and(edges, mask)

        cv2.imshow("Original", frame)
        cv2.imshow("Edges1", adjusted)
        cv2.imshow("Edges2", edges)
        cv2.imshow("ROI Edges", roi_edges)

        if cv2.waitKey(delay_ms) & 0xFF == ord("q"):
            break

finally:
    capture.release()
    cv2.destroyAllWindows()