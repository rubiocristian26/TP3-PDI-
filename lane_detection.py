# lane_detection.py
import cv2
import numpy as np

# ---------- Configuración (ajustar si hace falta) ----------
INPUT_VIDEO = "ruta_2.mp4"      # o "ruta_2.mp4"
OUTPUT_VIDEO = "ruta_2_lanes.mp4"

# ROI (valores por defecto tomados y adaptables)
# Si prefieres usar los valores del ej_roi.py, están similares a éstos.
ROI_LEFT_TOP = (450, 320)
ROI_RIGHT_TOP = (520, 320)

# Canny
CANNY_LOW = 50
CANNY_HIGH = 150

# Gaussian blur
GAUSSIAN_KERNEL = (5, 5)

# HoughLinesP params
RHO = 1
THETA = np.pi / 180
HOUGH_THRESHOLD = 30   # mínimo número de votos
MIN_LINE_LENGTH = 40
MAX_LINE_GAP = 20

# Pendiente mínima absoluta para considerar como línea de carril
MIN_SLOPE = 0.4

# Grosor y color (BGR)
LANE_COLOR = (255, 0, 0)   # azul en BGR
LANE_THICKNESS = 4

# ------------------------------------------------------------

def region_of_interest(img):
    h, w = img.shape[:2]
    left_bottom = (0, h - 1)
    left_top = ROI_LEFT_TOP
    right_top = ROI_RIGHT_TOP
    right_bottom = (w - 1, h - 1)
    verts = np.array([left_bottom, left_top, right_top, right_bottom], dtype=np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [verts], 255)
    masked = cv2.bitwise_and(img, img, mask=mask)
    return masked, mask

def canny_edge(img_gray):
    blur = cv2.GaussianBlur(img_gray, GAUSSIAN_KERNEL, 0)
    edges = cv2.Canny(blur, CANNY_LOW, CANNY_HIGH)
    return edges

def slope_intercept_from_line(line):
    # line: [x1,y1,x2,y2]
    x1, y1, x2, y2 = line
    if x2 == x1:
        return None  # vertical
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return m, b

def separate_lines(lines):
    left_lines = []
    right_lines = []
    if lines is None:
        return left_lines, right_lines
    for l in lines:
        x1, y1, x2, y2 = l.reshape(4)
        si = slope_intercept_from_line((x1, y1, x2, y2))
        if si is None:
            continue
        m, b = si
        # Filter small slopes
        if abs(m) < MIN_SLOPE:
            continue
        # In image coords y grows downward: left lane typically has negative slope
        if m < 0:
            left_lines.append((m, b, x1, y1, x2, y2))
        else:
            right_lines.append((m, b, x1, y1, x2, y2))
    return left_lines, right_lines

def average_line(lines, y_min, y_max):
    """
    lines: list of tuples (m,b,...) -> hacemos promedio ponderado por longitud del segmento
    devolvemos (x1,y1,x2,y2) para dibujar entre y_max (arriba) y y_min (abajo)
    """
    if len(lines) == 0:
        return None
    ms = []
    bs = []
    weights = []
    for (m, b, x1, y1, x2, y2) in lines:
        length = np.hypot(x2 - x1, y2 - y1)
        ms.append(m * length)
        bs.append(b * length)
        weights.append(length)
    wm = sum(ms) / (sum(weights) + 1e-8)
    wb = sum(bs) / (sum(weights) + 1e-8)
    # calcular puntos extremos dados y = m*x + b -> x = (y - b)/m
    try:
        x_top = int((y_max - wb) / wm)
        x_bottom = int((y_min - wb) / wm)
    except Exception:
        return None
    return (x_top, int(y_max), x_bottom, int(y_min))

def draw_lane_lines(img, left_line, right_line):
    overlay = img.copy()
    if left_line is not None:
        x1, y1, x2, y2 = left_line
        cv2.line(overlay, (x1, y1), (x2, y2), LANE_COLOR, LANE_THICKNESS, lineType=cv2.LINE_AA)
    if right_line is not None:
        x1, y1, x2, y2 = right_line
        cv2.line(overlay, (x1, y1), (x2, y2), LANE_COLOR, LANE_THICKNESS, lineType=cv2.LINE_AA)
    # devolver imagen combinada (puedes variar alpha)
    return cv2.addWeighted(img, 0.8, overlay, 0.6, 0)

def process_video(input_path=INPUT_VIDEO, output_path=OUTPUT_VIDEO):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error: no se pudo abrir el video:", input_path)
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ROI
        roi_frame, mask = region_of_interest(frame)

        # Preprocesado
        gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        edges = canny_edge(gray)

        # Hough
        lines = cv2.HoughLinesP(edges, RHO, THETA, HOUGH_THRESHOLD,
                                minLineLength=MIN_LINE_LENGTH, maxLineGap=MAX_LINE_GAP)

        left_lines, right_lines = separate_lines(lines)

        # Definir y_min (arriba) y y_min (abajo) para las rectas (en coordenadas de imagen)
        y_min = frame.shape[0] - 1         # borde inferior
        y_max = int(frame.shape[0] * 0.55) # un punto por encima del centro (ajustable)

        left_avg = average_line(left_lines, y_min, y_max)
        right_avg = average_line(right_lines, y_min, y_max)

        # Dibujar sobre el frame original (no sobre ROI recortado)
        # NOTA: las coordenadas son compatibles porque usamos toda la imagen y la máscara recortó los píxeles.
        result = draw_lane_lines(frame, left_avg, right_avg)

        # Opcional: mostrar máscara y edges (descomentar si querés ver en tiempo real)
        # cv2.imshow("Mask", mask)
        # cv2.imshow("Edges", edges)
        cv2.imshow("Lanes", result)

        out.write(result)

        frame_idx += 1
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Procesado finalizado. Video guardado en:", output_path)

if __name__ == "__main__":
    process_video()