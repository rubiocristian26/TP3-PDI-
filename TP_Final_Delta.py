import cv2
import numpy as np

# --- Configuración ---
INPUT_VIDEO = "ruta_1.mp4"       # Cambiar a "ruta_2.mp4" cuando quieras
OUTPUT_VIDEO = "resultado_carriles_full.mp4"

# Parámetros de detección
ROI_OFFSET_TOP = 5              # Margen extra para bajar el horizonte si es necesario
CANNY_THRESHOLDS = (50, 150)     # Umbrales para bordes
HOUGH_PARAMS = {
    "rho": 1,
    "theta": np.pi / 180,
    "threshold": 30,             # Mínimo de votos para aceptar una línea
    "minLineLength": 40,         # Longitud mínima
    "maxLineGap": 100            # Hueco máximo para unir segmentos
}

def region_of_interest(img):
    """
    Define una máscara trapezoidal para enfocarse solo en la carretera.
    """
    height, width = img.shape[:2]
    
    # Definimos los vértices del trapecio
    # (Ajustados para que cubran bien el carril en tus videos)
    polygons = np.array([
        [(0, height),                   # Esquina inferior izquierda
         (450, 320),                    # Punto superior izquierdo (horizonte)
         (520, 320),                    # Punto superior derecho (horizonte)
         (width, height)]               # Esquina inferior derecha
    ])
    
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def make_coordinates(image, line_parameters):
    """
    Convierte (pendiente, intersección) en coordenadas (x1, y1, x2, y2)
    para dibujar una línea que vaya desde el fondo hasta el horizonte.
    """
    if line_parameters is None:
        return None
        
    slope, intercept = line_parameters
    
    # y1: La parte inferior de la imagen (donde está el capó)
    y1 = image.shape[0] 
    
    # y2: Hasta dónde queremos que llegue la línea (el horizonte)
    # Aquí usamos 6/10 de la altura (aprox el centro vertical)
    y2 = int(y1 * (6/10)) + ROI_OFFSET_TOP
    
    # Calculamos x usando: x = (y - b) / m
    # Protegemos contra división por cero por si acaso
    if slope == 0: 
        slope = 0.001
        
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    
    return np.array([x1, y1, x2, y2])

def average_slope_intercept(image, lines):
    """
    Calcula la línea promedio para la izquierda y la derecha.
    Devuelve las coordenadas de las dos líneas principales.
    """
    left_fit = []
    right_fit = []
    
    if lines is None:
        return None, None
        
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        
        # Ajustamos un polinomio de grado 1 (una recta y=mx+b)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        
        # Filtramos por pendiente para diferenciar izquierda de derecha
        # Pendiente negativa = Carril Izquierdo (en coordenadas de imagen y crece hacia abajo)
        # Pendiente positiva = Carril Derecho
        # Umbral 0.4 para ignorar líneas horizontales ruidosas
        if slope < -0.4:
            left_fit.append((slope, intercept))
        elif slope > 0.4:
            right_fit.append((slope, intercept))
            
    # Hacemos el promedio de todas las líneas encontradas
    left_fit_average = np.average(left_fit, axis=0) if left_fit else None
    right_fit_average = np.average(right_fit, axis=0) if right_fit else None
    
    # Convertimos pendiente/intersección a coordenadas pixel
    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)
    
    return left_line, right_line

def display_lines(image, lines):
    """ Dibuja las líneas sobre una imagen negra """
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            if line is not None:
                x1, y1, x2, y2 = line
                # Dibujamos en AZUL (BGR: 255, 0, 0) con grosor 10
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 8)
    return line_image

# --- Ejecución Principal ---
def procesar_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    
    # Obtener propiedades para guardar el video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # --- CORRECCIÓN DE VELOCIDAD ---
    # Calculamos cuánto tiempo (ms) debe durar cada frame.
    # Ejemplo: Si el video es 30 FPS -> 1000 / 30 = 33 ms de espera.
    if fps > 0:
        delay_ms = int(1000 / fps)
    else:
        delay_ms = 30 # Valor por defecto seguro si no se detectan FPS
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    print(f"Procesando {input_path} a {fps} FPS (Delay: {delay_ms}ms)...")

    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        
        # 1. Copia del frame para trabajar
        lane_image = np.copy(frame)
        
        # 2. Canny (Bordes)
        gray = cv2.cvtColor(lane_image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        canny_image = cv2.Canny(blur, CANNY_THRESHOLDS[0], CANNY_THRESHOLDS[1])
        
        # 3. ROI (Región de Interés)
        cropped_image = region_of_interest(canny_image)
        
        # 4. Hough Transform (Detectar segmentos)
        lines = cv2.HoughLinesP(cropped_image, 
                                HOUGH_PARAMS["rho"], 
                                HOUGH_PARAMS["theta"], 
                                HOUGH_PARAMS["threshold"], 
                                np.array([]), 
                                minLineLength=HOUGH_PARAMS["minLineLength"], 
                                maxLineGap=HOUGH_PARAMS["maxLineGap"])
        
        # 5. Promedio y Extrapolación (EL PASO CLAVE PARA TU PROBLEMA)
        left_line, right_line = average_slope_intercept(lane_image, lines)
        
        # 6. Dibujar y Superponer
        line_image = display_lines(lane_image, [left_line, right_line])
        combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
        
        # Mostrar y guardar
        cv2.imshow('Resultado', combo_image)
        out.write(combo_image)
        
        # --- MODIFICADO: Usamos el delay calculado en lugar de '1' ---
        if cv2.waitKey(delay_ms) & 0xFF == ord('q'):
            break
            
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Video guardado exitosamente en: {output_path}")

# Ejecutar
if __name__ == "__main__":
    procesar_video(INPUT_VIDEO, OUTPUT_VIDEO)