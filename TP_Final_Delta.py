import cv2
import numpy as np

# --- Configuración ---
VIDEO_ENTRADA = "ruta_2.mp4"      # Cambiar a "ruta_2.mp4" cuando quieras
VIDEO_SALIDA = "resultado_carriles_final.mp4"

# Parámetros de detección
roi_offset = 5                   # Margen extra para bajar el horizonte
canny_umbrales = (50, 150)       # Umbrales para bordes
parametros_hough = {
    "rho": 1,
    "theta": np.pi / 180,
    "threshold": 30,             # Mínimo de votos para aceptar una línea
    "minLineLength": 40,         # Longitud mínima
    "maxLineGap": 100            # Hueco máximo para unir segmentos
}

def region_interes(img):
    alto, ancho = img.shape[:2]
    
    # Definimos los vértices del trapecio
    poligonos = np.array([
        [(0, alto),                     # Esquina inferior izquierda
         (450, 320),                    # Punto superior izquierdo (horizonte)
         (520, 320),                    # Punto superior derecho (horizonte)
         (ancho, alto)]                 # Esquina inferior derecha
    ])
    
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, poligonos, 255)
    img_mask = cv2.bitwise_and(img, mask)
    return img_mask

def coordenadas(imagen, parametros_linea):
    """
    Convierte (pendiente, interseccion) en coordenadas (x1, y1, x2, y2).
    """
    if parametros_linea is None:return None
        
    pendiente, interseccion = parametros_linea
    
    # y1: La parte inferior de la imagen
    y1 = imagen.shape[0] 
    
    # y2: Hasta dónde queremos que llegue la línea (el horizonte)
    y2 = int(y1 * (6/10)) + roi_offset
    
    # Calculamos x usando: x = (y - b) / m
    if pendiente == 0: 
        pendiente = 0.001
        
    x1 = int((y1 - interseccion) / pendiente)
    x2 = int((y2 - interseccion) / pendiente)
    
    return np.array([x1, y1, x2, y2])

def average_slope_intercept(imagen, lineas):
    """
    Calcula la línea promedio para la izquierda y la derecha.
    """
    ajuste_izq = []
    ajuste_der = []
    
    if lineas is None:
        return None, None
        
    for linea in lineas:
        x1, y1, x2, y2 = linea.reshape(4)
        
        # Ajustamos un polinomio de grado 1 (recta y=mx+b)
        parametros = np.polyfit((x1, x2), (y1, y2), 1)
        pendiente = parametros[0]
        interseccion = parametros[1]
        
        # Filtramos por pendiente
        if pendiente < -0.4:
            ajuste_izq.append((pendiente, interseccion))
        elif pendiente > 0.4:
            ajuste_der.append((pendiente, interseccion))
            
    # Promedio de líneas encontradas
    promedio_izq = np.average(ajuste_izq, axis=0) if ajuste_izq else None
    promedio_der = np.average(ajuste_der, axis=0) if ajuste_der else None
    
    linea_izq = coordenadas(imagen, promedio_izq)
    linea_der = coordenadas(imagen, promedio_der)
    
    return linea_izq, linea_der

def mostrar_lineas(imagen, lineas):
    """ Dibuja las líneas sobre una imagen negra """
    imagen_lineas = np.zeros_like(imagen)
    if lineas is not None:
        for linea in lineas:
            if linea is not None:
                x1, y1, x2, y2 = linea
                # Dibujamos en AZUL (BGR: 255, 0, 0)
                cv2.line(imagen_lineas, (x1, y1), (x2, y2), (255, 0, 0), 8)
    return imagen_lineas

# --- Ejecución Principal ---
def procesar_video(ruta_entrada, ruta_salida):
    captura = cv2.VideoCapture(ruta_entrada)
    
    # Propiedades del video
    ancho = int(captura.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto = int(captura.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = captura.get(cv2.CAP_PROP_FPS)

    # --- CORRECCIÓN DE VELOCIDAD ---
    if fps > 0:
        retraso_ms = int(1000 / fps)
    else:
        retraso_ms = 30
    
    salida = cv2.VideoWriter(ruta_salida, cv2.VideoWriter_fourcc(*'mp4v'), fps, (ancho, alto))
    
    print(f"Procesando {ruta_entrada} a {fps} FPS (Retraso: {retraso_ms}ms)...")

    while(captura.isOpened()):
        check, frame = captura.read()
        if not check:   
            break
        
        # 1. Copia del frame
        imagen_carril = np.copy(frame)
        
        # 2. Canny (Bordes)
        grises = cv2.cvtColor(imagen_carril, cv2.COLOR_BGR2GRAY)
        desenfoque = cv2.GaussianBlur(grises, (5, 5), 0)
        imagen_canny = cv2.Canny(desenfoque, canny_umbrales[0], canny_umbrales[1])
        
        # 3. ROI (Región de Interés)
        img_recortada = region_interes(imagen_canny)
        
        # 4. Hough Transform
        lineas = cv2.HoughLinesP(img_recortada, 
                                parametros_hough["rho"], 
                                parametros_hough["theta"], 
                                parametros_hough["threshold"], 
                                np.array([]), 
                                minLineLength=parametros_hough["minLineLength"], 
                                maxLineGap=parametros_hough["maxLineGap"])
        
        # 5. Promedio y Extrapolación
        linea_izq, linea_der = average_slope_intercept(imagen_carril, lineas)
        
        # 6. Dibujar y Superponer
        imagen_lineas = mostrar_lineas(imagen_carril, [linea_izq, linea_der])
        imagen_combinada = cv2.addWeighted(imagen_carril, 0.8, imagen_lineas, 1, 1)
        
        # Mostrar y guardar
        cv2.imshow('Resultado', imagen_combinada)
        salida.write(imagen_combinada)
        
        if cv2.waitKey(retraso_ms) & 0xFF == ord('q'):
            break
            
    captura.release()
    salida.release()
    cv2.destroyAllWindows()
    print(f"Video guardado exitosamente en: {ruta_salida}")

# Ejecutar
if __name__ == "__main__":
    procesar_video(VIDEO_ENTRADA, VIDEO_SALIDA)
