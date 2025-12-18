import cv2
import numpy as np
import matplotlib.pyplot as plt

# Cargamos el video en la variable 'ruta'
ruta = "ruta_1.mp4"       
# Parámetros. 
roi_offset = 5                              # Desplazamiento vertical para ajustar el punto superior de las lineas        
canny_umbrales = (50, 150)                  # Umbrales Inferior y Superior para el detector de bordes de Canny  
parametros_hough = {                    
    "rho": 1,                               # Resolución en píxeles del parámetro rho. 1 pixel de resolucion
    "theta": np.pi / 180,                   # Resolución angular en radianes.          1 radian de resolucion.
    "threshold": 30,                        # Umbral mínimo de intersecciones para detectar una línea
    " minLineLength": 40,                   # Longitud mínima de un segmento para ser considerado línea
    " maxLineGap": 100                      # Máxima separación permitida entre segmentos para unirlos
}


# --- Funciones de Procesamiento  ---

def region_interes(img):
    ancho, alto = img.shape[:2]         # Adquirimos las dimensiones de la imagen.
    poligonos = np.array([               # Definimos la region de interes.
        [(0, ancho), (450, 320), (520, 320), (alto, ancho)]
    ])
    mask = np.zeros_like(img)           # Creamos una mascara negra
    cv2.fillPoly(mask, poligonos, 255)   # Máscara blanca de la ROI sobre fondo negro
    img_mask = cv2.bitwise_and(img, mask)   #Aplicamos la mascara a la imagen original
    return img_mask, mask               #Devolvemos la imagen en la region de la mascara

def coordenadas(imagen, parametros_linea):
    if parametros_linea is None: return None    #En caso de no detectar una linea, no devolvemos nada.
    pendiente, interseccion = parametros_linea  # parametros_linea guarda la pendiente y la interseccion de la recta
    y1 = imagen.shape[0]                        # Cordenada eje Y inferior.
    y2 = int(y1 * (6/10)) + roi_offset          # Cooordenada eje Y superior.    
    if pendiente == 0: pendiente = 0.001        #En caso de tener una recta con pendiente cero, reemplazamos para evitar la division por cero.
    x1 = int((y1 - interseccion) / pendiente)   #Calculamos las coordenadas de X para y1 e y2   
    x2 = int((y2 - interseccion) / pendiente)
    return np.array([x1, y1, x2, y2])           #Devolvemos las coordenadas de la linea extrapolada

def average_slope_intercept(imagen, lineas):
    car_izq = []                               #Lineas para almacenar las lineas de los carriles izquierdo y derecho
    car_der = []                

    if lineas is None: return None, None

    for line in lineas:                        # Recorremos cada segmento detectado por Hough
        x1, y1, x2, y2 = line.reshape(4)
        parametros = np.polyfit((x1, x2), (y1, y2), 1)  #Ajustamos usando una ecuacion lineal.
        pendiente_2 = parametros[0]
        interseccion_2 = parametros[1]
        #En funcion de la pendiente, podemos determinar si se trata del carril izquierdo o derecho. 
        if pendiente_2 < -0.4:                              #Pendiente negativa == Carril izquierdo
            car_izq.append((pendiente_2, interseccion_2))       
        elif pendiente_2 > 0.4:
            car_der.append((pendiente_2, interseccion_2))   # Pendiente positiva == Carril positivo

    promedio_izquierdo = np.average(car_izq, axis=0) if car_izq else None
    promedio_derecho = np.average(car_der, axis=0) if car_der else None
    linea_izq = coordenadas(imagen, promedio_izquierdo)     #Calculamos la linea izquierda y derecha extrapolando.  
    linea_der = coordenadas(imagen, promedio_derecho)
    return linea_izq, linea_der                             #Devolvemos las lineas finales.

def mostrar_lineas(imagen, lineas):         
    linea_imagen = np.zeros_like(imagen)                    #Creamos una imagen negra para dibujarle las lineas encima
    if lineas is not None:                                  #verificamos que lineas no este vacio
        for line in lineas:
            if line is not None:                            #Y Tambien que la linea que contenga lineas sea valida
                x1, y1, x2, y2 = line
                cv2.line(linea_imagen, (x1, y1), (x2, y2), (255, 0, 0), 10)     #Dibujamos la linea en Azul (recordando que cv2 utiliza BGR)
    return linea_imagen

# --- Función de Análisis de un Frame ---

def analizar_frame_paso_a_paso(direccion_video):
    ruta_vid = cv2.VideoCapture(direccion_video)         #guardamos en ruta_vid, el video de la direccion dada.
    
    # Leemos solo el primer frame válido

    frame = ruta_vid.read()[1]   #Guardamos el primer frame del video

    # --- ETAPA 1: Imagen Original ---
    # Convertimos BGR (OpenCV) a RGB (Matplotlib) para que se vea bien
    img_original_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 

    plt.imshow(img_original_rgb)  # 1. Mostramos la imagen
    plt.title("1. Frame Video Original")
    plt.show()  # "Abre" la imagen, y frena el codigo hasta que se cierre  

    # --- ETAPA 2: Detección de Bordes (Canny) ---
    lane_image = np.copy(frame)                                         #Realizamos una copia de la imagen original
    gray = cv2.cvtColor(lane_image, cv2.COLOR_BGR2GRAY)                 #Pasamos la imagen a escala de grises
    blur = cv2.GaussianBlur(gray, (5, 5), 0)                            #Realizamos un filtrado gaussiano, para reducir el ruido.
    img_canny = cv2.Canny(blur, canny_umbrales[0], canny_umbrales[1])   #Detectamos los bordes con canny.

    plt.imshow(img_canny, cmap='gray')
    plt.title("2. Bordes detectados con CANNY")
    plt.show()  #  

    # --- ETAPA 3: Región de Interés (ROI) ---
    # Obtenemos la imagen recortada y la máscara visual
    img_recortada, mask_visual = region_interes(img_canny)      #Aplicamos ROI
    plt.imshow(img_recortada, cmap='gray')                      #Mostramos la imagen recortada
    plt.title("3. Region de Interes (ROI)")
    plt.show()
    
    # --- ETAPA 4: Detección Hough (Segmentos sueltos) ---
    # Para visualizar esto, dibujamos los segmentos "crudos" que ve Hough
    lines_raw = cv2.HoughLinesP(
        img_recortada,
        parametros_hough["rho"], 
        parametros_hough["theta"], 
        parametros_hough["threshold"],
        np.array([]), 
         minLineLength=parametros_hough[" minLineLength"], 
         maxLineGap=parametros_hough[" maxLineGap"])
    
    img_hough = np.copy(img_original_rgb)       # realizamos una copia de la imagen en rgb, para mostrar las lineas
    if lines_raw is not None:
        for line in lines_raw:
            x1, y1, x2, y2 = line[0]
            # Dibujamos segmentos en VERDE para diferenciar
            cv2.line(img_hough, (x1, y1), (x2, y2), (0, 255, 0), 2)

    #Mostramos los segmentos de HOUGH
    plt.imshow(img_hough)
    plt.title("4. Segmentos de Hough")
    plt.show()

    # --- ETAPA 5: Extrapolación y Promedio (Resultado Final) ---
    left_line, right_line = average_slope_intercept(lane_image, lines_raw) 
    imagen_lineas = mostrar_lineas(lane_image, [left_line, right_line])
    # Convertimos la imagen de líneas (que es negra con lineas azules) a RGB
    # Pero como las funcion mostrar_lineas usa BGR (Azul 255,0,0), 
    # por lo que no es necesario convertir a RGB.
    imagen_lineas_rgb = imagen_lineas # Lo dejamos así para superponer
    
    # Superposición final
    imagen_combinada = cv2.addWeighted(frame, 0.8, imagen_lineas, 1, 1)
    imagen_combinada_rgb = cv2.cvtColor(imagen_combinada, cv2.COLOR_BGR2RGB)

    #Mostramos el resultado Final
    plt.imshow(imagen_combinada_rgb)
    plt.title("5. Imagen Final")
    plt.show()

if __name__ == "__main__":
    analizar_frame_paso_a_paso(ruta)
