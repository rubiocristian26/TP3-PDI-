import numpy as np
import cv2

# --- Input ---------------------------------------------------------------------------------
file_name = "ruta_1.mp4"

# --- Proceso video -------------------------------------------------------------------------
capture = cv2.VideoCapture(file_name)
while True:
    # --- Obtengo frame -----------------------------------------------------------------
    ret, img = capture.read()
    if not ret: 
        break
    cv2.imshow("Original", img)

    # --- ROI --------------------------------------------------------------------------
    dims = img.shape
    left_bottom = [0, dims[0]-1]
    left_top = [450, 320]
    right_top = [520, 320]
    right_bottom = [dims[1]-1, dims[0]-1]
    roi_vertices = np.array([left_bottom, left_top, right_top, right_bottom])

    mask = np.zeros(img.shape[:-1], dtype=np.uint8)
    cv2.fillPoly(mask, [roi_vertices], 255)
    img_roi = cv2.bitwise_and(img, img, mask=mask)

    cv2.imshow("Mask", mask)
    cv2.imshow("ROI", img_roi)

    # --- Para salir del algoritmo ----------------------------------------------------------- 
    if cv2.waitKey(20) & 0xFF==ord('q'):     # Si la 'q' es pulsada, salimos del programa
        break

