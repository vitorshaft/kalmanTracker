import cv2
import numpy as np

# Inicializa o filtro de Kalman
kalman = cv2.KalmanFilter(4, 2)
kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 5, 0], [0, 0, 0, 5]], np.float32)

# Captura de vídeo
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Conversão para o espaço de cor HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define o intervalo de cor para a bola de tênis amarela
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])

    # Cria uma máscara para a cor amarela
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Encontrar contornos na máscara
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Encontra o maior contorno
        largest_contour = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(largest_contour)

        if radius > 10:
            # Coordenadas de medição
            measured = np.array([[np.float32(x)], [np.float32(y)]])

            # Correcção do filtro de Kalman
            kalman.correct(measured)

            # Predição do filtro de Kalman
            predicted = kalman.predict()

            # Desenha o círculo em torno do objeto detectado
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)

            # Desenha a predição do filtro de Kalman
            cv2.circle(frame, (int(predicted[0]), int(predicted[1])), int(radius), (0, 0, 255), 2)

    # Mostra o vídeo
    cv2.imshow('Frame', frame)

    # Sai do loop quando a tecla 'q' é pressionada
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera os recursos
cap.release()
cv2.destroyAllWindows()
