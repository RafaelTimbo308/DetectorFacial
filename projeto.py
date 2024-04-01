import dlib
import numpy as np
import pyautogui
import time
import cv2

# Inicializar detector de rostos e pontos-chave do rosto
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Baixe o arquivo correspondente

# Função para calcular a distância entre dois pontos
def calculate_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

# Função para calcular a distância entre os pontos 0 e 36 (extremidade esquerda e canto superior esquerdo do olho esquerdo)
def calculate_distance_right(landmarks):
    point0 = (landmarks.part(0).x, landmarks.part(0).y)
    point36 = (landmarks.part(36).x, landmarks.part(36).y)
    return calculate_distance(point0, point36)

def calculate_distance_left(landmarks):
    point0 = (landmarks.part(16).x, landmarks.part(16).y)
    point36 = (landmarks.part(45).x, landmarks.part(45).y)
    return calculate_distance(point0, point36)

# Função para verificar se a distância entre os pontos 0 e 36 é menor que o valor mínimo
def check_move_right(landmarks, min_distance):
    distance = calculate_distance_right(landmarks)
    return distance < min_distance


# Função para verificar se a distância entre os pontos 16 e 45 é menor que o valor mínimo
def check_move_left(landmarks, min_distance):
    distance = calculate_distance_left(landmarks)
    return distance < min_distance

# Valor mínimo de distância para acionar o evento
min_distance = 13

# Função principal para detectar a distância do olho esquerdo e acionar o evento correspondente
def detect_and_act():
    # Inicializar o vídeo
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Conversão para escala de cinza
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detecção de rostos
        faces = detector(gray)
        for face in faces:
            landmarks = predictor(gray, face)

            # Verificar se a distância do olho esquerdo é menor que o valor mínimo
            if check_move_right(landmarks, min_distance):
                # Acionar o evento correspondente
                pyautogui.press('right')
                print("Direita")
                time.sleep(1)
            
            if check_move_left(landmarks, min_distance):
                # Acionar o evento correspondente
                pyautogui.press('left')
                print("Esquerda")
                time.sleep(1)

        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Exemplo de uso da função principal
if __name__ == "__main__":
    detect_and_act()
