import cv2
import numpy as np
import mediapipe as mp

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Configuración de la ventana del juego
width, height = 800, 600
window_name = "Game Ball and Brick con OpenCV"

# Configuración de la pelota
ball_radius = 10
ball_pos = np.array([width // 2, height // 2])
ball_speed = np.array([10, -10])

# Configuración de la barra del jugador
paddle_width, paddle_height = 100, 20
paddle_pos = np.array([width // 2 - paddle_width // 2, height - 50])

# Configuración de los ladrillos
brick_rows, brick_cols = 5, 8
brick_width, brick_height = width // brick_cols, 30
bricks = [[True] * brick_cols for _ in range(brick_rows)]

# Configuración de la cámara
cap = cv2.VideoCapture(0)
game_over = False

# Función para dibujar los elementos en el juego
def draw_elements(frame):
    # Dibujar la pelota
    cv2.circle(frame, tuple(ball_pos), ball_radius, (0, 255, 255), -1)
    # Dibujar la barra del jugador
    cv2.rectangle(frame, tuple(paddle_pos), (paddle_pos[0] + paddle_width, paddle_pos[1] + paddle_height), (255, 0, 0), -1)
    # Dibujar los ladrillos
    for row in range(brick_rows):
        for col in range(brick_cols):
            if bricks[row][col]:
                top_left = (col * brick_width, row * brick_height)
                bottom_right = (top_left[0] + brick_width, top_left[1] + brick_height)
                cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), -1)

# Lógica principal del juego
while cap.isOpened():
    ret, camera_frame = cap.read()
    if not ret:
        break

    # Redimensionar la vista de la cámara para que encaje con el tamaño del juego
    camera_frame = cv2.resize(camera_frame, (width, height))
    
    # Convertir la imagen de la cámara a RGB para MediaPipe
    rgb_frame = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    # Verificar si hay alguna mano detectada y mover la barra
    if results.multi_hand_landmarks and not game_over:
        for hand_landmarks in results.multi_hand_landmarks:
            x_position = int(hand_landmarks.landmark[9].x * width)
            paddle_pos[0] = np.clip(x_position - paddle_width // 2, 0, width - paddle_width)

            # Cambiar colores de los nodos de la mano
            for idx, landmark in enumerate(hand_landmarks.landmark):
                # Asignar colores por dedo
                if idx in [0, 1, 2, 3, 4]:  # Pulgar
                    color = (0, 0, 255)  # Rojo
                elif idx in [5, 6, 7, 8]:  # Índice
                    color = (255, 0, 0)  # Azul
                elif idx in [9, 10, 11, 12]:  # Medio
                    color = (0, 255, 0)  # Verde
                elif idx in [13, 14, 15, 16]:  # Anular
                    color = (255, 255, 0)  # Cian
                elif idx in [17, 18, 19, 20]:  # Meñique
                    color = (255, 0, 255)  # Magenta
                else:
                    color = (255, 255, 255)  # Blanco (por defecto)

                # Convertir coordenadas normalizadas a píxeles
                h, w, _ = camera_frame.shape
                x = int(landmark.x * w)
                y = int(landmark.y * h)

                # Dibujar cada nodo con el color asignado
                cv2.circle(camera_frame, (x, y), 5, color, -1)

            # Dibujar las conexiones manualmente
            for connection in mp_hands.HAND_CONNECTIONS:
                start_idx = connection[0]
                end_idx = connection[1]

                start = hand_landmarks.landmark[start_idx]
                end = hand_landmarks.landmark[end_idx]

                start_point = (int(start.x * w), int(start.y * h))
                end_point = (int(end.x * w), int(end.y * h))

                # Conexión estándar en gris claro
                cv2.line(camera_frame, start_point, end_point, (200, 200, 200), 2)

    # Crear el marco de juego
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Movimiento de la pelota
    if not game_over:
        ball_pos += ball_speed
    
    # Rebote en los bordes de la ventana
    if ball_pos[0] <= ball_radius or ball_pos[0] >= width - ball_radius:
        ball_speed[0] = -ball_speed[0]
    if ball_pos[1] <= ball_radius:
        ball_speed[1] = -ball_speed[1]
    elif ball_pos[1] >= height - ball_radius:  # Juego termina si toca la parte inferior
        game_over = True
    
    # Rebote con la barra del jugador
    if (paddle_pos[1] <= ball_pos[1] + ball_radius <= paddle_pos[1] + paddle_height and
        paddle_pos[0] <= ball_pos[0] <= paddle_pos[0] + paddle_width):
        ball_speed[1] = -ball_speed[1]
    
    # Rebote con los ladrillos
    for row in range(brick_rows):
        for col in range(brick_cols):
            if bricks[row][col]:
                brick_x, brick_y = col * brick_width, row * brick_height
                if (brick_x <= ball_pos[0] <= brick_x + brick_width and
                    brick_y <= ball_pos[1] <= brick_y + brick_height):
                    bricks[row][col] = False
                    ball_speed[1] = -ball_speed[1]
                    break
    
    # Dibujar todos los elementos en el marco del juego
    draw_elements(frame)
    
    # Fusionar la vista de la cámara con el marco del juego
    combined_frame = cv2.addWeighted(camera_frame, 0.7, frame, 0.8, 0)
    
    # Mostrar el mensaje de "Game Over" si el juego ha terminado
    if game_over:
        cv2.putText(combined_frame, "Game Over", (width // 2 - 100, height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4, cv2.LINE_AA)

    # Mostrar la combinación en una sola ventana
    cv2.imshow(window_name, combined_frame)

    # Salir con la tecla Escape
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
hands.close()
cv2.destroyAllWindows()