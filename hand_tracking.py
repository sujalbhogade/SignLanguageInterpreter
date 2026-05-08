import mediapipe as mp

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def get_hand_bbox(hand_landmarks, frame_width, frame_height, padding=50):
    xs = [lm.x * frame_width for lm in hand_landmarks.landmark]
    ys = [lm.y * frame_height for lm in hand_landmarks.landmark]

    x_min = int(max(min(xs) - padding, 0))
    x_max = int(min(max(xs) + padding, frame_width))

    y_min = int(max(min(ys) - padding, 0))
    y_max = int(min(max(ys) + padding, frame_height))

    return x_min, y_min, x_max, y_max
