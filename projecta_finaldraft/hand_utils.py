import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return round(np.degrees(angle), 2)

def extract_features(image, show_debug=False):
    try:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        with mp_hands.Hands(static_image_mode=True, max_num_hands=1) as hands:
            results = hands.process(image_rgb)

            if results.multi_hand_landmarks is None:
                return None

            landmarks = results.multi_hand_landmarks[0].landmark
            h, w, _ = image.shape
            points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

            # Joint angles: MCP -> PIP -> DIP
            thumb = calculate_angle(points[1], points[2], points[3])
            index = calculate_angle(points[5], points[6], points[7])
            middle = calculate_angle(points[9], points[10], points[11])
            ring = calculate_angle(points[13], points[14], points[15])
            pinky = calculate_angle(points[17], points[18], points[19])

            # Span (thumb tip to pinky tip)
            span = np.linalg.norm(np.array(points[4]) - np.array(points[20]))

            # Aspect ratio (bounding box of hand)
            xs, ys = zip(*points)
            hand_width = max(xs) - min(xs)
            hand_height = max(ys) - min(ys)
            aspect_ratio = round(hand_width / hand_height, 2)

            if show_debug:
                annotated = image.copy()
                mp_drawing.draw_landmarks(annotated, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
                cv2.imshow("Debug View", annotated)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            return [thumb, index, middle, ring, pinky, span, aspect_ratio]

    except Exception as e:
        print(f"[ERROR] Feature extraction failed: {e}")
        return None

def match_hand(input_features):
    try:
        df = pd.read_csv("data/features.csv")
        for _, row in df.iterrows():
            stored = [
                row['thumb'], row['index'], row['middle'],
                row['ring'], row['pinky'], row['span'], row['aspect_ratio']
            ]
            dist = np.linalg.norm(np.array(input_features) - np.array(stored))
            if dist < 25:  # adjust this if needed
                return row['username']
        return None
    except Exception as e:
        print(f"[ERROR] Matching failed: {e}")
        return None
