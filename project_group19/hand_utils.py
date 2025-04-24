# cv2 for image processing
# numpy for image array and calculate the hand vector
# pandas for CSV manage
# mediapipe for hand recognition (hand landmark)
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

# mediapipe setup
# loading module for recognition hand
mp_hands = mp.solutions.hands
# loading module for drawing line between finger
mp_drawing = mp.solutions.drawing_utils


def calculate_angle(a, b, c):
    # make a be array
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    # vector distance calculation
    ba = a - b
    bc = c - b

    # calculate the cosine using dot product
    # using 1e-6 to prevent divide 0
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    # convert cosine result to angle
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    # return in degree
    return round(np.degrees(angle), 2)

# hand extract features
def extract_features(image, show_debug=False):
    try:
        # convert BGR back to RGB format for mediapipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # hand object create
        # static image mode for image (max hand only 1)
        with mp_hands.Hands(static_image_mode=True, max_num_hands=1) as hands:
            results = hands.process(image_rgb)

            # if no hand in image
            if results.multi_hand_landmarks is None:
                return None

            # 21 point landmark from hand
            landmarks = results.multi_hand_landmarks[0].landmark
            h, w, _ = image.shape
            # calculate the point (lm.x, lm.y are nomalize)
            points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

            # pull calculate the angle data from every finger
            # Joint angles: MCP -> PIP -> DIP
            thumb = calculate_angle(points[1], points[2], points[3])
            index = calculate_angle(points[5], points[6], points[7])
            middle = calculate_angle(points[9], points[10], points[11])
            ring = calculate_angle(points[13], points[14], points[15])
            pinky = calculate_angle(points[17], points[18], points[19])

            # Span (thumb tip to pinky tip) distance different calculate
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

# authentication match hand feature
def match_hand(input_features):
    try:
        # read the registered CSV data
        df = pd.read_csv("data/hand_database.csv")
        # loop to access every row of data and convert to list
        for _, row in df.iterrows():
            stored = [
                row['thumb'], row['index'], row['middle'],
                row['ring'], row['pinky'], row['span'], row['aspect_ratio']
            ]
            dist = np.linalg.norm(np.array(input_features) - np.array(stored))
            # if the dist is less than the match
            if dist < 25:  # add this if need more accurate
                return row['username']
        return None
    except Exception as e:
        print(f"[ERROR] Matching failed: {e}")
        return None
