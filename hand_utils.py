import cv2
import numpy as np
import pandas as pd

def extract_features(image, show_debug=False):
    try:
        # Resize for consistent processing
        image = cv2.resize(image, (400, 400))

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Adaptive thresholding for better hand segmentation
        thresh = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )

        # Optional: show threshold image (for local debugging only!)
        if show_debug:
            cv2.imshow("Threshold", thresh)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        large_contours = [c for c in contours if cv2.contourArea(c) > 3000]

        if len(large_contours) == 0:
            return None

        contour = max(large_contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(contour)

        aspect_ratio = round(w / h, 2)
        return [w, h, aspect_ratio]

    except Exception as e:
        print(f"[ERROR] Feature extraction failed: {e}")
        return None


def match_hand(input_features):
    try:
        df = pd.read_csv("data/features.csv")
        for i, row in df.iterrows():
            stored = [row['width'], row['height'], row['aspect_ratio']]
            dist = np.linalg.norm(np.array(input_features) - np.array(stored))
            if dist < 20:  # tweak this threshold as needed
                return row['username']
        return None
    except Exception as e:
        print(f"[ERROR] Matching failed: {e}")
        return None