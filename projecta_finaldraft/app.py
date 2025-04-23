import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import os
from hand_utils import extract_features, match_hand

st.set_page_config(page_title="Hand Geometry System", layout="centered")
st.title("🖐️ Hand Geometry System")

mode = st.radio("Choose Mode:", ["Register", "Authenticate"])

img_input = st.camera_input("Scan your hand")

if img_input is not None:
    image = Image.open(img_input)
    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    st.image(image_np, caption="Scanned Hand", use_container_width=True)
    features = extract_features(image_bgr)

    if features is None:
        st.error("❗ Feature extraction failed. Please try again.")
    else:
        st.success("🔍 Features extracted successfully.")

        if mode == "Register":
            username = st.text_input("Enter your username")

            if st.button("Register"):
                if username:
                    # Save to CSV
                    os.makedirs("data", exist_ok=True)
                    df_path = "data/features.csv"
                    new_data = pd.DataFrame([{
                    "username": username,
                    "thumb": features[0],
                    "index": features[1],
                    "middle": features[2],
                    "ring": features[3],
                    "pinky": features[4],
                    "span": features[5],
                    "aspect_ratio": features[6]
                }])

                    if os.path.exists(df_path):
                        df = pd.read_csv(df_path)
                        df = pd.concat([df, new_data], ignore_index=True)
                    else:
                        df = new_data
                    df.to_csv(df_path, index=False)
                    st.success(f"✅ Registered user: {username}")
                else:
                    st.warning("Please enter a username.")
        
        elif mode == "Authenticate":
            matched_user = match_hand(features)
            if matched_user:
                st.success(f"✅ Authentication success! Welcome, {matched_user}.")
            else:
                st.error("❌ Authentication failed.")