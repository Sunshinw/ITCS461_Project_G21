# streamlit for interactive website
# cv2 for image processing
# numpy for image array
# PIL for image manage
# pandas dataframe
# hand_utils file import features
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import os
from hand_utils import extract_features, match_hand

# page layout title
st.set_page_config(page_title="Hand Geometry Authentication System", layout="centered")
# page topic
st.title("Hand Geometry Authentucation System")

# mode select register or authenticate
mode = st.radio("Choose Mode:", ["Register", "Authenticate"])

# input image from camera
img_input = st.camera_input("Scan your hand")

# if image input
if img_input is not None:
    image = Image.open(img_input)
    # save image to array
    image_np = np.array(image)
    # convert RGB to BGR color (to use in OpenCV)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # show the image taken
    st.image(image_np, caption="Scanned Hand", use_container_width=True)
    # use features extract
    features = extract_features(image_bgr)

    # show the result from extract
    # error extract
    if features is None:
        st.error("Hand Extraction Failed. Please try again.")
    # extract complete
    else:
        st.success("Hand Extraction Succesfully.")

        # register mode
        if mode == "Register":
            username = st.text_input("Enter your username")

            if st.button("Register"):
                if username:
                    # create new dataframe in CSV format
                    os.makedirs("data", exist_ok=True)
                    df_path = "data/hand_database.csv"
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
                    # if the file already exist
                    if os.path.exists(df_path):
                        df = pd.read_csv(df_path)
                        # if already have concat
                        df = pd.concat([df, new_data], ignore_index=True)
                    else:
                        # if not create new
                        df = new_data
                    df.to_csv(df_path, index=False)
                    st.success(f"✅ Registered user: {username}")
                else:
                    st.warning("Please enter a username.")
        
        # authentication mode
        elif mode == "Authenticate":
            matched_user = match_hand(features)
            if matched_user:
                st.success(f"✅ Authentication success! Welcome, {matched_user}.")
            else:
                st.error("❌ Authentication failed.")