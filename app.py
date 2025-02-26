import streamlit as st
import torch
import cv2
import tempfile
import torch.nn as nn
import numpy as np
from moviepy import VideoFileClip
import dlib

# Initialize Dlib face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Ensure this file is in your directory

class LipNetModel(nn.Module):
    def __init__(self, vocab_size):
        super(LipNetModel, self).__init__()
        self.conv3d_1 = nn.Conv3d(1, 128, kernel_size=3, padding=1)
        self.pool3d_1 = nn.MaxPool3d((1, 2, 2))
        self.conv3d_2 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.pool3d_2 = nn.MaxPool3d((1, 2, 2))
        self.conv3d_3 = nn.Conv3d(256, 75, kernel_size=3, padding=1)
        self.pool3d_3 = nn.MaxPool3d((1, 2, 2))
        self.lstm = nn.LSTM(75 * 5 * 17, 128, num_layers=2, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(256, vocab_size + 1)

    def forward(self, x):
        x = self.pool3d_1(torch.relu(self.conv3d_1(x)))
        x = self.pool3d_2(torch.relu(self.conv3d_2(x)))
        x = self.pool3d_3(torch.relu(self.conv3d_3(x)))
        x = x.view(x.size(0), x.size(2), -1)  # Flatten for LSTM
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x

# Load LipNet model (modify this according to your model loading method)
@st.cache_resource
def load_model():
    vocab_size = 39
    model = LipNetModel(vocab_size)
    model.load_state_dict(torch.load("lipnet_static_crop.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# Function to process video and extract lip region
def load_video(path: str, target_height=46, target_width=140) -> list:
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        if len(faces) > 0:
            face = faces[0]
            landmarks = predictor(gray, face)
            lip_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(48, 68)]
            lip_rect = cv2.boundingRect(np.array(lip_points))
            lip_region = frame[lip_rect[1]:lip_rect[1]+lip_rect[3], lip_rect[0]:lip_rect[0]+lip_rect[2]]
            lip_region_resized = cv2.resize(lip_region, (target_width, target_height))
            frames.append(lip_region_resized)

    cap.release()
    return frames  # Return extracted lip regions in color

# Streamlit UI
st.title("Lip Detection Dashboard")
st.write("Upload a video, and the model will process it for lip detection.")

uploaded_file = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov", "mpg"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        temp_file.write(uploaded_file.read())
        video_path = temp_file.name

    st.video(video_path)

    if st.button("Process Video"):
        frames = load_video(video_path)
        st.write("Extracted Lip Regions")

        st.image(frames, caption=["Extracted Lip Region"]*len(frames), use_column_width=True)