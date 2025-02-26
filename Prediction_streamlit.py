import streamlit as st
import torch
import torch.nn as nn
import cv2
import tempfile
import numpy as np
import dlib
from moviepy.editor import VideoFileClip
from torch.utils.data import DataLoader, Dataset

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the vocabulary for lip reading
vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
char_to_num = {char: idx for idx, char in enumerate(vocab)}
num_to_char = {idx: char for char, idx in char_to_num.items()}

# Initialize Dlib face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Define LipNet Model
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

# Load LipNet model
@st.cache_resource
def load_model():
    model = LipNetModel(len(vocab)).to(device)
    model.load_state_dict(torch.load("models/lipnet.pth", map_location=device))
    model.eval()
    return model

model = load_model()

# Function to process video and extract lip region
def load_video(path: str, target_height=46, target_width=140) -> torch.Tensor:
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
    frames = np.array(frames)
    frames = torch.tensor(frames, dtype=torch.float32)
    mean = frames.mean()
    std = frames.std()
    return (frames - mean) / std, frames.numpy()  # Return normalized frames and raw frames

# Function to predict words from video
def predict_from_video(video_path: str):
    frames, raw_frames = load_video(video_path)
    frames = frames.unsqueeze(0).unsqueeze(0).to(device)  # Reshape for model input

    with torch.no_grad():
        outputs = model(frames)
        decoded_predictions = torch.argmax(outputs, dim=2)
        decoded_text = ''.join([num_to_char[idx.item()] for idx in decoded_predictions[0] if idx.item() < len(vocab)])

    return decoded_text, raw_frames

# Streamlit UI
st.title("Lip Reading Application")
st.write("Upload a video, and the model will predict the spoken words and show the extracted lip frames.")

uploaded_file = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov", "mpg"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        temp_file.write(uploaded_file.read())
        video_path = temp_file.name
    
    st.video(video_path)

    if st.button("Process Video"):
        st.write("Processing video...")
        predicted_text, extracted_frames = predict_from_video(video_path)

        # Display predicted words
        st.subheader("Predicted Words:")
        st.write(f"**{predicted_text}**")

        # Display extracted frames
        st.subheader("Extracted Lip Frames:")
        st.image(extracted_frames, caption=["Lip Region"] * len(extracted_frames), use_column_width=True)
