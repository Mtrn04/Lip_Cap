import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
from typing import List
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt
import dlib

# Device configuration (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Vocabulary for lip reading model
vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
char_to_num = {char: idx for idx, char in enumerate(vocab)}
num_to_char = {idx: char for char, idx in char_to_num.items()}

# Initialize dlib's face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')  # Ensure you have this file

# Load and preprocess video frames
# def load_video(path: str) -> torch.Tensor:
#     cap = cv2.VideoCapture(path)
#     frames = []
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         frame = frame[190:236, 80:220]  # Crop the region of interest (ROI)
#         frames.append(frame)
#     cap.release()
#     frames = np.array(frames)
#     frames = torch.tensor(frames, dtype=torch.float32)
#     mean = frames.mean()
#     std = frames.std()
#     return (frames - mean) / std

def load_video(path: str, target_height=46, target_width=140) -> torch.Tensor:
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = detector(gray)
        if len(faces) > 0:
            # Take the first detected face (you can adjust for multiple faces)
            face = faces[0]

            # Get facial landmarks
            landmarks = predictor(gray, face)

            # Get the coordinates of the lip region (mouth landmarks: 48-67)
            lip_points = []
            for i in range(48, 68):
                lip_points.append((landmarks.part(i).x, landmarks.part(i).y))

            # Find the bounding box for the lip region
            lip_rect = cv2.boundingRect(np.array(lip_points))

            # Crop the lip region from the frame
            lip_region = frame[lip_rect[1]:lip_rect[1]+lip_rect[3], lip_rect[0]:lip_rect[0]+lip_rect[2]]
            lip_region = cv2.cvtColor(lip_region, cv2.COLOR_BGR2GRAY)

            # Resize to ensure consistent dimensions
            lip_region_resized = cv2.resize(lip_region, (target_width, target_height))
            frames.append(lip_region_resized)

    cap.release()

    # Convert frames to a tensor
    frames = np.array(frames)
    frames = torch.tensor(frames, dtype=torch.float32)

    # Normalize frames
    mean = frames.mean()
    std = frames.std()
    return (frames - mean) / std

# Load alignments from file
def load_alignments(path: str) -> List[int]:
    with open(path, 'r') as f:
        lines = f.readlines()
    tokens = []
    for line in lines:
        parts = line.split()
        if parts[2] != 'sil':  # Exclude silence tokens
            tokens.extend([' ', parts[2]])
    return [char_to_num[char] for char in ''.join(tokens)]

# Dataset class for loading frames and alignments
class LipNetDataset(Dataset):
    def __init__(self, file_paths: List[str], base_dir: str, target_length: int):
        self.file_paths = file_paths
        self.base_dir = base_dir
        self.target_length = target_length

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_name = os.path.splitext(os.path.basename(self.file_paths[idx]))[0]
        video_path = os.path.join(self.base_dir, 's1', f'{file_name}.mpg')
        alignment_path = os.path.join(self.base_dir, 'alignments', 's1', f'{file_name}.align')
        frames = load_video(video_path)
        alignments = load_alignments(alignment_path)

        # Pad or slice frames to match target length
        if frames.size(0) < self.target_length:
            padding = self.target_length - frames.size(0)
            frames = torch.nn.functional.pad(frames, (0, 0, 0, 0, 0, padding))
        else:
            frames = frames[:self.target_length]

        return frames.unsqueeze(0), torch.tensor(alignments, dtype=torch.long)

# LipNet model class
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

# Collate function for padding sequences in batches
def collate_fn(batch):
    frames = [item[0] for item in batch]
    alignments = [item[1] for item in batch]
    frames_padded = torch.stack(frames)
    alignments_padded = nn.utils.rnn.pad_sequence(alignments, batch_first=True)
    return frames_padded, alignments_padded

# Load the trained model
model = LipNetModel(vocab_size=len(vocab)).to(device)
model.load_state_dict(torch.load('models/lipnet.pth'))
model.eval()

# Function to predict words from video using DataLoader and LipNetDataset
def predict_from_video(video_path: str, base_dir: str, target_length: int):
    # Get the file name and create the corresponding dataset and dataloader
    file_name = os.path.splitext(os.path.basename(video_path))[0]
    file_paths = [video_path]  # Here, we assume that video_path points to a valid video file
    dataset = LipNetDataset(file_paths, base_dir, target_length)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    # Run inference on the video
    with torch.no_grad():
        for frames, _ in dataloader:
            frames = frames.to(device)

            # Pass the frames through the model
            outputs = model(frames)

            # Decode the predictions
            decoded_predictions = torch.argmax(outputs, dim=2)

            # Convert the predictions to text
            decoded_text = ''.join([num_to_char[idx.item()] for idx in decoded_predictions[0] if idx.item() < len(vocab)])

    return decoded_text

# Example usage
video_path = '.\\predictions\\final_output_lgbgzp.mpg'  # Provide the path to the video file
base_dir = './data_new'  # Make sure to adjust this to your dataset directory
target_length = 77  # Set the target length as per your training setup

predicted_words = predict_from_video(video_path, base_dir, target_length)
print(f"Predicted Words: {predicted_words}")
