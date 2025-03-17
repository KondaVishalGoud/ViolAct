import os
import cv2
import numpy as np
import torch
from torchvision import models
import torch.nn as nn
from flask import Flask, render_template, request, jsonify
import base64
from PIL import Image
import io
import matplotlib.pyplot as plt
import tempfile

# Constants
IMAGE_HEIGHT, IMAGE_WIDTH = 224, 224
SEQUENCE_LENGTH = 15
CLASSES_LIST = ["NonViolence", "Violence"]
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max upload size

class ViolenceDetectionModel(nn.Module):
    def __init__(self, num_classes=2):
        super(ViolenceDetectionModel, self).__init__()
        efficientnet = models.efficientnet_b7(pretrained=True)
        self.features = nn.Sequential(*list(efficientnet.children())[:-1])
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.feature_dim = 2560
        self.lstm = nn.LSTM(input_size=self.feature_dim, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(128, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()

    def forward(self, x):
        batch_size, channels, seq_len, height, width = x.size()
        features_seq = []
        for t in range(seq_len):
            frame = x[:, :, t, :, :]
            frame_features = self.features(frame)
            frame_features = self.avg_pool(frame_features).view(batch_size, -1)
            features_seq.append(frame_features)
        features_seq = torch.stack(features_seq, dim=1)
        lstm_out, _ = self.lstm(features_seq)
        lstm_out = lstm_out[:, -1, :]
        x = self.dropout(lstm_out)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.relu(self.fc4(x))
        x = self.dropout(x)
        x = self.fc5(x)
        return x

def load_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ViolenceDetectionModel(num_classes=len(CLASSES_LIST)).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, device

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_video(video_path):
    frames_list = []
    video_reader = cv2.VideoCapture(video_path)
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    skip_frames_window = max(int(video_frames_count/SEQUENCE_LENGTH), 1)

    for frame_counter in range(SEQUENCE_LENGTH):
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)
        success, frame = video_reader.read()
        if not success:
            break
        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        normalized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB) / 255.0
        frames_list.append(normalized_frame)

    video_reader.release()
    return frames_list

def predict_on_video(model, device, video_path):
    # Preprocess the video
    frames = preprocess_video(video_path)
    if len(frames) < SEQUENCE_LENGTH:
        return None, None, frames

    # Prepare the input tensor
    input_tensor = torch.FloatTensor(frames).permute(3, 0, 1, 2).unsqueeze(0).to(device)

    # Make prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)

    predicted_class_idx = predicted.item()
    predicted_class = CLASSES_LIST[predicted_class_idx]

    return predicted_class, frames

def frames_to_base64(frames):
    base64_frames = []
    for i, frame in enumerate(frames):
        if i >= 15:  # Limit to 15 frames
            break
        # Convert numpy array to PIL Image
        img = Image.fromarray((frame * 255).astype(np.uint8))
        # Save image to bytes buffer
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        # Encode as base64
        img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        base64_frames.append(img_str)
    return base64_frames

# Load model at startup
MODEL_PATH = 'best_model_efficientnetb7.pth'  # Update with your model path
try:
    model, device = load_model(MODEL_PATH)
    model_loaded = True
except Exception as e:
    print(f"Error loading model: {e}")
    model_loaded = False

# Update the routes to handle both pages
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect')
def detect():
    return render_template('detect.html', model_status=model_loaded)

# Update the predict route to match the new structure
@app.route('/predict', methods=['POST'])
def predict():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    
    if file.filename == '':
        return jsonify({'error': 'No video selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed. Please upload mp4, avi, mov, or mkv files.'}), 400
    
    if not model_loaded:
        return jsonify({'error': 'Model not loaded. Please check server logs.'}), 500
    
    try:
        # Create a temporary file to store the uploaded video
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.filename.split('.')[-1]}") as temp_file:
            file.save(temp_file.name)
            temp_path = temp_file.name
        
        # Process the video
        predicted_class, frames = predict_on_video(model, device, temp_path)
        
        # Clean up the temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        if predicted_class is None:
            return jsonify({'error': 'Could not extract enough frames from the video'}), 400
        
        # Convert frames to base64 for display
        base64_frames = frames_to_base64(frames)
        
        # Return the prediction result
        return jsonify({
            'prediction': predicted_class,
            'frames': base64_frames
        })
    
    except Exception as e:
        # Clean up the temporary file if it exists
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({'error': f'Error processing video: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)
