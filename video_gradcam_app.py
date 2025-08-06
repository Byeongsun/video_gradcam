import torch
import torch.nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

import streamlit as st 
import tempfile


@st.cache_resource
def load_model():
    model = models.resnet18(pretrained=True)
    model.eval()
    return model

@st.cache_resource
def load_labels():
    import urllib.request
    url = 'https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt'
    return [line.strip().decode('utf-8') for line in urllib.request.urlopen(url).readlines()]

model = load_model()
labels = load_labels()

print(model)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device)
model.to(device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Streamlit UI
st.title("🎬 동영상 입력 Grad-CAM 분류기")

uploaded_video = st.file_uploader('MP4 동영상을 업로드하세요', type=['mp4'])

if uploaded_video is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    cap = cv2.VideoCapture(tfile.name)
    
    st.video(uploaded_video)
    st.subheader('분석된 샘플 프레임 (Grad-CAM')
    
    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    
    frame_count = 0
    shown = 0
    max_frames = 20
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1

        if frame_count < 100:
            continue

        if frame_count % 20 != 0:
            continue
                
        # openCV -> PIL 이미지
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        
        input_tensor = transform(pil_img).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_tensor)
            pred_class = torch.argmax(output).item()
            class_name = labels[pred_class]
            
        # GradCAM
        grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(pred_class)])[0]
        image_np = np.array(pil_img.resize((224, 224))).astype(np.float32) / 255.0
        cam_image = show_cam_on_image(image_np, grayscale_cam, use_rgb=True)
        
        st.image(cam_image, caption=f'🔍 예측: {class_name}', use_container_width=True)
        
        shown += 1
        if shown >= max_frames:
            break
    cap.release()
    