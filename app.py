import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Traffic Light AI",
    page_icon="🚦",
    layout="wide"
)

# ================= CUSTOM CSS =================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #2e003e, #4a148c, #6a1b9a);
    color: white;
}

.title {
    text-align:center;
    font-size:45px;
    font-weight:bold;
    color:white;
    margin-bottom:20px;
}

.card {
    background: rgba(255,255,255,0.10);
    padding:25px;
    border-radius:20px;
    text-align:center;
    backdrop-filter: blur(10px);
    box-shadow:0 8px 25px rgba(0,0,0,0.3);
}

.box {
    padding:12px;
    border-radius:10px;
    font-weight:bold;
    margin-top:10px;
}

.red {background:#ff4b4b;}
.green {background:#00c853;}
.yellow {background:#ffd600; color:black;}
.back {background:#333;}

/* ================= UPLOAD BOX TEXT ================= */

div[data-testid="stFileUploader"] {
    background-color: #000000;
    padding: 20px;
    border-radius: 15px;
    border: 2px solid #6a1b9a;
}

section[data-testid="stFileUploaderDropzone"] * {
    color: #2d1b12 !important;
    font-weight: bold !important;
}

</style>
""", unsafe_allow_html=True)
# ================= TITLE =================
st.markdown('<div class="title">🚦 Traffic Light Detection System 🚦</div>', unsafe_allow_html=True)

# ================= MODEL =================
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3,16,3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16,32,3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*20*20,128), nn.ReLU(),
            nn.Linear(128,4)
        )

    def forward(self, x):
        return self.model(x)

model = CNN()
model.load_state_dict(torch.load("traffic_light_model.pth", map_location="cpu"))
model.eval()

classes = ["back", "green", "red", "yellow"]

# ================= TRANSFORM =================
transform = transforms.Compose([
    transforms.Resize((180,180)),
    transforms.ToTensor()
])

# ================= UPLOAD =================
file = st.file_uploader("📤 Upload Traffic Light Image", type=["jpg","png","jpeg"])

probs = None
label = None
confidence = 0

# ================= PREDICTION =================
if file:
    image = Image.open(file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=280)

    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(img)
        probs = F.softmax(output, dim=1)[0]

        confidence, pred = torch.max(probs, 0)

    label = classes[pred.item()]
    confidence = confidence.item() * 100

# ================= RESULT =================
col1, col2 = st.columns(2)

with col1:
    if label:
        st.markdown(f"""
        <div class="card">
            <h2>Prediction Result</h2>
            <div class="box {label}">
                {label.upper()}
            </div>
            <h3>Confidence: {confidence:.2f}%</h3>
        </div>
        """, unsafe_allow_html=True)

# ================= CONFIDENCE BREAKDOWN =================
with col2:
    if probs is not None:
        st.markdown("### 📊 Class Confidence Breakdown")

        for i, cls in enumerate(classes):
            percent = probs[i].item() * 100

            color_class = cls

            st.markdown(f"""
            <div class="box {color_class}">
                {cls.upper()} : {percent:.2f}%
            </div>
            """, unsafe_allow_html=True)

# ================= FOOTER =================
st.markdown("---")
st.markdown("🚦 Computer Vision Project | Traffic Light Detection System")