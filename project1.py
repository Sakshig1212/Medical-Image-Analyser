"""
project.py
Multi-Modal Medical Image Report Generation + VQA + Explainability + Second Opinion + PDF Export + Evaluation

Features added:
1) Grad-CAM heatmap overlay on uploaded images
2) Second-opinion local model (DenseNet121) inference & ensemble agreement
3) PDF export of full report
4) Interactive VQA (ask questions about the image)
7) Safety & disclaimer section
8) Multi-image upload & combined reporting
9) Preprocessing: contrast (CLAHE), denoise (bilateral), cropping helper
Also includes evaluate_model(...) to measure accuracy/precision/recall/f1/roc_auc
"""
import os
import time
import tempfile
from typing import Tuple, Optional, Dict

import streamlit as st
import numpy as np
from PIL import Image
import cv2
import torch
from torchvision import transforms, models
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import pandas as pd
import time
from dotenv import load_dotenv
load_dotenv()

# os.environ["GOOGLE_API_KEY"] = "AIzaSyBR7HrWZzYnPY-NV4MTE0QbBPotjn_DzEY"

try:
    from agno.agent import Agent
    from agno.models.google import Gemini
    from agno.media import Image as AgnoImage
    from agno.tools.duckduckgo import DuckDuckGoTools
except Exception:
    Agent = None
    Gemini = None
    AgnoImage = None
    DuckDuckGoTools = None

# ----------------- CONFIG -----------------
SECOND_OPINION_MODEL_PATH = "second_opinion.pth"
SYLLABUS_PDF = "/mnt/data/11.pdf"

# ----------------- DARK THEME CSS -----------------
st.markdown("""
<style>
body { background-color:#0d1117; color:#e6edf3; }
.stApp { background-color:#0d1117; color:#e6edf3; }
[data-testid="stSidebar"] { background-color:#0b0f14; }
h1,h2,h3,h4 { color:#e6edf3; }
.stButton>button, .stDownloadButton>button {
    background-color:#161b22; border:1px solid #30363d;
    color:#e6edf3; border-radius:8px;
}
.card {
    background:#111820; border:1px solid #30363d;
    padding:15px; border-radius:10px; margin-bottom:20px;
}
</style>
""", unsafe_allow_html=True)

# ----------------- IMAGE PREPROCESSING -----------------
def preprocess_image_pil(img, clahe=True, denoise=True, crop=None):
    img_gray = img.convert("L")
    arr = np.array(img_gray)

    if clahe:
        arr = cv2.createCLAHE(2.0, (8,8)).apply(arr)

    if denoise:
        arr = cv2.bilateralFilter(arr, 9, 75, 75)

    if crop:
        x1,y1,x2,y2 = crop
        arr = arr[y1:y2, x1:x2]

    return Image.fromarray(arr).convert("RGB")

# ----------------- SECOND OPINION MODEL -----------------
def load_second_opinion_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = models.densenet121(pretrained=False)

    num_ftrs = model.classifier.in_features
    model.classifier = torch.nn.Linear(num_ftrs, 1)

    if os.path.exists(SECOND_OPINION_MODEL_PATH):
        model.load_state_dict(torch.load(SECOND_OPINION_MODEL_PATH, map_location=device))

    model.to(device).eval()
    return model, device

def predict_second_opinion(model, device, img):
    tfm = transforms.Compose([
        transforms.Resize((320,320)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485,0.456,0.406],
            [0.229,0.224,0.225]
        )
    ])
    x = tfm(img).unsqueeze(0).to(device)
    with torch.no_grad():
        prob = torch.sigmoid(model(x)).item()
    return prob, prob >= 0.5


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.layer = target_layer
        self.grad = None
        self.act = None

        def fw(m,i,o): self.act = o
        def bw(m,gi,go): self.grad = go[0]

        target_layer.register_forward_hook(fw)
        target_layer.register_backward_hook(bw)

    def generate(self, x):
        out = self.model(x)
        score = out[0][0]
        self.model.zero_grad()
        score.backward()

        pooled = torch.mean(self.grad, dim=[0,2,3])
        act = self.act.clone()

        for i in range(act.shape[1]):
            act[:,i,:,:] *= pooled[i]

        heatmap = act.mean(dim=1).squeeze().cpu().numpy()
        heatmap = np.maximum(heatmap,0)
        heatmap /= (heatmap.max() + 1e-6)
        return heatmap

# def overlay_heatmap(img, heatmap):
#     img = np.array(img.resize((heatmap.shape[1], heatmap.shape[0])))
#     colored = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
#     blend = cv2.addWeighted(img, 0.5, colored, 0.5, 0)
#     return Image.fromarray(blend)

# ----------------- GEMINI AGENT -----------------
def create_agent(use_duck=False):
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        return None

    tools = [DuckDuckGoTools()] if use_duck else []

    return Agent(
        model=Gemini(
            id="gemini-2.0-flash",  
            api_key=api_key
        ),
        tools=tools,
        markdown=True
    )


def run_gemini(agent, prompt, img):
    try:
        resp = agent.run(prompt, images=[img])
        return resp.content
    except Exception as e:
        return f"Gemini Error: {e}"

# ----------------- PDF EXPORT -----------------
def generate_pdf(path, text, imgs):
    c = canvas.Canvas(path, pagesize=letter)
    w,h = letter
    c.setFont("Helvetica-Bold",16)
    c.drawString(40,h-50,"Detailed Medical Report")

    y = h-90
    for img in imgs:
        try:
            c.drawImage(img, 40, y-120, 140,120)
            y -= 140
        except:
            pass

    text_obj = c.beginText(200, h-120)
    text_obj.setFont("Helvetica",10)
    for line in text.splitlines():
        text_obj.textLine(line)
    c.drawText(text_obj)
    c.save()
# ----------------- STREAMLIT UI -----------------

st.title("🖤 Medical Image Analyzer")

uploaded = st.sidebar.file_uploader("Upload medical image(s)", accept_multiple_files=True)

clahe = st.sidebar.checkbox("Apply CLAHE", True)
denoise = st.sidebar.checkbox("Denoise Image", True)
crop = st.sidebar.checkbox("Crop Image Manually")

coords = None
if crop:
    txt = st.sidebar.text_input("x1,y1,x2,y2")
    try:
        coords = tuple(map(int, txt.split(",")))
    except:
        coords = None

use_gemini = st.sidebar.checkbox("Use Gemini AI", True)
use_duck = st.sidebar.checkbox("Use DuckDuckGo Search", False)

tab1, tab2= st.tabs(["🔍 Analysis", "❓ VQA"])

# ----------------- Load Second Opinion Model -----------------
try:
    second_model, device = load_second_opinion_model()
except:
    second_model = None

# ----------------- ANALYSIS TAB -----------------
with tab1:
    if not uploaded:
        st.info("Upload images to analyze")
    else:
        images = []
        processed_paths = []

        for i,u in enumerate(uploaded):
            img = Image.open(u).convert("RGB")
            proc = preprocess_image_pil(img, clahe, denoise, coords)

            images.append(proc)

            tp = os.path.join(tempfile.gettempdir(), f"img_{i}.png")
            proc.save(tp)
            processed_paths.append(tp)

        cols = st.columns(len(images))
        for i,im in enumerate(images):
            cols[i].image(im, caption=f"Image {i+1}")

        if st.button("Run Analysis"):
            agent = create_agent(use_duck) if use_gemini else None

            final_report = []

            for i,img in enumerate(images):
                st.subheader(f"Image {i+1}")

                # ---------- Second Opinion ----------
                sec_prob, sec_pred = (None,None)
                heatmap_img = None

                if second_model:
                    sec_prob, sec_pred = predict_second_opinion(second_model, device, img)

                    try:
                        tfm = transforms.Compose([
                            transforms.Resize((320,320)),
                            transforms.ToTensor(),
                            transforms.Normalize(
                                [0.485,0.456,0.406],
                                [0.229,0.224,0.225]
                            )
                        ])
                        x = tfm(img).unsqueeze(0).to(device)
                        cam = GradCAM(second_model, second_model.features.norm5)
                        hm = cam.generate(x)
                        # heatmap_img = overlay_heatmap(img, hm)
                    except:
                        heatmap_img = None

                # ---------- Gemini ----------
                short = "Gemini disabled."
                full = "Gemini disabled."

                if agent:
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                    img.save(tmp.name)
                    ag = AgnoImage(filepath=tmp.name)
                    ANALYSIS_PROMPT = """
You are a radiology analysis assistant.

Provide a structured, concise medical image report with the following sections:

1. Modality & Region
Identify the imaging type and anatomical area.

2. Key Findings
Short bullet points describing visible abnormalities, patterns, densities, shapes, or issues.

3. Impression (Most Likely Interpretation)
Summarize what the findings most likely represent medically.

4. Patient-Friendly Explanation
Explain the findings in simple language but in detail.

Recommendation
"Please consult a licensed radiologist or physician for confirmation."

Keep the report professional, detailed, and avoid disclaimers about being an AI.
"""

                    short = run_gemini(agent, "Give a short 3 line medical summary explaining the image attached without any disclaimer in the end and also write as you are an Radiologist.", ag)
                    time.sleep(1.5)

                    full = run_gemini(agent, ANALYSIS_PROMPT, ag)
                    time.sleep(1.5)

                    tmp.close()
                    os.unlink(tmp.name)

                # ---------- SHOW RESULTS ----------
                st.markdown(f"""
                **Short Summary:**  
                {short}

                **Second Opinion Model:**  
                - Probability: **{sec_prob}**  
                - Prediction: **{sec_pred}**
                """)

                if heatmap_img:
                    st.image(heatmap_img, caption="Grad-CAM Heatmap")

                final_report.append(f"=== Image {i+1} ===\n{full}")

            # ---------- PDF EXPORT ----------
            pdf_path = os.path.join(tempfile.gettempdir(), "final_report.pdf")
            generate_pdf(pdf_path, "\n\n".join(final_report), processed_paths)

            with open(pdf_path, "rb") as f:
                st.download_button("Download Detailed PDF", f, file_name="medical_report.pdf")

# ----------------- VQA TAB -----------------
with tab2:
    if not uploaded:
        st.info("Upload an image first")
    else:
        question = st.text_input("Ask a question about the image:")

        if st.button("Ask Gemini"):
            if not use_gemini:
                st.warning("Gemini is disabled.")
            else:
                agent = create_agent(use_duck)
                img = images[0]

                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                img.save(tmp.name)
                ag = AgnoImage(filepath=tmp.name)

                VQA_PROMPT = f"""
You are a medical imaging expert. 
You are allowed to describe possible abnormalities, patterns, findings, shapes, densities, and medical interpretations in the image.

DO NOT refuse to answer medical questions.
DO NOT say "I am an AI model" or "I cannot diagnose."
Provide your best radiology-style interpretation based ONLY on the image.

If the user asks:
- "Do I have X disease?" 
- "Is this cancer/pneumonia/etc.?"
→ Answer with evidence-based imaging observations (e.g., “there are signs consistent with…”),
NOT with a refusal.

Structure your answer as:

### Medical Interpretation
(Direct radiological analysis)

### Notes
(State if features are consistent with diseases, but phrase as possibilities, not certainties.)

### Recommendation
Add this line ALWAYS:
"Please consult a licensed radiologist or physician for a definitive diagnosis."

"""

                ans = run_gemini(agent, VQA_PROMPT, ag)
   
                st.markdown("### Answer:")
                st.markdown(ans)

                tmp.close()
                os.unlink(tmp.name)

st.markdown("---")
st.warning("This tool is for research use only. Not for medical diagnosis.")

