# 🩺 Medical Image Analysis System (X-Ray, CT, MRI)  
### **AI-Powered Report Generation, VQA, Grad-CAM, & Second-Opinion Diagnosis**

A multimodal AI system that performs:

- **Medical Image Report Generation** (using Gemini)
- **Visual Question Answering (VQA)**
- **Second-Opinion Disease Classification** (DenseNet121)
- **Explainability with Grad-CAM**
- **PDF Export of Radiology Reports**
- **Advanced Preprocessing (CLAHE, Denoise, Cropping)**

Built with **Python, Streamlit, Gemini API, PyTorch, and OpenCV**.

---

# 🔥 Demo Screenshots

### 🖼️ **Image Upload + Preprocessing**
<img width="1819" height="765" alt="image" src="https://github.com/user-attachments/assets/6cc8363f-2307-4cbb-bf81-73abefb4c344" />


<img width="1503" height="745" alt="image" src="https://github.com/user-attachments/assets/1ae816dc-83f8-4310-a8ad-89dfe4a1ee28" />

### 🔍 **Short Summary + Detailed Report**

<img width="1691" height="783" alt="image" src="https://github.com/user-attachments/assets/95cb6463-6682-46d6-9285-3c7c7d6c38c9" />


### ❓ **Visual Question Answering (VQA)**

<img width="1747" height="718" alt="image" src="https://github.com/user-attachments/assets/62f8c6ff-456c-4815-bbf5-46685c7262b8" />


### 📄 **PDF Export**

<img width="998" height="697" alt="image" src="https://github.com/user-attachments/assets/b3c9167d-898c-4b3e-a638-5d288d890850" />


<img width="998" height="697" alt="image" src="https://github.com/user-attachments/assets/149eb708-95c8-4298-853d-f22a98065d05" />


---

# ⭐ Features

## 🔍 1. **Medical Image Analysis**
- Detects modality (X-ray, CT, MRI, Ultrasound)
- Identifies abnormalities
- Produces structured radiology reports:
  - Image Type & View
  - Key Findings
  - Impression
  - Patient-Friendly Explanation
  - Clinical Recommendation

---

## ❓ 2. **Visual Question Answering (VQA)**
Ask any question about the image:

> “Is the heart enlarged?”  
> “Does the X-ray show pneumonia?”  
> “Is the trachea shifted?”

AI understands the **image + question** and responds accordingly.

---

## 🧠 3. **Second-Opinion Model (DenseNet121)**
A locally loaded ML model provides:

- Probability score (0–1)
- Normal vs Abnormal prediction
- Works offline (no API needed)

---

## 🔥 4. **Explainable AI with Grad-CAM**
Shows which parts of the image the model used for prediction.

Used for:
- Transparency  
- Debugging  
- Medical reliability  

---

## 🛠 5. **Smart Preprocessing**
- **CLAHE** (Contrast Limited Adaptive Histogram Equalization)
- **Denoising** (bilateral filter)
- **Manual Cropping**

Enhances medical images for better AI interpretation.

---

## 📄 6. **PDF Export**
AI-generated report exported as a clean PDF with:

- Images  
- Findings  
- Impression   

---

# 🧱 Tech Stack

| Component | Technology |
|----------|------------|
| Frontend | Streamlit |
| Backend | Python |
| AI Model | Google Gemini |
| Classifier | PyTorch DenseNet121 |
| Explainability | Grad-CAM |
| Image Processing | OpenCV, PIL |
| PDF Export | ReportLab |
| Environment | python-dotenv |

---


