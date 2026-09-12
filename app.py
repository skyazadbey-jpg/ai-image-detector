from fastapi import FastAPI, UploadFile, File
import requests
import io
import os

app = FastAPI()

API_URL = "https://router.huggingface.co/hf-inference/models/umm-maybe/AI-image-detector"

# Token'ı doğrudan açık yazmak yerine Render'ın Environment Variables alanından güvenle çekiyoruz
HF_TOKEN = os.getenv("HF_TOKEN")
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

@app.get("/")
def home():
    return {"status": "AI Image Detector API is running (Lightweight Mode)"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        response = requests.post(API_URL, headers=HEADERS, data=contents)
        
        if response.status_code != 200:
            return {"error": f"Hugging Face API Error: {response.text}"}

        result = response.json()
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}