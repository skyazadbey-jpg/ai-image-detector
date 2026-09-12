from fastapi import FastAPI, UploadFile, File
import os
import requests

app = FastAPI()

API_URL = "https://router.huggingface.co/hf-inference/models/umm-maybe/AI-image-detector"
HF_TOKEN = os.getenv("HF_TOKEN")

@app.get("/")
def home():
    return {"status": "AI Image Detector API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        
        # Eklentiden gelen dosyanın kendi türünü (content-type) dinamik olarak alıyoruz
        content_type = file.content_type or "application/octet-stream"
        
        headers = {
            "Authorization": f"Bearer {HF_TOKEN}",
            "Content-Type": content_type
        }
        
        response = requests.post(API_URL, headers=headers, data=contents)
        
        if response.status_code != 200:
            return {"error": f"Hugging Face API Error: {response.text}"}

        result = response.json()
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}