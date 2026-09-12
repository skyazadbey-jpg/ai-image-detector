from fastapi import FastAPI, UploadFile, File, Form
import os
import requests

app = FastAPI()

API_URL = "https://router.huggingface.co/hf-inference/models/umm-maybe/AI-image-detector"
HF_TOKEN = os.getenv("HF_TOKEN")

@app.get("/")
def home():
    return {"status": "AI Image Detector API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(None), image_url: str = Form(None)):
    try:
        contents = None
        content_type = "application/octet-stream"

        if file:
            contents = await file.read()
            content_type = file.content_type or "application/octet-stream"
        elif image_url:
            # Tarayıcı engelini aşmak için resmi sunucu tarafında indiriyoruz
            img_response = requests.get(image_url)
            if img_response.status_code != 200:
                return {"error": "Görsel internet adresinden indirilemedi."}
            contents = img_response.content
            content_type = img_response.headers.get("content-type", "image/jpeg")
        else:
            return {"error": "Görsel bulunamadı."}

        headers = {
            "Authorization": f"Bearer {HF_TOKEN}",
            "Content-Type": content_type
        }
        
        response = requests.post(API_URL, headers=headers, data=contents)
        
        if response.status_code != 200:
            return {"error": f"Hugging Face API Error: {response.text}"}

        return {"result": response.json()}
    except Exception as e:
        return {"error": str(e)}