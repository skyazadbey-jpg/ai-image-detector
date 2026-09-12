from fastapi import FastAPI, File, UploadFile, HTTPException
from transformers import pipeline
from PIL import Image
import io

app = FastAPI()

# Modelimizi yüklüyoruz
print("Model yükleniyor...")
pipe = pipeline("image-classification", model="umm-maybe/AI-image-detector")
print("Model başarıyla yüklendi!")

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        results = pipe(image)
        
        fake_score = 0.0
        for res in results:
            label = res['label'].lower()
            if "fake" in label or "ai" in label or "artificial" in label:
                fake_score = res['score']
                break
        else:
            if len(results) > 0:
                # Eğer etiket doğrudan 'artificial' veya benzeri değilse ilk skoru baz alalım
                sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)
                top_label = sorted_results[0]['label'].lower()
                if "human" in top_label or "real" in top_label:
                    fake_score = 1.0 - sorted_results[0]['score']
                else:
                    fake_score = sorted_results[0]['score']

        fake_probability = round(fake_score * 100, 2)
        return {"fake_probability": fake_probability}

    except Exception as e:
        return {"error": str(e)}

# Gradio arayüzü entegrasyonu (Hugging Face Space'in ücretsiz ayağa kalkması için şarttır)
import gradio as gr
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def predict_ui(image):
    # Görseli işleyip yüzde döndüren arayüz fonksiyonu
    results = pipe(image)
    fake_score = 0.0
    for res in results:
        label = res['label'].lower()
        if "fake" in label or "ai" in label:
            fake_score = res['score']
            break
    else:
        if len(results) > 0:
            fake_score = sorted(results, key=lambda x: x['score'], reverse=True)[0]['score']
    return f"Yapay Zeka Üretimi Olma İhtimali: %{round(fake_score * 100, 2)}"

demo = gr.Interface(
    fn=predict_ui, 
    inputs=gr.Image(type="pil"), 
    outputs="text",
    title="AI Görsel Dedektörü",
    description="Görselin yapay zeka ürünü olup olmadığını test edin."
)

app = gr.mount_gradio_app(app, demo, path="/")