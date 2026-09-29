from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, EmailStr
import os
import sqlite3
import secrets
import string
import time
import jwt
import requests
import io
import re
import base64
import random
import asyncio
import concurrent.futures
from PIL import Image, ImageFilter, ImageDraw
from passlib.hash import bcrypt
from google.oauth2 import id_token as google_id_token
from google.auth.transport import requests as google_requests

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- CONFIG ----------
# ONEMLI: Butun gizli anahtarlar SADECE environment variable'dan okunur, koda asla sabit yazilmaz.
HF_TOKEN = os.getenv("HF_TOKEN")

# ---------- ENSEMBLE MODEL AYARLARI ----------
# Birden fazla modeli aynı anda calistirip sonuclari agirlikli olarak birlestiriyoruz.
# Format: "model_id:agirlik,model_id:agirlik,..."  (agirlik verilmezse 1.0 kabul edilir)
# Render'da AI_MODELS environment variable'ini degistirerek model listesini
# kod dokunmadan guncelleyebilirsiniz. Yeni bir model eklemeden once o modelin
# huggingface.co uzerinde "image-classification" gorevini destekledigini ve
# router.huggingface.co uzerinden erisilebilir oldugunu dogrulayin.
_DEFAULT_MODELS = "Vontra/detectra-v1:0.55,prithivMLmods/deepfake-detector-model-v1:0.30,umm-maybe/AI-image-detector:0.15"

def _parse_model_configs(raw: str):
    configs = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            model_id, weight_str = part.rsplit(":", 1)
            try:
                weight = float(weight_str)
            except ValueError:
                model_id, weight = part, 1.0
        else:
            model_id, weight = part, 1.0
        model_id = model_id.strip()
        if model_id:
            configs.append({
                "model_id": model_id,
                "url": f"https://router.huggingface.co/hf-inference/models/{model_id}",
                "weight": weight,
            })
    return configs


MODEL_CONFIGS = _parse_model_configs(os.getenv("AI_MODELS", _DEFAULT_MODELS))

# Karar esigi: ai_probability bu degerin ustundeyse "AI/FAKE" olarak isaretlenir.
# test_elsa setinizle calibrate_threshold.py calistirarak en iyi degeri bulabilirsiniz.
DECISION_THRESHOLD = float(os.getenv("DECISION_THRESHOLD", "50"))

RESEND_API_KEY = os.getenv("RESEND_API_KEY")
JWT_SECRET = os.getenv("JWT_SECRET", "change-me-in-production")
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
FROM_EMAIL = os.getenv("FROM_EMAIL", "onboarding@resend.dev")

DB_PATH = "users.db"


# ---------- DATABASE ----------
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT,
            google_id TEXT,
            is_verified INTEGER DEFAULT 0,
            is_pro INTEGER DEFAULT 0,
            credits INTEGER DEFAULT 3,
            otp_code TEXT,
            otp_expires REAL,
            created_at REAL
        )
        """
    )
    try:
        conn.execute("ALTER TABLE users ADD COLUMN is_pro INTEGER DEFAULT 0")
    except Exception:
        pass
    try:
        conn.execute("ALTER TABLE users ADD COLUMN credits INTEGER DEFAULT 3")
    except Exception:
        pass
    conn.commit()
    conn.close()


init_db()


# ---------- MODELS ----------
class RegisterRequest(BaseModel):
    email: EmailStr


class SetPasswordRequest(BaseModel):
    setup_token: str
    password: str


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class VerifyRequest(BaseModel):
    email: EmailStr
    otp: str


class ResendRequest(BaseModel):
    email: EmailStr


class ForgotPasswordRequest(BaseModel):
    email: EmailStr


class ResetPasswordRequest(BaseModel):
    reset_token: str
    new_password: str


class GoogleAuthRequest(BaseModel):
    id_token: str


# ---------- HELPERS ----------
def make_jwt(user_id: int, email: str) -> str:
    payload = {"user_id": user_id, "email": email, "exp": time.time() + 60 * 60 * 24 * 7}
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")


def make_setup_token(user_id: int, email: str) -> str:
    payload = {"user_id": user_id, "email": email, "scope": "set_password", "exp": time.time() + 60 * 15}
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")


def make_reset_token(user_id: int, email: str) -> str:
    payload = {"user_id": user_id, "email": email, "scope": "reset_password", "exp": time.time() + 60 * 30}
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")


def decode_jwt(token: str):
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
    except jwt.PyJWTError:
        return None


def get_current_user(authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Oturum bulunamadı.")
    token = authorization.split(" ", 1)[1]
    payload = decode_jwt(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Geçersiz veya süresi dolmuş oturum.")
    return payload


def generate_otp() -> str:
    return "".join(secrets.choice(string.digits) for _ in range(6))


def send_email(to_email: str, subject: str, html: str):
    if not RESEND_API_KEY:
        raise RuntimeError("RESEND_API_KEY sunucuda tanımlı değil.")
    resp = requests.post(
        "https://api.resend.com/emails",
        headers={
            "Authorization": f"Bearer {RESEND_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "from": f"AI Image Detector <{FROM_EMAIL}>",
            "to": [to_email],
            "subject": subject,
            "html": html,
        },
        timeout=15,
    )
    if resp.status_code >= 300:
        raise RuntimeError(f"Email gönderilemedi: {resp.text}")


def user_to_public(row) -> dict:
    is_pro = False
    try:
        is_pro = bool(row["is_pro"])
    except (IndexError, KeyError):
        is_pro = False
    return {"email": row["email"], "is_verified": bool(row["is_verified"]), "is_pro": is_pro}


# ---------- AUTH ROUTES ----------
@app.post("/auth/register")
def register(data: RegisterRequest):
    conn = get_db()
    existing = conn.execute("SELECT * FROM users WHERE email = ?", (data.email,)).fetchone()
    otp = generate_otp()
    otp_expires = time.time() + 60 * 10

    if existing and existing["is_verified"] and existing["password_hash"]:
        conn.close()
        raise HTTPException(status_code=400, detail="Bu email zaten kayıtlı. Giriş yapmayı deneyin.")

    if existing:
        conn.execute("UPDATE users SET otp_code=?, otp_expires=? WHERE email=?", (otp, otp_expires, data.email))
    else:
        conn.execute(
            "INSERT INTO users (email, otp_code, otp_expires, created_at) VALUES (?,?,?,?)",
            (data.email, otp, otp_expires, time.time()),
        )
    conn.commit()
    conn.close()

    try:
        send_email(
            data.email,
            "Doğrulama Kodunuz",
            f"<p>Merhaba,</p><p>Doğrulama kodunuz: <b style='font-size:24px'>{otp}</b></p><p>Bu kod 10 dakika geçerlidir.</p>",
        )
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"message": "Doğrulama kodu email adresinize gönderildi."}


@app.post("/auth/verify")
def verify(data: VerifyRequest):
    conn = get_db()
    row = conn.execute("SELECT * FROM users WHERE email = ?", (data.email,)).fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Kullanıcı bulunamadı.")
    if row["otp_code"] != data.otp:
        conn.close()
        raise HTTPException(status_code=400, detail="Hatalı doğrulama kodu.")
    if row["otp_expires"] and time.time() > row["otp_expires"]:
        conn.close()
        raise HTTPException(status_code=400, detail="Doğrulama kodunun süresi dolmuş, yeni kod isteyin.")

    conn.execute("UPDATE users SET is_verified=1, otp_code=NULL, otp_expires=NULL WHERE email=?", (data.email,))
    conn.commit()
    updated = conn.execute("SELECT * FROM users WHERE email=?", (data.email,)).fetchone()
    conn.close()

    setup_token = make_setup_token(updated["id"], updated["email"])
    return {"setup_token": setup_token, "email": updated["email"]}


@app.post("/auth/set-password")
def set_password(data: SetPasswordRequest):
    payload = decode_jwt(data.setup_token)
    if not payload or payload.get("scope") != "set_password":
        raise HTTPException(status_code=401, detail="Doğrulama süresi dolmuş, lütfen tekrar deneyin.")

    if len(data.password) < 6:
        raise HTTPException(status_code=400, detail="Şifre en az 6 karakter olmalı.")

    conn = get_db()
    conn.execute(
        "UPDATE users SET password_hash=? WHERE id=?",
        (bcrypt.hash(data.password), payload["user_id"]),
    )
    conn.commit()
    row = conn.execute("SELECT * FROM users WHERE id=?", (payload["user_id"],)).fetchone()
    conn.close()

    token = make_jwt(row["id"], row["email"])
    return {"token": token, "user": user_to_public(row)}


@app.post("/auth/resend-otp")
def resend_otp(data: ResendRequest):
    conn = get_db()
    row = conn.execute("SELECT * FROM users WHERE email=?", (data.email,)).fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Kullanıcı bulunamadı.")
    if row["is_verified"]:
        conn.close()
        raise HTTPException(status_code=400, detail="Hesap zaten doğrulanmış.")

    otp = generate_otp()
    otp_expires = time.time() + 60 * 10
    conn.execute("UPDATE users SET otp_code=?, otp_expires=? WHERE email=?", (otp, otp_expires, data.email))
    conn.commit()
    conn.close()

    try:
        send_email(data.email, "Yeni Doğrulama Kodunuz", f"<p>Yeni kodunuz: <b style='font-size:24px'>{otp}</b></p>")
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"message": "Yeni kod gönderildi."}


@app.post("/auth/login")
def login(data: LoginRequest):
    conn = get_db()
    row = conn.execute("SELECT * FROM users WHERE email=?", (data.email,)).fetchone()
    conn.close()

    if not row or not row["password_hash"] or not bcrypt.verify(data.password, row["password_hash"]):
        raise HTTPException(status_code=401, detail="Email veya şifre hatalı.")

    if not row["is_verified"]:
        raise HTTPException(status_code=403, detail="Hesabınız henüz doğrulanmamış.")

    token = make_jwt(row["id"], row["email"])
    return {"token": token, "user": user_to_public(row)}


@app.post("/auth/forgot-password")
def forgot_password(data: ForgotPasswordRequest, request: Request):
    conn = get_db()
    row = conn.execute("SELECT * FROM users WHERE LOWER(TRIM(email))=?", (data.email.strip().lower(),)).fetchone()

    generic_msg = "Eğer bu e-posta kayıtlıysa, sıfırlama linki gönderildi."

    if not row:
        conn.close()
        return {"message": generic_msg}

    if not row["password_hash"]:
        conn.close()
        return {"message": "Bu hesap Google ile oluşturulmuş. Lütfen Google ile giriş yapın."}

    reset_token = make_reset_token(row["id"], row["email"])
    conn.close()

    base_url = str(request.base_url).rstrip("/")
    reset_link = f"{base_url}/?reset_token={reset_token}"

    try:
        send_email(
            row["email"],
            "Şifre Sıfırlama - AI Image Detector",
            f"""
            <div style="font-family:sans-serif;max-width:500px;margin:auto">
                <h2>Şifre Sıfırlama</h2>
                <p>Merhaba,</p>
                <p>Şifrenizi sıfırlamak için aşağıdaki butona tıklayın:</p>
                <p style="text-align:center;margin:30px 0">
                    <a href="{reset_link}" style="background:#6366f1;color:white;padding:12px 24px;border-radius:8px;text-decoration:none;font-weight:bold">
                        Şifremi Sıfırla
                    </a>
                </p>
                <p style="color:#666;font-size:14px">Bu link 30 dakika geçerlidir.</p>
                <p style="color:#666;font-size:14px">Bu talebi siz yapmadıysanız, bu e-postayı görmezden gelebilirsiniz.</p>
            </div>
            """,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"message": generic_msg}


@app.post("/auth/reset-password")
def reset_password(data: ResetPasswordRequest):
    payload = decode_jwt(data.reset_token)
    if not payload or payload.get("scope") != "reset_password":
        raise HTTPException(status_code=401, detail="Geçersiz veya süresi dolmuş link.")

    if len(data.new_password) < 6:
        raise HTTPException(status_code=400, detail="Şifre en az 6 karakter olmalı.")

    conn = get_db()
    row = conn.execute("SELECT * FROM users WHERE id=?", (payload["user_id"],)).fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Kullanıcı bulunamadı.")

    conn.execute(
        "UPDATE users SET password_hash=? WHERE id=?",
        (bcrypt.hash(data.new_password), payload["user_id"]),
    )
    conn.commit()
    updated = conn.execute("SELECT * FROM users WHERE id=?", (payload["user_id"],)).fetchone()
    conn.close()

    token = make_jwt(updated["id"], updated["email"])
    return {"token": token, "user": user_to_public(updated), "message": "Şifreniz başarıyla güncellendi."}


@app.post("/auth/google")
def google_auth(data: GoogleAuthRequest):
    if not GOOGLE_CLIENT_ID:
        raise HTTPException(status_code=500, detail="Google Client ID sunucuda tanımlı değil.")
    try:
        idinfo = google_id_token.verify_oauth2_token(
            data.id_token, google_requests.Request(), GOOGLE_CLIENT_ID
        )
    except ValueError:
        raise HTTPException(status_code=401, detail="Geçersiz Google token.")

    email = idinfo["email"]
    google_id = idinfo["sub"]

    conn = get_db()
    row = conn.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
    if not row:
        conn.execute(
            "INSERT INTO users (email, google_id, is_verified, created_at) VALUES (?,?,1,?)",
            (email, google_id, time.time()),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
    elif not row["google_id"]:
        conn.execute("UPDATE users SET google_id=?, is_verified=1 WHERE email=?", (google_id, email))
        conn.commit()
        row = conn.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
    conn.close()

    token = make_jwt(row["id"], row["email"])
    return {"token": token, "user": user_to_public(row)}


@app.get("/user/credits")
def get_credits(current=Depends(get_current_user)):
    conn = get_db()
    row = conn.execute("SELECT credits, is_pro FROM users WHERE id=?", (current["user_id"],)).fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Kullanıcı bulunamadı.")
    return {"credits": row["credits"], "is_pro": bool(row["is_pro"])}


@app.post("/user/use-credit")
def use_credit(current=Depends(get_current_user)):
    conn = get_db()
    row = conn.execute("SELECT credits, is_pro FROM users WHERE id=?", (current["user_id"],)).fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Kullanıcı bulunamadı.")

    if row["is_pro"]:
        conn.close()
        return {"credits": 9999, "is_pro": True}

    if row["credits"] <= 0:
        conn.close()
        raise HTTPException(status_code=403, detail="Kredi tükendi. Pro'ya yükseltin.")

    new_credits = row["credits"] - 1
    conn.execute("UPDATE users SET credits=? WHERE id=?", (new_credits, current["user_id"]))
    conn.commit()
    conn.close()
    return {"credits": new_credits, "is_pro": False}


@app.get("/auth/me")
def me(current=Depends(get_current_user)):
    conn = get_db()
    row = conn.execute("SELECT * FROM users WHERE id=?", (current["user_id"],)).fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Kullanıcı bulunamadı.")
    return {"user": user_to_public(row), "is_pro": bool(row["is_pro"]) if "is_pro" in row.keys() else False}


# ---------- HOME ----------
@app.get("/api")
def api_home():
    return {"status": "AI Image Detector API is running"}
@app.get("/test-ateeqq-raw")
def test_ateeqq_raw():
    """GEÇİCİ TEST: Ateeqq modelinin ham cevabını gösterir."""
    import requests, os, json, io
    from PIL import Image
    try:
        # Kod içinde basit bir görsel oluştur (kırmızı kare)
        img = Image.new("RGB", (384, 384), color=(200, 100, 150))
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG", quality=90)
        img_bytes = buffered.getvalue()
        
        print(f"[TEST] Görsel boyutu: {len(img_bytes)} bytes")
        
        # Ateeqq modeline gönder
        url = "https://router.huggingface.co/hf-inference/models/Ateeqq/ai-vs-human-image-detector"
        headers = {"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "image/jpeg"}
        r = requests.post(url, headers=headers, data=img_bytes, timeout=30)
        
        print(f"[TEST] HF cevabı: {r.status_code}")
        print(f"[TEST] HF body: {r.text[:500]}")
        
        return {
            "status_code": r.status_code,
            "raw_response": r.json() if r.status_code == 200 else r.text,
            "image_size_bytes": len(img_bytes)
        }
    except Exception as e:
        return {"error": f"{type(e).__name__}: {str(e)}"}


# ---------- GUMROAD WEBHOOK ----------
@app.post("/webhooks/gumroad")
async def gumroad_webhook(request: Request):
    try:
        form_data = await request.form()
        data = dict(form_data)
        event_name = data.get("event_name", "sale")
        user_email = data.get("email")

        print(f"GUMROAD WEBHOOK: event={event_name} email={user_email}")

        conn = get_db()
        matched = False

        if user_email:
            normalized_email = user_email.strip().lower()
            row = conn.execute(
                "SELECT id FROM users WHERE LOWER(TRIM(email))=?", (normalized_email,)
            ).fetchone()
            if row:
                matched = True
                conn.execute("UPDATE users SET is_pro=1 WHERE id=?", (row["id"],))
                conn.commit()
                print(f"GUMROAD: email={normalized_email} eslesti, Pro yapildi.")

        conn.close()
        if not matched:
            print(f"GUMROAD UYARI: Hicbir kullanici eslesmedi. email={user_email}")

        return {"status": "ok", "matched": matched}
    except Exception as e:
        print(f"Gumroad Webhook Hatasi: {e}")
        return {"status": "error", "message": str(e)}


# ---------- PREDICT (tek ve doğru versiyon) ----------
def normalize_image_bytes(contents: bytes):
    """Gelen görseli her zaman JPEG'e çevirir, model bazı formatlarda hata verebiliyor."""
    try:
        image = Image.open(io.BytesIO(contents))
        if image.mode in ("RGBA", "P"):
            image = image.convert("RGB")
        output_buffer = io.BytesIO()
        image.save(output_buffer, format="JPEG")
        return output_buffer.getvalue(), "image/jpeg"
    except Exception:
        return contents, "application/octet-stream"


CATEGORY_CHECK_MODEL = "google/vit-base-patch16-224"

# ImageNet-1k etiketlerinden arabayla ilgili olanlar
CAR_KEYWORDS = [
    "car", "convertible", "cab", "jeep", "limousine", "minivan", "pickup",
    "racer", "beach wagon", "police van", "sports car", "go-kart", "golfcart",
    "moving van", "recreational vehicle", "wagon",
]
# ImageNet-1k etiketlerinden bina/emlakla ilgili olanlar
BUILDING_KEYWORDS = [
    "house", "home", "building", "boathouse", "barn", "castle", "church",
    "dam", "dome", "greenhouse", "lighthouse", "mobile home", "monastery",
    "mosque", "palace", "patio", "picket fence", "planetarium", "stupa",
    "yurt", "residential", "estate", "cottage", "mansion", "villa",
]


def classify_image_category(contents: bytes, content_type: str = "image/jpeg") -> str:
    """Görselin genel olarak ne olduğunu (araba / emlak-bina / diğer) tahmin eder.
    'Araba' ve 'Emlak' moduna özel sayfalarda, o kategoriyle alakasız bir görsel
    yüklenip analiz edilmesini engellemek için kullanılır.

    Zero-shot modeller (CLIP vb.) HuggingFace'in ücretsiz altyapısında artık
    desteklenmiyor, o yüzden standart bir ImageNet siniflandiricisi kullanip
    en olası 5 tahminin arabayla/binayla alakalı bir anahtar kelime içerip
    içermediğine bakıyoruz. Üst 5 tahminin HİÇBİRİ eşleşmezse "other" döner.
    API'ye ulaşılamazsa "unknown" döner - bu durumda kullanıcıyı YANLIŞLIKLA
    engellememek için analiz normal şekilde devam eder."""
    try:
        headers = {"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": content_type}
        url = f"https://router.huggingface.co/hf-inference/models/{CATEGORY_CHECK_MODEL}"
        response = requests.post(url, headers=headers, data=contents, timeout=20)
        if response.status_code != 200:
            print(f"[kategori kontrolü] HF API hatası: {response.status_code} - {response.text[:200]}")
            return "unknown"

        result = response.json()
        if isinstance(result, list) and result:
            if isinstance(result[0], list):
                result = result[0]
            top5 = sorted(result, key=lambda r: r.get("score", 0), reverse=True)[:5]
            labels_combined = " | ".join(str(r.get("label", "")).lower() for r in top5)

            def has_keyword(keywords, text):
                # Tam kelime eslesmesi ariyoruz - basit "in" kontrolu "car"
                # kelimesini "cardigan" gibi alakasiz kelimelerin icinde de
                # buluyordu, bu yuzden kelime sinirlarina (\b) dikkat ediyoruz.
                return any(re.search(rf"\b{re.escape(kw)}\b", text) for kw in keywords)

            if has_keyword(CAR_KEYWORDS, labels_combined):
                return "car"
            if has_keyword(BUILDING_KEYWORDS, labels_combined):
                return "realestate"
            return "other"
    except Exception as e:
        print(f"[kategori kontrolü] hata: {e}")
    return "unknown"


def _call_single_model(model_config: dict, contents: bytes, content_type: str):
    """Tek bir Hugging Face modelini çağırır. Başarısız olursa None döner
    (None dönmesi ensemble ortalamasını bozmasın diye önemli - modelin
    "bilmiyorum" demesi 50 puan olarak sayılmamalı)."""
    headers = {"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": content_type}
    try:
        response = requests.post(model_config["url"], headers=headers, data=contents, timeout=30)
    except Exception as e:
        print(f"[{model_config['model_id']}] Bağlantı hatası: {e}")
        return None

    if response.status_code != 200:
        print(f"[{model_config['model_id']}] HF API hatası: {response.status_code} - {response.text[:200]}")
        return None

    try:
        result = response.json()
        if isinstance(result, list) and len(result) > 0:
            if isinstance(result[0], list):
                result = result[0]
            for item in result:
                label = str(item.get("label", "")).lower()
                score = float(item.get("score", 0.5))
                if any(k in label for k in ["artificial", "fake", "ai", "generated", "deepfake", "label_1"]):
                    return score * 100
                elif any(k in label for k in ["hum", "human", "real", "authentic", "natural", "label_0"]):
                    return (1.0 - score) * 100
    except Exception as e:
        print(f"[{model_config['model_id']}] Cevap çözümlenemedi: {e}")

    return None


async def call_ai_model_ensemble(contents: bytes, content_type: str):
    """Tüm modelleri PARALEL çağırır, sonuçları ağırlıklı ortalamayla birleştirir.
    Dönüş: (final_ai_score, model_breakdown_listesi)
    """
    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(len(MODEL_CONFIGS), 1)) as executor:
        futures = [
            loop.run_in_executor(executor, _call_single_model, cfg, contents, content_type)
            for cfg in MODEL_CONFIGS
        ]
        raw_scores = await asyncio.gather(*futures)

    breakdown = []
    weighted_sum = 0.0
    weight_total = 0.0
    for cfg, score in zip(MODEL_CONFIGS, raw_scores):
        if score is None:
            breakdown.append({"model": cfg["model_id"], "ai_score": None, "status": "failed"})
            continue
        breakdown.append({"model": cfg["model_id"], "ai_score": round(score, 1), "weight": cfg["weight"]})
        weighted_sum += score * cfg["weight"]
        weight_total += cfg["weight"]

    if weight_total == 0:
        # Hiçbir model cevap vermediyse en güvenli varsayım: kararsız (50)
        return 50.0, breakdown

    final_ai_score = weighted_sum / weight_total
    return final_ai_score, breakdown


@app.post("/predict")
async def predict(file: UploadFile = File(None), image_url: str = Form(None), mode: str = Form("general")):
    try:
        contents = None
        content_type = "application/octet-stream"

        if file:
            contents = await file.read()
            contents, content_type = normalize_image_bytes(contents)
        elif image_url:
            img_response = requests.get(image_url, timeout=20)
            if img_response.status_code != 200:
                return {"error": "Görsel indirilemedi."}
            contents = img_response.content
            content_type = img_response.headers.get("content-type", "image/jpeg")
        else:
            return {"error": "Görsel bulunamadı."}

        # ---------- KATEGORI DOGRULAMASI (araba / emlak modlarinda) ----------
        # "Araba" veya "Emlak" sekmesinden yuklenen gorselin gercekten o
        # kategoriyle ilgili olup olmadigini kontrol ediyoruz. Alakasiz bir
        # gorsel (ornegin araba sekmesine manzara fotografi) yuklenirse
        # kullaniciya bunu bildirip analiz yapmadan durduruyoruz - bu da
        # sitenin guvenilirligini korur.
        if mode in ("car", "realestate"):
            category = classify_image_category(contents, content_type)
            if category == "other":
                if mode == "car":
                    return {"error": "Bu görsel bir araba fotoğrafına benzemiyor. Lütfen incelemek istediğiniz aracın net bir fotoğrafını yükleyin."}
                else:
                    return {"error": "Bu görsel bir emlak/bina fotoğrafına benzemiyor. Lütfen incelemek istediğiniz mülkün net bir fotoğrafını yükleyin."}
            # category == "unknown" ise (API'ye ulasilamadi vb.) kullaniciyi
            # yanlislikla engellememek icin analiz normal sekilde devam eder.

        final_ai_score, breakdown = await call_ai_model_ensemble(contents, content_type)
        final_real_score = 100.0 - final_ai_score
        verdict = "AI" if final_ai_score > DECISION_THRESHOLD else "REAL"

        # Not: hem yeni (ai_probability) hem eski (result dizisi) formatı birlikte
        # döndürüyoruz ki frontend hangi sürümde olursa olsun doğru okuyabilsin.
        return {
            "ai_probability": round(final_ai_score, 1),
            "real_probability": round(final_real_score, 1),
            "verdict": verdict,
            "status": "success",
            "result": [
                {"label": "artificial", "score": final_ai_score / 100},
                {"label": "human", "score": final_real_score / 100},
            ],
            "model_breakdown": breakdown,
        }
    except Exception as e:
        return {"error": str(e)}


# ---------- HEATMAP (eksik olan endpoint geri eklendi) ----------
@app.post("/predict-heatmap")
async def predict_heatmap(file: UploadFile = File(None), image_url: str = Form(None)):
    try:
        contents = None
        if file:
            contents = await file.read()
        elif image_url:
            img_response = requests.get(image_url, timeout=20)
            if img_response.status_code != 200:
                return {"error": "Görsel indirilemedi."}
            contents = img_response.content
        else:
            return {"error": "Görsel bulunamadı."}

        normalized_contents, content_type = normalize_image_bytes(contents)
        ai_score_pct, _breakdown = await call_ai_model_ensemble(normalized_contents, content_type)
        ai_score = ai_score_pct / 100.0

        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img = img.resize((512, 512))
        width, height = img.size

        # ---------- GERCEK IZGARA TABANLI ANALIZ ----------
        # Gorseli bir izgaraya bolup HER PARCAYI gercekten modele gonderiyoruz.
        # Boylece renklendirme rastgele degil, modelin o bolge icin verdigi
        # gercek AI-olma skoruna dayanir. Izgara boyutunu buyutmek daha
        # detayli ama daha yavas/maliyetli bir analiz anlamina gelir.
        grid_size = int(os.getenv("HEATMAP_GRID_SIZE", "3"))
        tile_w = width // grid_size
        tile_h = height // grid_size

        async def score_tile(row, col):
            box = (col * tile_w, row * tile_h, (col + 1) * tile_w, (row + 1) * tile_h)
            tile = img.crop(box)
            tile_buffer = io.BytesIO()
            tile.save(tile_buffer, format="JPEG")
            tile_score, _ = await call_ai_model_ensemble(tile_buffer.getvalue(), "image/jpeg")
            return row, col, tile_score

        tile_tasks = [score_tile(r, c) for r in range(grid_size) for c in range(grid_size)]
        tile_results = await asyncio.gather(*tile_tasks)

        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        for row, col, tile_score in tile_results:
            # tile_score 0-100 arasi: yuksek = AI supheli (kirmizi), dusuk = gercek (mavi)
            red = int(min(255, tile_score * 2.55))
            blue = int(min(255, (100 - tile_score) * 2.55))
            # modelin ne kadar "emin" oldugu (50'den uzaklik) opakligi belirler -
            # kararsiz (50'ye yakin) bolgeler daha soluk gorunur
            confidence = abs(tile_score - 50) / 50
            alpha = int(50 + confidence * 130)

            box = (col * tile_w, row * tile_h, (col + 1) * tile_w, (row + 1) * tile_h)
            draw.rectangle(box, fill=(red, 40, blue, alpha))

        overlay = overlay.filter(ImageFilter.GaussianBlur(radius=20))
        combined = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")

        buffered = io.BytesIO()
        combined.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return {
            "heatmap_base64": f"data:image/png;base64,{img_base64}",
            "ai_score": round(ai_score_pct, 1),
            "human_score": round(100 - ai_score_pct, 1),
        }
    except Exception as e:
        return {"error": str(e)}


# ---------- STATIC FILES ----------
@app.get("/")
def serve_index():
    return FileResponse("index.html")


@app.get("/privacy")
def serve_privacy():
    return FileResponse("privacy.html")


@app.get("/privacy.html")
def serve_privacy_html():
    return FileResponse("privacy.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)