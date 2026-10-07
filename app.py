from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
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
import hashlib
import concurrent.futures
import math
import json
from evidence_report import build_report
from detector_core import parse_prediction, decide_verdict
import local_detector
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
_DEFAULT_MODELS = "Ateeqq/ai-vs-human-image-detector:1.0"

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
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError("AI_MODELS ağırlıkları pozitif ve sonlu olmalı")
        if model_id:
            configs.append({
                "model_id": model_id,
                "url": f"https://router.huggingface.co/hf-inference/models/{model_id}",
                "weight": weight,
            })
    return configs


AI_BACKEND = os.getenv("AI_BACKEND", "local").strip().lower()
if AI_BACKEND not in ("local", "hf"):
    raise ValueError("AI_BACKEND local veya hf olmalı")
MODEL_CONFIGS = ([{"model_id": local_detector.MODEL_ID, "weight": 1.0}]
                 if AI_BACKEND == "local" else _parse_model_configs(os.getenv("AI_MODELS", _DEFAULT_MODELS)))
ANALYSIS_GATE = asyncio.Lock()

# Karar esigi: ai_probability bu degerin ustundeyse "AI/FAKE" olarak isaretlenir.
# Eşik bağımsız ve doğrulanmış bir test setiyle ayarlanmalıdır.
DECISION_THRESHOLD = float(os.getenv("DECISION_THRESHOLD", "60"))
UNCERTAINTY_MARGIN = float(os.getenv("UNCERTAINTY_MARGIN", "10"))
if not 0 <= UNCERTAINTY_MARGIN < min(DECISION_THRESHOLD, 100 - DECISION_THRESHOLD):
    raise ValueError("Karar eşiği / belirsizlik aralığı geçersiz")

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

def init_feedback_table():
    """Feedback (kullanıcı geri bildirimi) tablosu."""
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_hash TEXT NOT NULL,
            ai_score REAL NOT NULL,
            predicted TEXT NOT NULL,
            actual TEXT NOT NULL,
            mode TEXT DEFAULT 'general',
            created_at REAL NOT NULL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_hash ON feedback(image_hash)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_created ON feedback(created_at)")
    conn.commit()
    conn.close()


init_feedback_table()

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
# ---------- FEEDBACK ----------
class FeedbackRequest(BaseModel):
    image_hash: str
    ai_score: float
    predicted: str
    actual: str
    mode: str = "general"
    analysis_token: str = ""


@app.post("/feedback")
def submit_feedback(data: FeedbackRequest):
    """Store self-reported labels only; never train from these automatically."""
    payload = decode_jwt(data.analysis_token)
    if not payload or payload.get("scope") != "analysis_feedback":
        raise HTTPException(status_code=400, detail="Önce yeniden analiz yapın.")
    if data.actual not in ("AI", "REAL", "UNCERTAIN"):
        raise HTTPException(status_code=400, detail="Geçersiz geri bildirim.")
    # Bind claims to the server-issued analysis; client cannot change score/prediction.
    data.image_hash = payload["image_hash"]
    data.ai_score = payload["score"]
    data.predicted = payload["verdict"]
    data.mode = payload["mode"]
    conn = get_db()
    duplicate = conn.execute("SELECT 1 FROM feedback WHERE image_hash=? AND predicted=? AND ai_score=?", (data.image_hash, data.predicted, data.ai_score)).fetchone()
    if duplicate:
        conn.close()
        return {"status": "ok", "message": "Bu analiz için geri bildirim zaten kaydedildi."}
    conn.execute(
        "INSERT INTO feedback (image_hash, ai_score, predicted, actual, mode, created_at) VALUES (?,?,?,?,?,?)",
        (data.image_hash, data.ai_score, data.predicted, data.actual, data.mode, time.time())
    )
    conn.commit()
    conn.close()
    print(f"[FEEDBACK] hash={data.image_hash} predicted={data.predicted} actual={data.actual} score={data.ai_score}")
    return {"status": "ok", "message": "Geri bildiriminiz kaydedildi, teşekkürler!"}


@app.get("/feedback/stats")
def feedback_stats():
    """Feedback istatistikleri (haftalık analiz için)."""
    conn = get_db()
    total = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
    if total == 0:
        conn.close()
        return {"total": 0, "self_reported_agreement": None, "verified_accuracy": None, "by_mode": []}
    
    correct = conn.execute(
        "SELECT COUNT(*) FROM feedback WHERE predicted = actual"
    ).fetchone()[0]
    
    modes = conn.execute(
        "SELECT mode, COUNT(*) as total, SUM(CASE WHEN predicted = actual THEN 1 ELSE 0 END) as correct FROM feedback GROUP BY mode"
    ).fetchall()
    
    conn.close()
    return {
        "total": total,
        "correct": correct,
        "wrong": total - correct,
        "self_reported_agreement": round(correct / total * 100, 1),
        "verified_accuracy": None,
        "note": "Kullanıcı beyanları doğrulanmış test seti değildir.",
        "by_mode": [dict(m) for m in modes]
    }
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
def download_analysis_image(url):
    if not url.lower().startswith(("http://", "https://")):
        raise ValueError("Geçerli bir HTTP/HTTPS görsel bağlantısı kullanın.")
    with requests.get(url, timeout=20, stream=True) as response:
        response.raise_for_status()
        result = io.BytesIO()
        for chunk in response.iter_content(64 * 1024):
            if result.tell() + len(chunk) > 20 * 1024 * 1024:
                raise ValueError("Görsel 20 MB sınırını aşıyor.")
            result.write(chunk)
        return result.getvalue(), response.headers.get("content-type", "image/jpeg")


def normalize_image_bytes(contents: bytes):
    """JPEG/PNG bytes are preserved; other formats use lossless PNG."""
    if not contents or len(contents) > 20 * 1024 * 1024:
        raise ValueError("Görsel boş veya 20 MB sınırını aşıyor.")
    with Image.open(io.BytesIO(contents)) as image:
        if image.width * image.height > 40_000_000:
            raise ValueError("Görsel 40 megapiksel sınırını aşıyor.")
        if AI_BACKEND == "local":
            if image.width * image.height > local_detector.MAX_PIXELS:
                raise ValueError("Görsel 20 megapiksel sınırını aşıyor. Daha küçük bir kopya yükleyin.")
            content_type = Image.MIME.get(image.format, "application/octet-stream")
            image.verify()
            return contents, content_type
        image.load()
        if image.format in ("JPEG", "PNG"):
            return contents, "image/jpeg" if image.format == "JPEG" else "image/png"
        output = io.BytesIO()
        image.convert("RGB").save(output, format="PNG")
        return output.getvalue(), "image/png"


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

def compute_image_hash(contents: bytes) -> str:
    """Görselin SHA256 hash'ini hesaplar (feedback için)."""
    return hashlib.sha256(contents).hexdigest()[:16]

def check_exif(contents: bytes) -> tuple:
    """
    EXIF (kamera meta verisi) analizi.
    Gerçek fotoğraflar: Telefon/kamera modeli, ISO, diyafram bilgisi içerir.
    AI görselleri: Genellikle EXIF yoktur veya çok azdır.
    Dönüş: (gerçek_olma_skoru 0-100, sebep)
    """
    try:
        from PIL import Image
        from PIL.ExifTags import TAGS
        import io

        img = Image.open(io.BytesIO(contents))
        exif_data = img.getexif()

        if not exif_data:
            return 15.0, "EXIF verisi YOK (AI olma ihtimali yüksek)"

        tag_names = []
        has_camera_make = False
        has_camera_model = False
        has_iso = False
        has_datetime = False
        has_lens = False

        for tag_id, value in exif_data.items():
            tag_name = TAGS.get(tag_id, tag_id)
            tag_names.append(str(tag_name))

            if tag_name == "Make":
                has_camera_make = True
            elif tag_name == "Model":
                has_camera_model = True
            elif tag_name == "ISOSpeedRatings":
                has_iso = True
            elif tag_name in ("DateTime", "DateTimeOriginal"):
                has_datetime = True
            elif tag_name in ("LensModel", "LensMake"):
                has_lens = True

        score = 20.0
        reasons = []

        if has_camera_make:
            score += 25
            reasons.append("kamera markası var")
        if has_camera_model:
            score += 20
            reasons.append("kamera modeli var")
        if has_iso:
            score += 10
            reasons.append("ISO var")
        if has_datetime:
            score += 10
            reasons.append("tarih var")
        if has_lens:
            score += 10
            reasons.append("lens bilgisi var")

        if len(tag_names) >= 15:
            score += 5
            reasons.append("zengin metadata")

        score = min(score, 100)
        reason = " | ".join(reasons) if reasons else "sadece temel EXIF"

        return score, reason
    except Exception as e:
        return 15.0, f"EXIF okunamadı: {str(e)[:50]}"

def check_plate_region(contents: bytes) -> tuple:
    """
    Araba fotoğraflarında plaka bölgesi kontrolü.
    
    Basit yaklaşım: Arabanın alt-orta bölgesini kes, 
    oradaki piksel yoğunluğunu ve deseni analiz et.
    
    AI üretimi plakalar: genellikle bulanık veya anlamsız desen.
    Gerçek plakalar: net, düzgün dikdörtgen, yazı var.
    
    Dönüş: (gerçek_olma_skoru 0-100, sebep)
    """
    try:
        import numpy as np
        from PIL import Image, ImageFilter
        import io
        
        img = Image.open(io.BytesIO(contents)).convert("L")
        w, h = img.size
        img = img.resize((512, 512), Image.LANCZOS)
        
        # Plaka genellikle alt-orta bölgede (%60-90 yükseklik, %30-70 genişlik)
        top = int(512 * 0.60)
        bottom = int(512 * 0.90)
        left = int(512 * 0.30)
        right = int(512 * 0.70)
        
        plate_region = img.crop((left, top, right, bottom))
        
        # Kenar yoğunluğu (yazı varsa yüksek olur)
        edges = plate_region.filter(ImageFilter.FIND_EDGES)
        edge_array = np.array(edges, dtype=np.float32)
        edge_variance = float(edge_array.var())
        
        # Kontrast (plaka beyaz, yazı siyah)
        region_array = np.array(plate_region, dtype=np.float32)
        contrast = float(region_array.std())
        
        # Puanlama
        # Gerçek plaka: yüksek kontrast + yüksek kenar yoğunluğu
        if edge_variance > 400 and contrast > 50:
            score = 80.0
            reason = f"Belirgin plaka bölgesi (edge={edge_variance:.0f}, contrast={contrast:.0f})"
        elif edge_variance > 200 and contrast > 35:
            score = 65.0
            reason = f"Muhtemel plaka (edge={edge_variance:.0f}, contrast={contrast:.0f})"
        elif edge_variance > 100:
            score = 50.0
            reason = f"Zayıf plaka bölgesi (edge={edge_variance:.0f})"
        else:
            score = 30.0
            reason = f"Plaka yok/bulanık (edge={edge_variance:.0f}) - AI olabilir"
        
        return score, reason
    except Exception as e:
        return 50.0, f"Plaka analizi hatası: {str(e)[:50]}"
    
def check_blur(contents: bytes) -> tuple:
    """
    Bulanıklık/keskinlik analizi.
    
    Dolandırıcılar genellikle bulanık screenshot kullanır.
    Ayrıca AI görselleri bazen aşırı pürüzsüz/bulanık olur.
    
    Dönüş: (gerçek_olma_skoru 0-100, sebep)
    """
    try:
        import numpy as np
        from PIL import Image, ImageFilter
        import io
        
        img = Image.open(io.BytesIO(contents)).convert("L")
        img = img.resize((512, 512), Image.LANCZOS)
        
        # Laplacian varyansı = keskinlik ölçüsü
        laplacian = img.filter(ImageFilter.FIND_EDGES)
        lap_array = np.array(laplacian, dtype=np.float32)
        variance = float(lap_array.var())
        
        # Puanlama
        if variance > 800:
            score = 85.0
            reason = f"Çok keskin görsel (variance={variance:.0f}) - gerçek fotoğraf"
        elif variance > 400:
            score = 70.0
            reason = f"Keskin görsel (variance={variance:.0f})"
        elif variance > 200:
            score = 50.0
            reason = f"Normal keskinlik (variance={variance:.0f})"
        elif variance > 100:
            score = 35.0
            reason = f"Bulanık görsel (variance={variance:.0f}) - şüpheli"
        else:
            score = 20.0
            reason = f"Çok bulanık (variance={variance:.0f}) - screenshot olabilir"
        
        return score, reason
    except Exception as e:
        return 50.0, f"Bulanıklık analizi hatası: {str(e)[:50]}"
    
def check_jpeg_blocks(contents: bytes) -> tuple:
    """
    JPEG 8x8 blok kalıntı analizi.
    
    Gerçek fotoğraflar: Kameradan gelen JPEG → düzgün blok pattern.
    AI görselleri: Yapay üretilmiş → düzensiz blok pattern veya çok pürüzsüz.
    
    Dönüş: (gerçek_olma_skoru 0-100, sebep)
    """
    try:
        import numpy as np
        from PIL import Image
        import io
        
        img = Image.open(io.BytesIO(contents)).convert("L")  # Gri tonlama
        img = img.resize((256, 256), Image.LANCZOS)
        arr = np.array(img, dtype=np.float32)
        
        # 8x8 blokların sınırlarındaki farklılıkları ölç
        # JPEG sıkıştırması 8x8 bloklarda "blocking artifacts" yaratır
        # Gerçek JPEG'lerde bu belirgin, AI görsellerinde zayıf
        
        h, w = arr.shape
        
        # Yatay blok sınırları (her 8. sütun)
        vertical_diffs = []
        for x in range(8, w, 8):
            diff = np.abs(arr[:, x-1] - arr[:, x]).mean()
            vertical_diffs.append(diff)
        
        # Dikey blok sınırları (her 8. satır)
        horizontal_diffs = []
        for y in range(8, h, 8):
            diff = np.abs(arr[y-1, :] - arr[y, :]).mean()
            horizontal_diffs.append(diff)
        
        # Blok sınırlarının ortalaması
        block_energy = np.mean(vertical_diffs + horizontal_diffs)
        
        # Blok içi (blok sınırı olmayan yerler) enerjisi
        non_block_x = [x for x in range(4, w, 8)]
        non_block_y = [y for y in range(4, h, 8)]
        
        non_block_energy = 0
        count = 0
        for x in non_block_x:
            if x < w - 1:
                non_block_energy += np.abs(arr[:, x] - arr[:, x+1]).mean()
                count += 1
        for y in non_block_y:
            if y < h - 1:
                non_block_energy += np.abs(arr[y, :] - arr[y+1, :]).mean()
                count += 1
        
        if count > 0:
            non_block_energy /= count
        
        # Oran: blok sınırları, blok içinden ne kadar farklı?
        # Gerçek JPEG → yüksek oran (belirgin blok sınırları)
        # AI görsel → düşük oran (her yer aynı)
        if non_block_energy > 0:
            ratio = block_energy / non_block_energy
        else:
            ratio = 1.0
        
        # Puanlama
        if ratio > 1.5:
            score = 85.0
            reason = f"Belirgin JPEG blok yapısı (ratio={ratio:.2f}) - gerçek fotoğraf"
        elif ratio > 1.2:
            score = 70.0
            reason = f"Orta JPEG blok yapısı (ratio={ratio:.2f})"
        elif ratio > 1.0:
            score = 55.0
            reason = f"Zayıf JPEG blok yapısı (ratio={ratio:.2f})"
        elif ratio > 0.9:
            score = 40.0
            reason = f"Çok zayıf blok yapısı (ratio={ratio:.2f}) - AI olabilir"
        else:
            score = 20.0
            reason = f"Blok yapısı yok (ratio={ratio:.2f}) - AI olma ihtimali yüksek"
        
        return score, reason
    except Exception as e:
        return 50.0, f"JPEG blok analizi hatası: {str(e)[:50]}"

def check_noise(contents: bytes) -> tuple:
    """
    Noise (piksel gürültüsü) analizi.
    Gerçek fotoğraflar: Sensör gürültüsü içerir (yüksek Laplacian varyansı).
    AI görselleri: Pürüzsüz, temiz pikseller (düşük varyans).
    Dönüş: (gerçek_olma_skoru 0-100, sebep)
    """
    try:
        import numpy as np
        from PIL import Image, ImageFilter
        import io

        img = Image.open(io.BytesIO(contents)).convert("L")
        img = img.resize((512, 512), Image.LANCZOS)

        laplacian = img.filter(ImageFilter.FIND_EDGES)
        lap_array = np.array(laplacian, dtype=np.float32)
        variance = float(lap_array.var())

        if variance > 1000:
            score = 90.0
            reason = f"Çok yüksek sensör gürültüsü (variance={variance:.0f}) - gerçek fotoğraf"
        elif variance > 500:
            score = 75.0
            reason = f"Yüksek gürültü (variance={variance:.0f}) - gerçek fotoğraf"
        elif variance > 300:
            score = 55.0
            reason = f"Orta gürültü (variance={variance:.0f}) - kararsız"
        elif variance > 150:
            score = 35.0
            reason = f"Düşük gürültü (variance={variance:.0f}) - AI olabilir"
        else:
            score = 15.0
            reason = f"Çok pürüzsüz (variance={variance:.0f}) - AI olma ihtimali yüksek"

        return score, reason
    except Exception as e:
        return 50.0, f"Noise analizi hatası: {str(e)[:50]}"
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
        return parse_prediction(response.json(), model_config["model_id"])
    except (ValueError, TypeError, KeyError):
        print(f"[{model_config['model_id']}] Geçersiz veya tanınmayan model cevabı")
        return None


async def call_ai_model_ensemble(contents: bytes, content_type: str):
    """Tüm modelleri PARALEL çağırır, sonuçları ağırlıklı ortalamayla birleştirir.
    Dönüş: (final_ai_score, model_breakdown_listesi)
    """
    if AI_BACKEND == "local":
        return await asyncio.to_thread(local_detector.analyze, contents)
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
        # Hiçbir model cevap vermediyse sonuç üretme.
        raise RuntimeError("Analiz modellerine ulaşılamadı veya cevapları doğrulanamadı. Lütfen daha sonra tekrar deneyin.")

    final_ai_score = weighted_sum / weight_total
    return final_ai_score, breakdown


# One successful photo per guest browser; reservations also stop concurrent requests.
GUEST_COOKIE = "detector_guest"

def guest_claims(request):
    claims = decode_jwt(request.cookies.get(GUEST_COOKIE, ""))
    return claims if claims and claims.get("scope") == "guest_trial" else None

def set_guest_cookie(response, request, guest_id, used=False):
    token = jwt.encode({"scope": "guest_trial", "guest_id": guest_id, "used": used,
                        "exp": time.time() + 365 * 86400}, JWT_SECRET, algorithm="HS256")
    response.set_cookie(GUEST_COOKIE, token, max_age=365 * 86400, httponly=True,
                        secure=request.url.scheme == "https", samesite="lax")

@app.get("/auth/guest")
def guest_status(request: Request, response: Response):
    claims = guest_claims(request)
    guest_id = claims["guest_id"] if claims else secrets.token_hex(24)
    conn = get_db()
    conn.execute("CREATE TABLE IF NOT EXISTS guest_trials (guest_id TEXT PRIMARY KEY, used INTEGER NOT NULL DEFAULT 0)")
    conn.execute("INSERT OR IGNORE INTO guest_trials (guest_id, used) VALUES (?, ?)",
                 (guest_id, int(bool(claims and claims.get("used")))))
    used = conn.execute("SELECT used FROM guest_trials WHERE guest_id=?", (guest_id,)).fetchone()["used"]
    conn.commit(); conn.close()
    set_guest_cookie(response, request, guest_id, bool(claims and claims.get("used")))
    response.headers["Cache-Control"] = "no-store"
    return {"remaining": 0 if used else 1}

def reserve_trial(request):
    authorization = request.headers.get("authorization", "")
    if authorization:
        claims = decode_jwt(authorization[7:]) if authorization.startswith("Bearer ") else None
        if not claims or claims.get("scope") or not claims.get("user_id"):
            raise HTTPException(401, detail="SESSION_EXPIRED")
        conn = get_db()
        row = conn.execute("SELECT is_verified FROM users WHERE id=?", (claims["user_id"],)).fetchone()
        conn.close()
        if not row or not row["is_verified"]:
            raise HTTPException(401, detail="SESSION_EXPIRED")
        return None
    claims = guest_claims(request)
    if not claims or claims.get("used"):
        raise HTTPException(401, detail="GUEST_LOGIN_REQUIRED")
    conn = get_db()
    try:
        updated = conn.execute("UPDATE guest_trials SET used=1 WHERE guest_id=? AND used=0", (claims["guest_id"],)).rowcount
        conn.commit()
    finally:
        conn.close()
    if updated != 1:
        raise HTTPException(401, detail="GUEST_LOGIN_REQUIRED")
    return claims["guest_id"]

def refund_trial(guest_id):
    if guest_id:
        conn = get_db()
        conn.execute("UPDATE guest_trials SET used=0 WHERE guest_id=?", (guest_id,))
        conn.commit(); conn.close()

@app.post("/predict")
async def predict(request: Request, response: Response, file: UploadFile = File(None), image_url: str = Form(None), mode: str = Form("general")):
    guest_id = None
    completed = False
    acquired = False
    if AI_BACKEND == "local":
        if ANALYSIS_GATE.locked():
            return {"error": "Analiz motoru şu anda meşgul. Birkaç saniye sonra tekrar deneyin."}
        await ANALYSIS_GATE.acquire()
        acquired = True
    try:
        started_at = time.monotonic()
        contents = None
        content_type = "application/octet-stream"

        if file:
            contents = await file.read(20 * 1024 * 1024 + 1)
        elif image_url:
            contents, content_type = await asyncio.to_thread(download_analysis_image, image_url)
        else:
            return {"error": "Görsel bulunamadı."}

        original_contents = contents
        contents, content_type = normalize_image_bytes(contents)
        if mode not in ("general", "car", "realestate"):
            return {"error": "Geçersiz analiz modu."}

        guest_id = reserve_trial(request)

        # Local object hints do not change the AI score or reject an upload.
        category = {"status": "not_checked", "detected": "unknown"}
        if mode in ("car", "realestate"):
            try:
                from category_detector import classify
                category = await asyncio.to_thread(classify, contents, mode)
            except Exception as error:
                print(json.dumps({"event": "category_check_unavailable", "error_type": type(error).__name__}))
                category = {"status": "unavailable", "detected": "unknown"}

        final_ai_score, breakdown = await call_ai_model_ensemble(contents, content_type)
        
        final_real_score = 100.0 - final_ai_score
        available = [item["ai_score"] for item in breakdown if item["ai_score"] is not None]
        degraded = len(available) != len(MODEL_CONFIGS)
        disagreement = len(available) > 1 and max(available) - min(available) > 40
        verdict = decide_verdict(final_ai_score, DECISION_THRESHOLD, UNCERTAINTY_MARGIN,
                                 degraded or disagreement)
        image_hash = compute_image_hash(original_contents)
        report = build_report(original_contents, mode, breakdown, verdict, degraded, disagreement, category)
        analysis_token = jwt.encode({"scope": "analysis_feedback", "image_hash": image_hash,
            "score": round(final_ai_score, 1), "verdict": verdict, "mode": mode,
            "exp": time.time() + 3600}, JWT_SECRET, algorithm="HS256")
        # Operational metrics only: no image bytes, filename, URL, EXIF values or image hash.
        print(json.dumps({"event": "analysis_completed", "version": "3.0", "backend": AI_BACKEND, "mode": mode,
            "duration_ms": round((time.monotonic() - started_at) * 1000),
            "verdict": verdict, "degraded": degraded, "disagreement": disagreement,
            "models": breakdown}, ensure_ascii=False), flush=True)
        completed = True
        if guest_id:
            set_guest_cookie(response, request, guest_id, True)
        return {
            "ai_probability": round(final_ai_score, 1),
            "image_hash": image_hash,
            "real_probability": round(final_real_score, 1),
            "verdict": verdict,
            "status": "success",
            "result": [
                {"label": "artificial", "score": final_ai_score / 100},
                {"label": "human", "score": final_real_score / 100},
            ],
            "model_breakdown": breakdown,
            "score_type": "uncalibrated_model_score",
            "analysis_version": "3.0",
            "analysis_backend": AI_BACKEND,
            "degraded": degraded,
            "model_disagreement": disagreement,
            "decision_threshold": DECISION_THRESHOLD,
            "evidence_report": report,
            "category_check": category,
            "analysis_token": analysis_token,
        }
    except HTTPException:
        raise
    except Exception as e:
        return {"error": str(e)}
    finally:
        if not completed:
            refund_trial(guest_id)
        if acquired:
            ANALYSIS_GATE.release()


# ---------- HEATMAP (eksik olan endpoint geri eklendi) ----------
@app.post("/predict-heatmap")
async def predict_heatmap(file: UploadFile = File(None), image_url: str = Form(None)):
    if AI_BACKEND == "local":
        return {"error": "Bu analiz motorunda ısı haritası desteklenmiyor."}
    try:
        contents = None
        if file:
            contents = await file.read(20 * 1024 * 1024 + 1)
        elif image_url:
            img_response = requests.get(image_url, timeout=20)
            if img_response.status_code != 200:
                return {"error": "Görsel indirilemedi."}
            contents = img_response.content
        else:
            return {"error": "Görsel bulunamadı."}

        normalized_contents, content_type = normalize_image_bytes(contents)
                # Feedback için görsel hash'i hesapla
        image_hash = compute_image_hash(contents)
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
        grid_size = max(1, min(4, int(os.getenv("HEATMAP_GRID_SIZE", "3"))))
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
@app.get("/brand-logo.png", include_in_schema=False)
def serve_brand_logo():
    return FileResponse(Path(__file__).resolve().parent / "brand-logo.png", media_type="image/png", headers={"Cache-Control": "public, max-age=86400"})



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
