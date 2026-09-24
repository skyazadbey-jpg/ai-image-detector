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
import base64
import random
import hmac
import hashlib
import json
from PIL import Image, ImageFilter, ImageDraw
from passlib.hash import bcrypt
from google.oauth2 import id_token as google_id_token
from google.auth.transport import requests as google_requests
import tempfile

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- CONFIG ----------
HF_TOKEN = os.getenv("HF_TOKEN")
API_URL = "https://router.huggingface.co/hf-inference/models/prithivMLmods/deepfake-detector-model-v1"
API_URL_2 = "https://router.huggingface.co/hf-inference/models/umm-maybe/AI-image-detector"
API_URL_3 = "https://router.huggingface.co/hf-inference/models/Vontra/detectra-v1"
RESEND_API_KEY = os.getenv("RESEND_API_KEY")
JWT_SECRET = os.getenv("JWT_SECRET", "change-me-in-production")
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
FROM_EMAIL = os.getenv("FROM_EMAIL", "onboarding@resend.dev")
LEMONSQUEEZY_WEBHOOK_SECRET = os.getenv("LEMONSQUEEZY_WEBHOOK_SECRET", "")

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
    print(f"[SEND_EMAIL] Başladı → to={to_email} from={FROM_EMAIL}")
    print(f"[SEND_EMAIL] RESEND_API_KEY mevcut mu? {bool(RESEND_API_KEY)}")
    if not RESEND_API_KEY:
        print("[SEND_EMAIL] HATA: RESEND_API_KEY yok!")
        raise RuntimeError("RESEND_API_KEY sunucuda tanımlı değil.")
    try:
        resp = requests.post(
            "https://api.resend.com/emails",
            headers={
                "Authorization": f"Bearer {RESEND_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "from": "Destek <destek@ai-image-detector.com>",
                "to": [to_email],
                "subject": subject,
                "html": html,
            },
            timeout=15,
        )
        print(f"[SEND_EMAIL] Resend yanıtı: status={resp.status_code}")
        print(f"[SEND_EMAIL] Resend cevabı: {resp.text[:500]}")
        if resp.status_code >= 300:
            raise RuntimeError(f"Email gönderilemedi: {resp.text}")
        print(f"[SEND_EMAIL] ✅ Başarılı!")
    except Exception as e:
        print(f"[SEND_EMAIL] ❌ Exception: {type(e).__name__} - {str(e)}")
        raise


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
    
    print(f"[FORGOT] Kullanıcı bulundu: {row['email']} (id={row['id']})")
    print(f"[FORGOT] password_hash var mı? {bool(row['password_hash'])}")

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


# ---------- GUMROAD WEBHOOK ----------
@app.post("/webhooks/gumroad")
async def gumroad_webhook(request: Request):
    try:
        # Gumroad form verilerini veya JSON verisini alır
        form_data = await request.form()
        data = dict(form_data)
        
        # Gumroad'dan gelen temel alanlar
        event_name = data.get("event_name", "sale") # satılık / abonelik durumu
        user_email = data.get("email")
        
        print(f"GUMROAD WEBHOOK: event={event_name} email={user_email} data={data}")

        # Pro durumunu güncelleyecek veritabanı bağlantısı
        conn = get_db()
        matched = False

        if user_email:
            normalized_email = user_email.strip().lower()
            row = conn.execute(
                "SELECT id FROM users WHERE LOWER(TRIM(email))=?", (normalized_email,)
            ).fetchone()
            
            if row:
                matched = True
                # Satış gerçekleştiğinde veya abonelik başladığında is_pro = 1 yapılır
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


# ---------- PREDICT ----------
@app.post("/predict")
async def predict(file: UploadFile = File(None), image_url: str = Form(None)):
    try:
        contents = None
        content_type = "application/octet-stream"

        if file:
            contents = await file.read()
            content_type = file.content_type or "application/octet-stream"
        elif image_url:
            img_response = requests.get(image_url, timeout=20)
            if img_response.status_code != 200:
                return {"error": "Görsel indirilemedi."}
            contents = img_response.content
            content_type = img_response.headers.get("content-type", "image/jpeg")
        else:
            return {"error": "Görsel bulunamadı."}

        headers = {"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": content_type}

        def get_score(api_url):
            try:
                r = requests.post(api_url, headers=headers, data=contents, timeout=20)
                if r.status_code != 200:
                    return None
                data = r.json()
                items = data.get("result", data) if isinstance(data, dict) else data
                if isinstance(items, list):
                    for item in items:
                        if isinstance(item, dict):
                            label = str(item.get("label", "")).lower()
                            if label in ["ai", "fake", "artificial", "generated", "label_1"]:
                                return float(item.get("score", 0))
                            elif label in ["hum", "human", "real", "label_0"]:
                                return 1 - float(item.get("score", 0))
                return None
            except Exception:
                return None

        score_1 = get_score(API_URL)
        score_2 = get_score(API_URL_2)

        scores = [s for s in [score_1, score_2] if s is not None]

        if not scores:
            return {"error": "Modeller yanıt vermedi. Lütfen tekrar deneyin."}

        final_ai_score = sum(scores) / len(scores)
        final_human_score = 1 - final_ai_score

        return {
            "result": [
                {"label": "artificial", "score": final_ai_score},
                {"label": "human", "score": final_human_score}
            ],
            "models_used": len(scores)
        }
    except Exception as e:
        return {"error": str(e)}


# ---------- HEATMAP ----------
@app.post("/predict-heatmap")
async def predict_heatmap(file: UploadFile = File(None), image_url: str = Form(None)):
    try:
        contents = None
        if file:
            contents = await file.read()
        elif image_url:
            img_response = requests.get(image_url)
            if img_response.status_code != 200:
                return {"error": "Görsel indirilemedi."}
            contents = img_response.content
        else:
            return {"error": "Görsel bulunamadı."}

        headers = {
            "Authorization": f"Bearer {HF_TOKEN}",
            "Content-Type": "application/octet-stream",
        }
        response = requests.post(API_URL, headers=headers, data=contents)
        if response.status_code != 200:
            return {"error": f"Model hatası: {response.text}"}

        result = response.json()
        ai_score = 0.5
        if isinstance(result, list) and result:
            for item in result:
                if item.get("label") in ["artificial", "fake", "ai"]:
                    ai_score = item.get("score", 0.5)
        elif isinstance(result, dict) and "score" in result:
            ai_score = result["score"]

        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img = img.resize((512, 512))
        width, height = img.size

        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        num_red_blobs = int(ai_score * 15) + 3
        num_blue_blobs = int((1 - ai_score) * 10) + 2
        random.seed(int(ai_score * 1000))

        for _ in range(num_red_blobs):
            x = random.randint(0, width)
            y = random.randint(0, height)
            r = random.randint(40, 120)
            draw.ellipse([x - r, y - r, x + r, y + r], fill=(255, 50, 50, 60))

        for _ in range(num_blue_blobs):
            x = random.randint(0, width)
            y = random.randint(0, height)
            r = random.randint(30, 90)
            draw.ellipse([x - r, y - r, x + r, y + r], fill=(50, 150, 255, 50))

        overlay = overlay.filter(ImageFilter.GaussianBlur(radius=25))
        combined = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")

        buffered = io.BytesIO()
        combined.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return {
            "heatmap_base64": f"data:image/png;base64,{img_base64}",
            "ai_score": round(ai_score * 100, 1),
            "human_score": round((1 - ai_score) * 100, 1),
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