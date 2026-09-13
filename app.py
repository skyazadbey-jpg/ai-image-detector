from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
import os
import sqlite3
import secrets
import string
import time
import jwt
import requests
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

HF_TOKEN = os.getenv("HF_TOKEN")
API_URL = "https://router.huggingface.co/hf-inference/models/umm-maybe/AI-image-detector"

RESEND_API_KEY = os.getenv("RESEND_API_KEY")
JWT_SECRET = os.getenv("JWT_SECRET", "change-me-in-production")
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
FROM_EMAIL = os.getenv("FROM_EMAIL", "onboarding@resend.dev")

DB_PATH = "users.db"


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    conn.execute(
        "CREATE TABLE IF NOT EXISTS users ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "email TEXT UNIQUE NOT NULL,"
        "password_hash TEXT,"
        "google_id TEXT,"
        "is_verified INTEGER DEFAULT 0,"
        "otp_code TEXT,"
        "otp_expires REAL,"
        "created_at REAL)"
    )
    conn.commit()
    conn.close()


init_db()


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class VerifyRequest(BaseModel):
    email: EmailStr
    otp: str


class ResendRequest(BaseModel):
    email: EmailStr


class GoogleAuthRequest(BaseModel):
    id_token: str


def make_jwt(user_id, email):
    payload = {"user_id": user_id, "email": email, "exp": time.time() + 604800}
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")


def decode_jwt(token):
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
    except jwt.PyJWTError:
        return None


def get_current_user(authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Oturum bulunamadi.")
    token = authorization.split(" ", 1)[1]
    payload = decode_jwt(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Gecersiz veya suresi dolmus oturum.")
    return payload


def generate_otp():
    return "".join(secrets.choice(string.digits) for _ in range(6))


def send_email(to_email, subject, html):
    if not RESEND_API_KEY:
        raise RuntimeError("RESEND_API_KEY sunucuda tanimli degil.")
    resp = requests.post(
        "https://api.resend.com/emails",
        headers={
            "Authorization": "Bearer " + RESEND_API_KEY,
            "Content-Type": "application/json",
        },
        json={
            "from": "AI Image Detector <" + FROM_EMAIL + ">",
            "to": [to_email],
            "subject": subject,
            "html": html,
        },
        timeout=15,
    )
    if resp.status_code >= 300:
        raise RuntimeError("Email gonderilemedi: " + resp.text)


def user_to_public(row):
    return {"email": row["email"], "is_verified": bool(row["is_verified"])}


@app.post("/auth/register")
def register(data: RegisterRequest):
    conn = get_db()
    existing = conn.execute("SELECT * FROM users WHERE email = ?", (data.email,)).fetchone()

    otp = generate_otp()
    otp_expires = time.time() + 600

    if existing:
        if existing["is_verified"]:
            conn.close()
            raise HTTPException(status_code=400, detail="Bu email zaten kayitli. Giris yapmayi deneyin.")
        conn.execute(
            "UPDATE users SET password_hash=?, otp_code=?, otp_expires=? WHERE email=?",
            (bcrypt.hash(data.password), otp, otp_expires, data.email),
        )
    else:
        conn.execute(
            "INSERT INTO users (email, password_hash, otp_code, otp_expires, created_at) VALUES (?,?,?,?,?)",
            (data.email, bcrypt.hash(data.password), otp, otp_expires, time.time()),
        )
    conn.commit()
    conn.close()

    try:
        send_email(
            data.email,
            "Dogrulama Kodunuz",
            "<p>Dogrulama kodunuz: <b style='font-size:24px'>" + otp + "</b></p><p>Bu kod 10 dakika gecerlidir.</p>",
        )
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"message": "Dogrulama kodu email adresinize gonderildi."}


@app.post("/auth/verify")
def verify(data: VerifyRequest):
    conn = get_db()
    row = conn.execute("SELECT * FROM users WHERE email = ?", (data.email,)).fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Kullanici bulunamadi.")
    if row["otp_code"] != data.otp:
        conn.close()
        raise HTTPException(status_code=400, detail="Hatali dogrulama kodu.")
    if row["otp_expires"] and time.time() > row["otp_expires"]:
        conn.close()
        raise HTTPException(status_code=400, detail="Dogrulama kodunun suresi dolmus, yeni kod isteyin.")

    conn.execute("UPDATE users SET is_verified=1, otp_code=NULL, otp_expires=NULL WHERE email=?", (data.email,))
    conn.commit()
    updated = conn.execute("SELECT * FROM users WHERE email=?", (data.email,)).fetchone()
    conn.close()

    token = make_jwt(updated["id"], updated["email"])
    return {"token": token, "user": user_to_public(updated)}


@app.post("/auth/resend-otp")
def resend_otp(data: ResendRequest):
    conn = get_db()
    row = conn.execute("SELECT * FROM users WHERE email=?", (data.email,)).fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Kullanici bulunamadi.")
    if row["is_verified"]:
        conn.close()
        raise HTTPException(status_code=400, detail="Hesap zaten dogrulanmis.")

    otp = generate_otp()
    otp_expires = time.time() + 600
    conn.execute("UPDATE users SET otp_code=?, otp_expires=? WHERE email=?", (otp, otp_expires, data.email))
    conn.commit()
    conn.close()

    try:
        send_email(data.email, "Yeni Dogrulama Kodunuz", "<p>Yeni kodunuz: <b style='font-size:24px'>" + otp + "</b></p>")
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"message": "Yeni kod gonderildi."}


@app.post("/auth/login")
def login(data: LoginRequest):
    conn = get_db()
    row = conn.execute("SELECT * FROM users WHERE email=?", (data.email,)).fetchone()
    conn.close()

    if not row or not row["password_hash"] or not bcrypt.verify(data.password, row["password_hash"]):
        raise HTTPException(status_code=401, detail="Email veya sifre hatali.")

    if not row["is_verified"]:
        raise HTTPException(status_code=403, detail="Hesabiniz henuz dogrulanmamis.")

    token = make_jwt(row["id"], row["email"])
    return {"token": token, "user": user_to_public(row)}


@app.post("/auth/google")
def google_auth(data: GoogleAuthRequest):
    if not GOOGLE_CLIENT_ID:
        raise HTTPException(status_code=500, detail="Google Client ID sunucuda tanimli degil.")
    try:
        idinfo = google_id_token.verify_oauth2_token(
            data.id_token, google_requests.Request(), GOOGLE_CLIENT_ID
        )
    except ValueError:
        raise HTTPException(status_code=401, detail="Gecersiz Google token.")

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


@app.get("/auth/me")
def me(current=Depends(get_current_user)):
    conn = get_db()
    row = conn.execute("SELECT * FROM users WHERE id=?", (current["user_id"],)).fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Kullanici bulunamadi.")
    return {"user": user_to_public(row)}


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
            img_response = requests.get(image_url)
            if img_response.status_code != 200:
                return {"error": "Gorsel internet adresinden indirilemedi."}
            contents = img_response.content
            content_type = img_response.headers.get("content-type", "image/jpeg")
        else:
            return {"error": "Gorsel bulunamadi."}

        headers = {
            "Authorization": "Bearer " + str(HF_TOKEN),
            "Content-Type": content_type,
        }

        response = requests.post(API_URL, headers=headers, data=contents)

        if response.status_code != 200:
            return {"error": "Hugging Face API Error: " + response.text}

        return {"result": response.json()}
    except Exception as e:
        return {"error": str(e)}