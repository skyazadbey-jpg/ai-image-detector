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

# ---------- CONFIG ----------
HF_TOKEN = os.getenv("HF_TOKEN")
API_URL = "https://router.huggingface.co/hf-inference/models/umm-maybe/AI-image-detector"

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
            otp_code TEXT,
            otp_expires REAL,
            created_at REAL
        )
        """
    )
    conn.commit()
    conn.close()


init_db()


# ---------- MODELS ----------
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


# ---------- HELPERS ----------
def make_jwt(user_id: int, email: str) -> str:
    payload = {"user_id": user_id, "email": email, "exp": time.time() + 60 * 60 * 24 * 7}
    return jwt.encode(payload,
