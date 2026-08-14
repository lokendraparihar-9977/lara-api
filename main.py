# LARA - Lightweight Adaptive Recognition API
# main.py - Mark II + Supabase: Real database backend

from fastapi import FastAPI, File, UploadFile, HTTPException, Header
from PIL import Image
from ultralytics import YOLO
from supabase import create_client
from dotenv import load_dotenv
import io
import os
from datetime import datetime
import razorpay
from enum import Enum

class PlanType(str, Enum):
    startup = "startup"
    pro = "pro"
    business = "business"

# def subscribe(plan: PlanType, x_api_key: str = Header(...)):

# ── Load environment variables ─────────────────────────
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
OWNER_KEY    = os.getenv("OWNER_API_KEY")

# ── Connect to Supabase ────────────────────────────────
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ── Razorpay client ────────────────────────────────────
RAZORPAY_KEY_ID = os.getenv("RAZORPAY_KEY_ID")
RAZORPAY_KEY_SECRET = os.getenv("RAZORPAY_KEY_SECRET")
razorpay_client = razorpay.Client(auth=(RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET))

# ── Tier limits ────────────────────────────────────────
TIER_LIMITS = {
    "free":     100,
    "startup":  500,
    "pro":      2000,
    "business": 10000,
    "owner":    999999999,
}

# ── LARA app ───────────────────────────────────────────
app = FastAPI(
    title="LARA API",
    description="Lightweight Adaptive Recognition API — AI vision for every machine",
    version="0.4.0"
)

# ── Load model ─────────────────────────────────────────
print("Loading LARA model...")
model = YOLO("yolov8n.pt")
print("Model loaded successfully!")

# ── Helper: get user from Supabase ────────────────────
def get_user(api_key: str):
    result = supabase.table("users").select("*").eq("api_key", api_key).execute()
    if result.data:
        return result.data[0]
    return None

# ── Helper: log usage to Supabase ─────────────────────
def log_usage(api_key: str, endpoint: str):
    supabase.table("usage").insert({
        "api_key": api_key,
        "endpoint": endpoint,
        "timestamp": datetime.now().isoformat()
    }).execute()

# ── Helper: count usage from Supabase ─────────────────
def count_usage(api_key: str):
    result = supabase.table("usage").select("id").eq("api_key", api_key).execute()
    return len(result.data)

# ── Health check ───────────────────────────────────────
@app.get("/")
def home():
    return {
        "name": "LARA API",
        "version": "0.4.0",
        "status": "running",
        "model": "yolov8n",
        "message": "Welcome to LARA — AI vision for every machine"
    }

# ── Usage endpoint ─────────────────────────────────────
@app.get("/usage")
def check_usage(x_api_key: str = Header(...)):
    user = get_user(x_api_key)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid API key")

    total_calls = count_usage(x_api_key)
    limit = TIER_LIMITS[user["tier"]]

    return {
        "name": user["name"],
        "tier": user["tier"],
        "calls_used": total_calls,
        "calls_limit": limit,
        "calls_remaining": limit - total_calls
    }

# ── Register new user ──────────────────────────────────
@app.post("/register")
def register(name: str, email: str):
    import secrets
    api_key = "lara-" + secrets.token_hex(16)

    existing = supabase.table("users").select("email").eq("email", email).execute()
    if existing.data:
        raise HTTPException(status_code=400, detail="Email already registered")

    supabase.table("users").insert({
        "name": name,
        "email": email,
        "api_key": api_key,
        "tier": "free"
    }).execute()

    return {
        "message": "Registration successful",
        "name": name,
        "api_key": api_key,
        "tier": "free",
        "calls_limit": TIER_LIMITS["free"]
    }

# ── Create payment order ───────────────────────────────
@app.post("/subscribe")
def subscribe(plan: str, x_api_key: str = Header(...)):
    user = get_user(x_api_key)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid API key")

    # Plan prices in paise (INR) — USD equivalent shown in comments
    plan_prices = {
        "startup":  24900,   # $2.99/month — 500 calls
        "pro":      49900,   # $5.99/month — 2000 calls
        "business": 164900,  # $19.99/month — 10000 calls
    }

    if plan not in plan_prices:
        raise HTTPException(status_code=400, detail="Invalid plan. Choose 'startup', 'pro', or 'business'")

    amount = plan_prices[plan]

    order = razorpay_client.order.create({
        "amount": amount,
        "currency": "INR",
        "receipt": f"lara_{user['api_key'][:10]}_{plan}",
        "notes": {
            "api_key": user["api_key"],
            "plan": plan,
            "name": user["name"]
        }
    })

    return {
        "order_id": order["id"],
        "amount": amount,
        "currency": "INR",
        "plan": plan,
        "key_id": RAZORPAY_KEY_ID,
        "name": user["name"]
    }

# ── Verify payment ─────────────────────────────────────
@app.post("/payment/verify")
def verify_payment(
    razorpay_order_id: str,
    razorpay_payment_id: str,
    razorpay_signature: str,
    x_api_key: str = Header(...)
):
    user = get_user(x_api_key)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid API key")

    # Verify signature
    try:
        razorpay_client.utility.verify_payment_signature({
            "razorpay_order_id": razorpay_order_id,
            "razorpay_payment_id": razorpay_payment_id,
            "razorpay_signature": razorpay_signature
        })
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid payment signature")

    # Upgrade user to pro
    supabase.table("users").update({"tier": "pro"}).eq("api_key", x_api_key).execute()

    # Log billing
    supabase.table("billing").upsert({
        "api_key": x_api_key,
        "plan": "pro",
        "status": "active",
        "stripe_customer_id": razorpay_payment_id
    }).execute()

    return {
        "status": "success",
        "message": "Payment verified. Your account has been upgraded to Pro.",
        "tier": "pro",
        "calls_limit": 10000
    }

# ── Detection endpoint ─────────────────────────────────
@app.post("/detect")
async def detect(
    file: UploadFile = File(...),
    x_api_key: str = Header(...)
):
    # Step 1 — Validate API key
    user = get_user(x_api_key)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid API key")

    # Step 2 — Check usage limit
    total_calls = count_usage(x_api_key)
    limit = TIER_LIMITS[user["tier"]]

    if total_calls >= limit:
        raise HTTPException(
            status_code=429,
            detail=f"Monthly limit of {limit} calls reached. Upgrade to Pro."
        )

    # Step 3 — Read and process image
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")

    # Step 4 — Run YOLOv8 inference
    results = model(img, verbose=False)

    # Step 5 — Parse detections
    detections = []
    for box in results[0].boxes:
        label = model.names[int(box.cls)]
        confidence = round(float(box.conf), 3)
        x1, y1, x2, y2 = [round(float(v)) for v in box.xyxy[0]]
        detections.append({
            "label": label,
            "confidence": confidence,
            "box": [x1, y1, x2, y2]
        })

    # Step 6 — Log to Supabase
    log_usage(x_api_key, "/detect")

    # Step 7 — Return response
    return {
        "status": "success",
        "model": "lara-detect-v1",
        "image_size": [img.width, img.height],
        "detections": detections,
        "count": len(detections),
        "calls_used": total_calls + 1,
        "calls_remaining": limit - total_calls - 1
    }