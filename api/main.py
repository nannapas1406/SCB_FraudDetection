import os
import json
import sqlite3
from datetime import datetime
from typing import Optional
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pathlib import Path

# -----------------------
# Config
# -----------------------
BASE_DIR = Path(__file__).resolve().parent  
ROOT_DIR = BASE_DIR.parent                

MODEL_PATH = os.getenv("MODEL_PATH", str(ROOT_DIR / "model" / "xgb_model.pkl"))
TRANSFORMER_PATH = os.getenv("TRANSFORMER_PATH", str(ROOT_DIR / "model" / "col_transformer.pkl"))
DB_PATH = os.getenv("DB_PATH", str(BASE_DIR / "fraud_db.sqlite"))

THRESHOLD = float(os.getenv("THRESHOLD", "0.5"))  # ปรับ threshold ได้ผ่าน ENV

# ฟีเจอร์ดิบที่ API รับ (ต้องสอดคล้องกับ col_transformer ตอนเทรน)
RAW_FEATURES = ["time_ind", "transac_type", "amount", "src_bal", "dst_bal"]

# -----------------------
# Load model & transformer
# -----------------------
try:
    model = joblib.load(MODEL_PATH)
    col_transformer = joblib.load(TRANSFORMER_PATH)
except Exception as e:
    raise RuntimeError(f"Cannot load model or transformer: {e}")

# -----------------------
# DB init
# -----------------------
def init_db():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS fraud_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            prob REAL NOT NULL,
            label INTEGER NOT NULL,
            status TEXT NOT NULL DEFAULT 'new'
        )
        """
    )
    con.commit()
    con.close()

init_db()

# -----------------------
# FastAPI
# -----------------------
app = FastAPI(title="Fraud Detection API", version="1.0.0")

class Transaction(BaseModel):
    time_ind: int = Field(..., description="Simulation time index (hour step)")
    transac_type: str = Field(..., description="CASH-IN, CASH-OUT, DEBIT, PAYMENT, TRANSFER")
    amount: float = Field(..., ge=0)
    src_bal: float = Field(..., description="Sender balance before txn")
    dst_bal: Optional[float] = Field(None, description="Receiver balance before txn (can be null)")
    is_fraud: Optional[int] = Field(None, description="Optional ground-truth label if available")

class PredictResponse(BaseModel):
    prob: float
    label: int
    threshold: float

def _to_dataframe(tx: Transaction) -> pd.DataFrame:
    """
    สร้าง DataFrame ให้ตรงกับอินพุตของ col_transformer ที่เคยเทรนไว้
    จะ 'ไม่ใส่' is_fraud ไปในฟีเจอร์สำหรับแปลง
    """
    row = {k: getattr(tx, k) for k in RAW_FEATURES}
    # ให้ None กลายเป็น NaN เพื่อให้ transformer handle ได้
    for k, v in row.items():
        if v is None:
            row[k] = np.nan

    df = pd.DataFrame([row], columns=RAW_FEATURES)

    # เพิ่มฟีเจอร์ hour_of_day 
    df["hour_of_day"] = df["time_ind"] % 24

    return df

def _persist_if_fraud(payload: dict, prob: float, label: int):
    if label != 1:
        return
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(
        """
        INSERT INTO fraud_predictions (created_at, payload_json, prob, label, status)
        VALUES (?, ?, ?, ?, ?)
        """,
        (datetime.utcnow().isoformat(), json.dumps(payload, ensure_ascii=False), float(prob), int(label), "new"),
    )
    con.commit()
    con.close()

@app.get("/", tags=["health"])
def health():
    return {"ok": True, "model": "xgb", "threshold": THRESHOLD}

@app.post("/predict", response_model=PredictResponse, tags=["inference"])
def predict(tx: Transaction):
    try:
        df = _to_dataframe(tx)
        X = col_transformer.transform(df)
        proba = float(model.predict_proba(X)[:, 1][0])
        label = int(proba >= THRESHOLD)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {e}")

    # เก็บลง DB เมื่อคาดว่าเป็น fraud
    _persist_if_fraud(payload=tx.dict(), prob=proba, label=label)
    return PredictResponse(prob=proba, label=label, threshold=THRESHOLD)

@app.get("/frauds", tags=["storage"])
def list_frauds(limit: int = 100, offset: int = 0):
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(
        """
        SELECT id, created_at, payload_json, prob, label, status
        FROM fraud_predictions
        ORDER BY id DESC
        LIMIT ? OFFSET ?
        """,
        (limit, offset),
    )
    rows = cur.fetchall()
    con.close()

    results = []
    for r in rows:
        results.append(
            {
                "id": r[0],
                "created_at": r[1],
                "payload": json.loads(r[2]),
                "prob": r[3],
                "label": r[4],
                "status": r[5],
            }
        )
    return {"count": len(results), "items": results}
