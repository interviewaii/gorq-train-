import threading
import subprocess
import os
import signal
import numpy as np
import xgboost as xgb
from groq import Groq

# Global XGBoost Model
xgb_model = None
try:
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model("my_xgboost_brain.json")
    print("🧠 XGBoost Brain Loaded Successfully! Ready to filter trades.")
except Exception as e:
    print(f"⚠️ Could not load XGBoost Brain (it's okay, continuing without ML filter). Error: {e}")
    xgb_model = None

# ── Cloud Readiness ────────────────────────────────────────────────────────────

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from datetime import datetime

app = FastAPI(title="Carl AI Predictor")

# CORS - Allow browser to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

import csv
import os
import re

# --- SUPABASE CONFIG ---
# Replace these with your actual keys or use Environment Variables
SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://nnurukmzbpcqdszcmsgq.supabase.co")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "") # Pull from Render Environment Variables

try:
    from supabase import create_client, Client
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    print("☁️ Supabase Cloud Database Connected!")
except Exception as e:
    supabase = None
    print(f"⚠️ Supabase not connected (using local CSV fallback?): {e}")

BASE_MODEL = "unsloth/Qwen2.5-0.5B-Instruct"
LOG_FILE = "training_data.csv"

# Lock for CSV operations to prevent race conditions between predicting and logging outcomes
csv_lock = threading.Lock()

def log_prediction_data(prompt_text, ai_reply, predicted_open="N/A", predicted_close="N/A", math_dir=""):
    """
    Parses EVERY technical indicator from the prompt and saves it to the cloud.
    Matches the Supabase SQL schema exactly.
    """
    try:
        # Extract fields using regex
        symbol = re.search(r'^([A-Z/]+)', prompt_text)
        sym_str = symbol.group(1) if symbol else "N/A"
        print(f"📝 Logging prediction for {sym_str}...")

        # Helper to extract values from the prompt text
        def find_val(label):
            match = re.search(rf'{label}: ([\d.-]+)', prompt_text)
            return match.group(1) if match else None

        # Prepare the exact row matching the SQL schema
        row = {
            "timestamp": datetime.now().isoformat(),
            "symbol": sym_str,
            "model_type": "Groq-Llama3-70B",
            "predicted_dir": math_dir.upper() or ("UP" if "DIRECTION: UP" in ai_reply.upper() else "DOWN"),
            "actual_open": None,
            "actual_close": None,
            "actual_dir": None,
            "correct": None,
            
            # Indicators (Must Match SQL labels exactly)
            "rsi_14": find_val("RSI"),
            "macd": find_val("MACD"),
            "macd_signal": find_val("MACD Signal"),
            "macd_hist": find_val("MACD Hist"),
            "ao": find_val("AO"),
            "stoch_k": find_val("Stoch K"),
            "stoch_d": find_val("Stoch D"),
            "willr": find_val("WillR"),
            "adx": find_val("ADX"),
            "adx_pos": find_val(r"ADX\+"),
            "adx_neg": find_val("ADX-"),
            "ema_8": find_val("EMA 8"),
            "ema_21": find_val("EMA 21"),
            "ema_50": find_val("EMA 50"),
            "ema_200": find_val("EMA 200"),
            "sma_200": find_val("SMA 200"),
            "bb_upper": find_val("BB Upper"),
            "bb_lower": find_val("BB Lower"),
            "atr": find_val("ATR"),
            "price_close": find_val("Price") or find_val("Current Price")
        }
        
        # --- CSV Logging (Fallback Backup) ---
        try:
            file_exists = os.path.isfile(LOG_FILE)
            with open(LOG_FILE, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=row.keys())
                if not file_exists: writer.writeheader()
                writer.writerow(row)
            print(f"✅ Row added to {LOG_FILE}")
        except Exception as csv_err:
            print(f"⚠️ CSV Logging Error: {csv_err}")

        # --- SUPABASE Logging (Primary Cloud) ---
        if supabase:
            try:
                # Supabase handles UUIDs and timestamps automatically if defined, 
                # but we send our generated timestamp for consistency.
                supabase.table("predictions").insert(row).execute()
                print(f"☁️ Supabase Cloud Logged: {sym_str}")
            except Exception as ex:
                print(f"⚠️ Supabase Log Error (Check your Table Columns!): {ex}")

    except Exception as e:
        print(f"⚠️ Data Parsing Error: {e}")

# Initialize Groq Client
groq_api_key = os.environ.get("GROQ_API_KEY", "") # Pull from Render Environment Variables
try:
    groq_client = Groq(api_key=groq_api_key)
    print("🚀 Groq API Client Initialized! (Llama 3 70B)")
except Exception as e:
    groq_client = None
    print(f"❌ Error initializing Groq: {e}")

def load_model():
    pass # No longer needed, model is in the cloud


class PredictRequest(BaseModel):
    messages: list
    math_dir: str = ""  # Direction computed from React math (RSI/EMA/DMI) — sent by the chart


# Lock to ensure only one prediction runs at a time (prevents GPU crashes/interference)
inference_lock = threading.Lock()

@app.get("/")
async def root():
    return {"message": "Carl AI Prediction Server is Running 🚀", "status": "online"}

@app.get("/ping")
async def ping_server():
    """
    Simple endpoint for UptimeRobot to ping every 5 minutes
    so the free Render server never goes to sleep.
    """
    return {"status": "alive", "time": datetime.now().isoformat()}

@app.post("/predict")
async def predict_candle(req: PredictRequest):
    if groq_client is None:
        return {"prediction": "Groq client is not initialized. Please check your API key."}

    # Use the user message as prompt
    user_msg = req.messages[0].get("content", "") if req.messages else ""

    # SAVING ROW IMMEDIATELY (Before AI thinks) to prevent race conditions 
    # where the candle closes before the AI finished thinking.
    with csv_lock: # csv_lock is used for both CSV and Supabase logging to prevent race conditions
        log_prediction_data(user_msg, "AI ANALYSING...", "N/A", "N/A", math_dir=req.math_dir)

    # Ask AI for direction, price targets, stars, and probabilities
    system_prompt = "You are a professional crypto analyst. \nINSTRUCTION: Compare 'Current Price' with 'Last Close' to determine trend. Do NOT always reply 50/50. Be DECISIVE based on price action.\nReply in this EXACT format:\nDIRECTION: UP or DOWN\nOPEN: [price]\nHIGH: [price]\nLOW: [price]\nCLOSE: [price]\nSTARS: [1-5]\nPROBABILITY: [Bullish X%, Bearish Y%]"

    try:
        with inference_lock:
            chat_completion = groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": user_msg,
                    }
                ],
                model="llama3-70b-8192",
                temperature=0.40,
                max_tokens=200,
            )
            reply = chat_completion.choices[0].message.content
    except Exception as e:
        print(f"❌ Groq API Error: {e}")
        reply = f"Error generating prediction. Did you hit API limits? {e}"

    reply = reply.strip("\n").strip()

    # ---------------------------------------------------------
    # 🧠 XGBoost Machine Learning Filter
    # ---------------------------------------------------------
    try:
        if xgb_model is not None:
            # 1. Extract the current math from the prompt
            price = float(re.search(r'Price:([\d.]+)', user_msg).group(1))
            score = float(re.search(r'Trend Score: ([\d.-]+)', user_msg).group(1))
            rsi = float(re.search(r'RSI: ([\d.]+)', user_msg).group(1))
            adx = float(re.search(r'ADX: ([\d.]+)', user_msg).group(1))

            # 2. Format it for XGBoost (must match Kaggle training EXACTLY)
            features = np.array([[price, score, rsi, adx]])

            # 3. Get probabilities [Prob DOWN, Prob UP]
            probs = xgb_model.predict_proba(features)[0]
            prob_down = probs[0]
            prob_up = probs[1]
            
            print(f"\n[XGBoost Filter] UP: {prob_up*100:.1f}% | DOWN: {prob_down*100:.1f}%")

            # 4. Read what Qwen (LLM) wanted to do
            llm_wants_up = "DIRECTION: UP" in reply.upper()
            llm_wants_down = "DIRECTION: DOWN" in reply.upper()

            # 5. The Override Logic
            threshold = 0.60 # Needs 60% confidence to override
            
            if llm_wants_up and prob_down > threshold:
                print("🚨 XGBoost OVERRIDE: LLM said UP, but Math says DOWN. Forcing DOWN.")
                reply = reply.replace("DIRECTION: UP", "DIRECTION: DOWN")
                reply += f"\n[ML Filter Triggered: {prob_down*100:.1f}% Bearish Math Probability]"
                
            elif llm_wants_down and prob_up > threshold:
                print("🚨 XGBoost OVERRIDE: LLM said DOWN, but Math says UP. Forcing UP.")
                reply = reply.replace("DIRECTION: DOWN", "DIRECTION: UP")
                reply += f"\n[ML Filter Triggered: {prob_up*100:.1f}% Bullish Math Probability]"
    except Exception as e:
        print(f"⚠️ XGBoost Filter Failed (Skipping step): {e}")
    # ---------------------------------------------------------

    return {"prediction": reply or "No prediction generated."}


class ResultRequest(BaseModel):
    symbol: str
    actual_open: float
    actual_close: float


@app.post("/log-result")
async def log_result(req: ResultRequest):
    """
    Called by the chart after the predicted candle CLOSES.
    Finds the last prediction row for this symbol and appends the actual result.
    """
    try:
        with csv_lock: # csv_lock is used for both CSV and Supabase logging to prevent race conditions
            def clean_sym(s):
                return re.sub(r'[^A-Z0-9]', '', str(s).upper())

            actual_dir = "UP" if req.actual_close >= req.actual_open else "DOWN"
            target_symbol = clean_sym(req.symbol)
            
            print(f"📊 Logging Result: {target_symbol} | {req.actual_open} -> {req.actual_close} ({actual_dir})")

            # --- CSV Update ---
            if not os.path.isfile(LOG_FILE):
                print("⚠️ No log file found for CSV update.")
            else:
                rows = []
                with open(LOG_FILE, 'r') as f:
                    reader = csv.DictReader(f)
                    fieldnames = reader.fieldnames
                    rows = list(reader)

                # Find ALL rows for this symbol that don't have an actual_dir yet
                updated = False
                for row in reversed(rows):
                    sym = clean_sym(row.get("symbol", ""))
                    if sym == target_symbol and not str(row.get("actual_dir", "")).strip():
                        row["actual_open"] = str(req.actual_open)
                        row["actual_close"] = str(req.actual_close)
                        row["actual_dir"] = actual_dir
                        row["correct"] = "TRUE" if str(row.get("predicted_dir", "")).strip().upper() == actual_dir else "FALSE"
                        updated = True
                        break

                if updated:
                    with open(LOG_FILE, 'w', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(rows)
                    print(f"✅ CSV Result Logged for {target_symbol}")
                else:
                    print(f"⚠️ No unmatched prediction found in CSV for {target_symbol}")

            # --- SUPABASE SYNC ---
            if supabase:
                try:
                    # Search for the latest record for this symbol without an outcome
                    res = supabase.table("predictions").select("*").eq("symbol", target_symbol).is_("actual_dir", "null").order("timestamp", desc=True).limit(1).execute()
                    if res.data:
                        row_id = res.data[0]['id']
                        pred_dir = str(res.data[0].get("predicted_dir", "")).upper()
                        is_correct = "TRUE" if pred_dir == actual_dir else "FALSE"
                        
                        supabase.table("predictions").update({
                            "actual_open": req.actual_open,
                            "actual_close": req.actual_close,
                            "actual_dir": actual_dir,
                            "correct": is_correct
                        }).eq("id", row_id).execute()
                        print(f"☁️ Supabase Cloud Result Updated for {target_symbol}")
                    else:
                        print(f"⚠️ No unmatched prediction found in Supabase for {target_symbol}")
                except Exception as ex:
                    print(f"⚠️ Supabase Log Result Error: {ex}")
        return {"status": "success"}
    except Exception as e:
        print(f"❌ Error in log_result: {e}")
        return {"status": "error", "detail": str(e)}

@app.get("/accuracy-stats")
async def get_accuracy_stats():
    """Reads the CSV and returns the total wins, losses, and win rate."""
    try:
        with csv_lock:
            if not os.path.isfile(LOG_FILE):
                 return {"wins": 0, "losses": 0, "win_rate": 0, "total": 0}
                 
            wins = 0
            losses = 0
            
            with open(LOG_FILE, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    correct = str(row.get("correct", "")).strip().upper()
                    if correct == "TRUE":
                        wins += 1
                    elif correct == "FALSE":
                        losses += 1
                        
            total = wins + losses
            win_rate = round((wins / total * 100), 1) if total > 0 else 0
            
            return {
                "wins": wins,
                "losses": losses,
                "win_rate": win_rate,
                "total": total
            }
    except Exception as e:
        print(f"❌ Error in accuracy-stats: {e}")
        return {"wins": 0, "losses": 0, "win_rate": 0, "total": 0}


def auto_label_outcomes():
    """
    Background thread: every 60s, checks the CSV for rows that are missing
    actual_dir/correct, fetches the latest CLOSED candle from Binance,
    and fills in TRUE/FALSE automatically.
    This makes labeling work even when NO browser is open!
    """
    import time
    import urllib.request
    import json

    def fetch_recent_candles(symbol, interval="3m", limit=30):
        """Fetch recent candles and return a dict keyed by open_time."""
        candles = {}
        try:
            url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
            with urllib.request.urlopen(url, timeout=8) as resp:
                data = json.loads(resp.read())
            
            # Binance format: [Open time, Open, High, Low, Close, Volume, Close time, ...]
            current_time_ms = time.time() * 1000
            for k in data:
                open_time = int(k[0])
                close_time = int(k[6])
                # Only use it if the candle is fully closed
                if current_time_ms > close_time:
                    candles[open_time] = (float(k[1]), float(k[4])) # open, close
        except Exception as e:
            print(f"[AutoLabel] Fetch error for {symbol}: {e}")
        return candles

    print("🤖 AutoLabel background thread started — will fill TRUE/FALSE using strict timestamps")

    from datetime import datetime

    while True:
        time.sleep(60)
        try:
            if not os.path.isfile(LOG_FILE):
                continue

            with csv_lock:
                with open(LOG_FILE, 'r') as f:
                    reader = csv.DictReader(f)
                    fieldnames = reader.fieldnames
                    rows = list(reader)

                pending_symbols = set()
                for row in rows:
                    if not str(row.get("actual_dir", "")).strip() and row.get("symbol", ""):
                        sym = re.sub(r'[^A-Z0-9]', '', str(row["symbol"]).upper())
                        if "USD" in sym and not sym.startswith("EUR") and not sym.startswith("GBP"):
                            pending_symbols.add(sym)

                if not pending_symbols:
                    continue

                actuals_cache = {}
                for sym in pending_symbols:
                    actuals_cache[sym] = fetch_recent_candles(sym, "3m", 50)

                labeled = 0
                for row in rows:
                    sym = re.sub(r'[^A-Z0-9]', '', str(row.get("symbol", "")).upper())
                    
                    # If this row needs labeling and we have data for this symbol
                    if not str(row.get("actual_dir", "")).strip() and sym in actuals_cache:
                        try:
                            # Parse CSV timestamp (e.g., "2026-03-14T05:52:08.781707")
                            dt = datetime.fromisoformat(row["timestamp"])
                            row_time_ms = int(dt.timestamp() * 1000)
                            
                            # Calculate the start time of the 3-minute candle this prediction was attempting to forecast
                            interval_ms = 3 * 60 * 1000
                            current_candle_start = (row_time_ms // interval_ms) * interval_ms
                            target_candle_start = current_candle_start + interval_ms # N+1 Candle
                            
                            binance_candles = actuals_cache[sym]
                            
                            # If the target candle has closed and exists in our history
                            if target_candle_start in binance_candles:
                                a_open, a_close = binance_candles[target_candle_start]
                                actual_dir = "UP" if a_close >= a_open else "DOWN"
                                pred_dir = str(row.get("predicted_dir", "")).strip().upper()
                                
                                row["actual_open"] = str(a_open)
                                row["actual_close"] = str(a_close)
                                row["actual_dir"] = actual_dir
                                row["correct"] = "TRUE" if pred_dir == actual_dir else "FALSE"
                                labeled += 1
                        except Exception as parse_error:
                            print(f"[AutoLabel] Date parsing error: {parse_error}")

                if labeled > 0:
                    with open(LOG_FILE, 'w', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(rows)
                    print(f"✅ [AutoLabel] Labeled {labeled} row(s) in CSV")

        except Exception as e:
            print(f"[AutoLabel] Error: {e}")


def start_loading():
    load_model()
    # Start auto-labeler AFTER model loads (so we know the server is healthy)
    threading.Thread(target=auto_label_outcomes, daemon=True).start()


@app.on_event("startup")
async def startup_event():
    threading.Thread(target=start_loading, daemon=True).start()


if __name__ == "__main__":
    import uvicorn
    # Important for Render Cloud: Must listen on 0.0.0.0 and whatever PORT they assign dynamically
    port = int(os.environ.get("PORT", 8000))
    print(f"\n🚀 Starting Carl AI Prediction Server on http://0.0.0.0:{port}")
    uvicorn.run("predict:app", host="0.0.0.0", port=port)
