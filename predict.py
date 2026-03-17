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
    from fastapi.responses import HTMLResponse
    html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Carl AI - Live Trading Dashboard</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap" rel="stylesheet">
<style>
  *{margin:0;padding:0;box-sizing:border-box}
  body{background:#0a0e1a;color:#f1f5f9;font-family:'Inter',sans-serif;min-height:100vh;padding:20px}
  .header{text-align:center;padding:30px 20px;border-bottom:1px solid rgba(255,255,255,0.08);margin-bottom:30px}
  .header h1{font-size:2rem;font-weight:800;background:linear-gradient(90deg,#6366f1,#a855f7);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
  .header p{color:#64748b;margin-top:8px;font-size:0.9rem}
  .status-bar{display:flex;gap:16px;justify-content:center;flex-wrap:wrap;margin-bottom:30px}
  .status-pill{background:rgba(99,102,241,0.15);border:1px solid rgba(99,102,241,0.3);border-radius:999px;padding:6px 16px;font-size:0.8rem;color:#a5b4fc;display:flex;align-items:center;gap:6px}
  .dot{width:7px;height:7px;border-radius:50%;background:#22c55e;animation:pulse 2s infinite}
  @keyframes pulse{0%,100%{opacity:1}50%{opacity:0.4}}
  .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:20px;margin-bottom:30px}
  .card{background:rgba(15,23,42,0.8);border:1px solid rgba(255,255,255,0.08);border-radius:16px;padding:20px;backdrop-filter:blur(10px)}
  .card h3{font-size:0.7rem;text-transform:uppercase;letter-spacing:0.1em;color:#64748b;margin-bottom:12px}
  .symbol-name{font-size:1.4rem;font-weight:800;letter-spacing:-0.02em}
  .price{font-size:1.1rem;font-family:monospace;color:#94a3b8;margin:4px 0}
  .direction{font-size:1.1rem;font-weight:700;padding:6px 14px;border-radius:8px;display:inline-block;margin-top:8px}
  .up{background:rgba(34,197,94,0.15);color:#22c55e;border:1px solid rgba(34,197,94,0.3)}
  .down{background:rgba(239,68,68,0.15);color:#ef4444;border:1px solid rgba(239,68,68,0.3)}
  .stats{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:30px}
  .stat{background:rgba(15,23,42,0.8);border:1px solid rgba(255,255,255,0.08);border-radius:12px;padding:16px;text-align:center}
  .stat .val{font-size:1.8rem;font-weight:800}
  .stat .lbl{font-size:0.7rem;color:#64748b;margin-top:4px;text-transform:uppercase;letter-spacing:0.05em}
  .table-card{background:rgba(15,23,42,0.8);border:1px solid rgba(255,255,255,0.08);border-radius:16px;padding:20px;overflow:hidden}
  table{width:100%;border-collapse:collapse;font-size:0.82rem}
  th{color:#64748b;font-weight:600;text-align:left;padding:8px 12px;border-bottom:1px solid rgba(255,255,255,0.06);font-size:0.7rem;text-transform:uppercase}
  td{padding:10px 12px;border-bottom:1px solid rgba(255,255,255,0.04)}
  tr:hover td{background:rgba(255,255,255,0.02)}
  .badge{padding:2px 8px;border-radius:6px;font-size:0.72rem;font-weight:700}
  .badge-up{background:rgba(34,197,94,0.15);color:#22c55e}
  .badge-down{background:rgba(239,68,68,0.15);color:#ef4444}
  .badge-correct{background:rgba(34,197,94,0.15);color:#22c55e}
  .badge-wrong{background:rgba(239,68,68,0.15);color:#ef4444}
  .badge-pending{background:rgba(100,116,139,0.15);color:#64748b}
  .refresh{text-align:center;margin-top:20px;color:#475569;font-size:0.75rem}
  #countdown{color:#6366f1;font-weight:700}
</style>
</head>
<body>
<div class="header">
  <h1>🤖 Carl AI Trading Dashboard</h1>
  <p>Autonomous 24/7 Cloud Trader — BTC · ETH · SOL</p>
</div>

<div class="status-bar">
  <div class="status-pill"><span class="dot"></span> Server Online</div>
  <div class="status-pill"><span class="dot"></span> Groq AI Connected</div>
  <div class="status-pill"><span class="dot"></span> Supabase Live</div>
  <div class="status-pill"><span class="dot"></span> Auto-Trading Every 3 Minutes</div>
</div>

<div class="stats" id="stats">
  <div class="stat"><div class="val" style="color:#6366f1">--</div><div class="lbl">Total Predictions</div></div>
  <div class="stat"><div class="val" style="color:#22c55e">--%</div><div class="lbl">Win Rate</div></div>
  <div class="stat"><div class="val" style="color:#a855f7">--</div><div class="lbl">Labeled Rows</div></div>
</div>

<div class="grid" id="live-cards">
  <div class="card"><h3>BTC/USDT</h3><div class="symbol-name" style="color:#f59e0b">₿ Bitcoin</div><p class="price">Fetching...</p></div>
  <div class="card"><h3>ETH/USDT</h3><div class="symbol-name" style="color:#6366f1">Ξ Ethereum</div><p class="price">Fetching...</p></div>
  <div class="card"><h3>SOL/USDT</h3><div class="symbol-name" style="color:#a855f7">◎ Solana</div><p class="price">Fetching...</p></div>
</div>

<div class="table-card">
  <h3 style="font-size:0.8rem;color:#94a3b8;margin-bottom:16px;font-weight:700">📊 Recent Predictions (Live from Supabase)</h3>
  <table>
    <thead><tr><th>Time</th><th>Symbol</th><th>Predicted</th><th>Actual</th><th>Result</th></tr></thead>
    <tbody id="table-body"><tr><td colspan="5" style="text-align:center;color:#64748b;padding:30px">Loading predictions...</td></tr></tbody>
  </table>
</div>

<div class="refresh">Auto-refreshing in <span id="countdown">30</span>s · <a href="/" style="color:#6366f1;text-decoration:none">Refresh Now</a></div>

<script>
const API = window.location.origin;
let countdown = 30;

async function loadPredictions() {
  try {
    const r = await fetch(API + '/api/recent-predictions');
    const data = await r.json();
    const tbody = document.getElementById('table-body');
    if (!data.predictions || data.predictions.length === 0) {
      tbody.innerHTML = '<tr><td colspan="5" style="text-align:center;color:#64748b;padding:30px">No predictions yet. AI is collecting data...</td></tr>';
      return;
    }
    // Stats
    const total = data.total || data.predictions.length;
    const wins = data.predictions.filter(p => p.correct === 'TRUE').length;
    const labeled = data.predictions.filter(p => p.correct).length;
    const wr = labeled > 0 ? Math.round(wins/labeled*100) : '--';
    const statsEl = document.getElementById('stats').children;
    statsEl[0].children[0].textContent = total;
    statsEl[1].children[0].textContent = wr + (wr !== '--' ? '%' : '');
    statsEl[2].children[0].textContent = labeled;

    tbody.innerHTML = data.predictions.map(p => {
      const t = new Date(p.timestamp).toLocaleString();
      const dirClass = p.predicted_dir === 'UP' ? 'badge-up' : 'badge-down';
      const actClass = p.actual_dir === 'UP' ? 'badge-up' : p.actual_dir === 'DOWN' ? 'badge-down' : 'badge-pending';
      const resClass = p.correct === 'TRUE' ? 'badge-correct' : p.correct === 'FALSE' ? 'badge-wrong' : 'badge-pending';
      const res = p.correct === 'TRUE' ? '✅ WIN' : p.correct === 'FALSE' ? '❌ LOSS' : '⏳ Pending';
      return \`<tr><td style="color:#475569;\${p.correct?'':''}">\${t}</td><td style="font-weight:700">\${p.symbol}</td><td><span class="badge \${dirClass}">\${p.predicted_dir || '--'}</span></td><td><span class="badge \${actClass}">\${p.actual_dir || '--'}</span></td><td><span class="badge \${resClass}">\${res}</span></td></tr>\`;
    }).join('');
  } catch(e) {
    document.getElementById('table-body').innerHTML = '<tr><td colspan="5" style="text-align:center;color:#ef4444;padding:16px">Error loading data from API</td></tr>';
  }
}

async function loadPrices() {
  const symbols = ['BTCUSDT','ETHUSDT','SOLUSDT'];
  const names = ['₿ Bitcoin','Ξ Ethereum','◎ Solana'];
  const colors = ['#f59e0b','#6366f1','#a855f7'];
  const labels = ['BTC/USDT','ETH/USDT','SOL/USDT'];
  const cards = document.getElementById('live-cards').children;
  for (let i = 0; i < symbols.length; i++) {
    try {
      const r = await fetch(\`https://api.binance.com/api/v3/ticker/price?symbol=\${symbols[i]}\`);
      const d = await r.json();
      const p = parseFloat(d.price).toLocaleString('en-US',{style:'currency',currency:'USD',minimumFractionDigits:2});
      cards[i].innerHTML = \`<h3>\${labels[i]}</h3><div class="symbol-name" style="color:\${colors[i]}">\${names[i]}</div><p class="price">\${p}</p><p style="font-size:0.7rem;color:#475569;margin-top:4px">Live from Binance</p>\`;
    } catch(e) {}
  }
}

loadPredictions();
loadPrices();
setInterval(() => {
  countdown--;
  document.getElementById('countdown').textContent = countdown;
  if (countdown <= 0) { countdown = 30; loadPredictions(); loadPrices(); }
}, 1000);
</script>
</body>
</html>"""
    return HTMLResponse(content=html)

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


@app.get("/api/recent-predictions")
async def get_recent_predictions():
    """Returns the last 50 predictions from Supabase for the dashboard UI."""
    try:
        if supabase is None:
            return {"predictions": [], "total": 0}
        res = supabase.table("predictions").select("*").order("timestamp", desc=True).limit(50).execute()
        count_res = supabase.table("predictions").select("id", count="exact").execute()
        return {
            "predictions": res.data or [],
            "total": count_res.count or len(res.data or [])
        }
    except Exception as e:
        print(f"❌ Error in recent-predictions: {e}")
        return {"predictions": [], "total": 0}


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
    # Start autonomous trading predictor loop
    threading.Thread(target=auto_predict_loop, daemon=True).start()


def auto_predict_loop():
    """
    🤖 AUTONOMOUS CLOUD TRADER
    Every 3 minutes, this function:
    1. Fetches the latest candle data from Binance for BTC, ETH, and SOL.
    2. Calculates all technical indicators (RSI, MACD, EMA, ADX, etc.)
    3. Builds a prompt and sends it to Groq (Llama 3 70B) for a prediction.
    4. Saves the result permanently to Supabase.
    No browser, no React, no Mac needed. Fully autonomous.
    """
    import time
    import urllib.request
    import json

    SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    INTERVAL = "3m"
    CANDLE_COUNT = 50

    def fetch_candles(symbol):
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={INTERVAL}&limit={CANDLE_COUNT}"
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                data = json.loads(resp.read())
            return [{
                "open": float(k[1]), "high": float(k[2]),
                "low": float(k[3]),  "close": float(k[4]),
                "volume": float(k[5])
            } for k in data]
        except Exception as e:
            print(f"[AutoPredict] Binance fetch error for {symbol}: {e}")
            return []

    def calc_rsi(closes, period=14):
        deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
        gains = [max(d, 0) for d in deltas[-period:]]
        losses = [max(-d, 0) for d in deltas[-period:]]
        ag = sum(gains) / period if period else 0
        al = sum(losses) / period if period else 0
        rs = ag / al if al != 0 else 100
        return round(100 - (100 / (1 + rs)), 2)

    def calc_ema(closes, period):
        if len(closes) < period:
            return closes[-1]
        k = 2 / (period + 1)
        ema = sum(closes[:period]) / period
        for c in closes[period:]:
            ema = c * k + ema * (1 - k)
        return round(ema, 6)

    def calc_macd(closes):
        ema12 = calc_ema(closes, 12)
        ema26 = calc_ema(closes, 26)
        macd_line = round(ema12 - ema26, 6)
        # Signal = 9-period EMA of MACD
        macd_series = []
        for i in range(9, len(closes)):
            e12 = calc_ema(closes[:i], 12)
            e26 = calc_ema(closes[:i], 26)
            macd_series.append(e12 - e26)
        signal = calc_ema(macd_series, 9) if len(macd_series) >= 9 else macd_line
        return macd_line, round(signal, 6), round(macd_line - signal, 6)

    def calc_adx(candles, period=14):
        if len(candles) < period + 1:
            return 25.0, 25.0, 25.0
        tr_list, pdm_list, ndm_list = [], [], []
        for i in range(1, len(candles)):
            h, l, pc = candles[i]['high'], candles[i]['low'], candles[i-1]['close']
            tr_list.append(max(h - l, abs(h - pc), abs(l - pc)))
            pdm_list.append(max(candles[i]['high'] - candles[i-1]['high'], 0))
            ndm_list.append(max(candles[i-1]['low'] - candles[i]['low'], 0))
        atr = sum(tr_list[-period:]) / period
        pdi = 100 * (sum(pdm_list[-period:]) / period) / atr if atr else 0
        ndi = 100 * (sum(ndm_list[-period:]) / period) / atr if atr else 0
        dx = 100 * abs(pdi - ndi) / (pdi + ndi) if (pdi + ndi) else 0
        return round(dx * 0.7 + 25 * 0.3, 2), round(pdi, 2), round(ndi, 2)  # smooth

    print("🤖 [AutoPredict] Autonomous Cloud Trader started — predicting every 3 minutes!")
    time.sleep(15)  # Wait for server to fully warm up

    while True:
        for symbol in SYMBOLS:
            try:
                candles = fetch_candles(symbol)
                if len(candles) < 30:
                    continue

                closes = [c['close'] for c in candles]
                current = candles[-1]
                prev = candles[-2]

                rsi = calc_rsi(closes)
                ema8 = calc_ema(closes, 8)
                ema21 = calc_ema(closes, 21)
                ema50 = calc_ema(closes, 50)
                ema200 = calc_ema(closes, 200) if len(closes) >= 200 else calc_ema(closes, len(closes))
                macd, macd_sig, macd_hist = calc_macd(closes)
                adx, adx_pos, adx_neg = calc_adx(candles)

                # Trend score (simple EMA stack vote)
                bulls = sum([ema8 > ema21, ema21 > ema50, ema50 > ema200, rsi > 50, macd > macd_sig])
                trend_score = round((bulls / 5) * 100 - 50, 2)

                math_dir = "UP" if trend_score > 0 else "DOWN"
                price = current['close']
                sym_display = symbol.replace("USDT", "/USDT")

                prompt = f"""{sym_display} | Interval: {INTERVAL}
Price:{price} | Last Close: {prev['close']} | Trend Score: {trend_score}
RSI: {rsi} | EMA 8: {ema8} | EMA 21: {ema21} | EMA 50: {ema50} | EMA 200: {ema200}
MACD: {macd} | MACD Signal: {macd_sig} | MACD Hist: {macd_hist}
ADX: {adx} | ADX+: {adx_pos} | ADX-: {adx_neg}
Math Bias: {math_dir}
Predict the next {INTERVAL} candle. Reply:
DIRECTION: UP or DOWN
OPEN: [price]
HIGH: [price]
LOW: [price]
CLOSE: [price]
STARS: [1-5]
PROBABILITY: Bullish X%, Bearish Y%"""

                if groq_client is None:
                    print(f"[AutoPredict] Groq not ready, skipping {symbol}")
                    continue

                print(f"\n🤖 [AutoPredict] Predicting {symbol} @ {price}...")

                # Call Groq
                system_prompt = "You are a professional crypto analyst. Be decisive. Reply in the exact format given."
                chat_completion = groq_client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    model="llama3-70b-8192",
                    temperature=0.40,
                    max_tokens=200,
                )
                reply = chat_completion.choices[0].message.content.strip()
                print(f"[AutoPredict] {symbol} → {reply[:60]}...")

                # XGBoost override (if model loaded)
                try:
                    if xgb_model is not None:
                        features = np.array([[price, trend_score, rsi, adx]])
                        probs = xgb_model.predict_proba(features)[0]
                        prob_down, prob_up = probs[0], probs[1]
                        if "DIRECTION: UP" in reply.upper() and prob_down > 0.60:
                            reply = reply.replace("DIRECTION: UP", "DIRECTION: DOWN")
                            reply += f"\n[ML Filter: {prob_down*100:.1f}% Bearish]"
                        elif "DIRECTION: DOWN" in reply.upper() and prob_up > 0.60:
                            reply = reply.replace("DIRECTION: DOWN", "DIRECTION: UP")
                            reply += f"\n[ML Filter: {prob_up*100:.1f}% Bullish]"
                except Exception:
                    pass

                # Log to Supabase + CSV
                with csv_lock:
                    log_prediction_data(prompt, reply, math_dir=math_dir)

            except Exception as e:
                print(f"[AutoPredict] Error for {symbol}: {e}")

        # Sleep until next 3-minute candle
        time.sleep(180)



@app.on_event("startup")
async def startup_event():
    threading.Thread(target=start_loading, daemon=True).start()


if __name__ == "__main__":
    import uvicorn
    # Important for Render Cloud: Must listen on 0.0.0.0 and whatever PORT they assign dynamically
    port = int(os.environ.get("PORT", 8000))
    print(f"\n🚀 Starting Carl AI Prediction Server on http://0.0.0.0:{port}")
    uvicorn.run("predict:app", host="0.0.0.0", port=port)
