# """
# app.py - Stock Picks MVP Backend (FastAPI)
# -----------------------------------------
# Usage:
#     uvicorn app:app --reload --host 127.0.0.1 --port 8000

# Endpoints:
# - POST /train
# - GET  /predict?ticker=XXX
# - GET  /top-picks?n=20
# - GET  /health
# """

# import os
# import joblib
# import logging
# from typing import List

# import pandas as pd
# import numpy as np
# import yfinance as yf
# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import roc_auc_score

# # --- Config ---
# MODEL_PATH = "model.joblib"
# WATCHLIST = [
#     "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META",
#     "TSLA", "JPM", "V", "MA", "INTC", "AMD"
# ]
# FORWARD_DAYS = 5
# RETURN_THRESHOLD = 0.02
# RANDOM_STATE = 42

# # --- Logging ---
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger("stock-picks-backend")

# # --- FastAPI app ---
# app = FastAPI(title="Stock Picks MVP API")

# # --- CORS: allow dev origins (add/remove as needed) ---
# origins = [
#     "http://localhost:5173",
#     "http://127.0.0.1:5173",
#     "http://localhost:5177",
#     "http://127.0.0.1:5177",
#     "http://localhost:5178",
#     "http://127.0.0.1:5178",
#     "http://localhost:3000",
#     "http://127.0.0.1:3000",
# ]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,     # change to ["*"] temporarily only for debugging if necessary
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # -----------------------------
# # Data fetch & feature pipeline
# # -----------------------------
# def fetch_prices(ticker: str, period: str = "2y", interval: str = "1d"):
#     """
#     Download OHLCV data for `ticker`.
#     Explicitly set auto_adjust=False to avoid YFinance future behavior change.
#     Returns DataFrame with 'Adj_Close' column (if possible).
#     """
#     try:
#         df = yf.download(
#             ticker,
#             period=period,
#             interval=interval,
#             progress=False,
#             auto_adjust=False  # explicit to avoid future warnings/behavior changes
#         )
#     except Exception as e:
#         logger.warning("yfinance download failed for %s: %s", ticker, e)
#         return None

#     if df is None or df.empty:
#         return None

#     # Some yfinance responses may lack 'Adj Close' column; fall back to 'Close'
#     if "Adj Close" not in df.columns:
#         df["Adj Close"] = df.get("Close", np.nan)

#     # Keep only expected columns (if present)
#     cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
#     present = [c for c in cols if c in df.columns]
#     df = df[present].copy()

#     # Normalize column name for internal usage
#     if "Adj Close" in df.columns:
#         df.rename(columns={"Adj Close": "Adj_Close"}, inplace=True)
#     else:
#         # If we still don't have Adj Close, bail out
#         if "Close" in df.columns:
#             df["Adj_Close"] = df["Close"]
#         else:
#             return None

#     df.index = pd.to_datetime(df.index)
#     df.sort_index(inplace=True)
#     return df


# def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Add a few technical features used by the model.
#     Returns DataFrame with new feature columns and drops NaNs produced by rolling windows.
#     """
#     df = df.copy()
#     # returns
#     df["return_1"] = df["Adj_Close"].pct_change(1)
#     df["return_5"] = df["Adj_Close"].pct_change(5)
#     # moving averages
#     df["ma_5"] = df["Adj_Close"].rolling(5).mean()
#     df["ma_10"] = df["Adj_Close"].rolling(10).mean()
#     df["ema_10"] = df["Adj_Close"].ewm(span=10, adjust=False).mean()
#     # RSI approximation
#     delta = df["Adj_Close"].diff()
#     up = delta.clip(lower=0)
#     down = -delta.clip(upper=0)
#     roll_up = up.rolling(14).mean()
#     roll_down = down.rolling(14).mean()
#     rs = roll_up / (roll_down + 1e-9)
#     df["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))
#     # volume rank
#     df["vol_rank_10"] = df["Volume"].rolling(10).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
#     return df.dropna()


# def create_labels(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Create binary label: 1 if forward FORWARD_DAYS return > RETURN_THRESHOLD else 0.
#     """
#     df = df.copy()
#     df["future_close"] = df["Adj_Close"].shift(-FORWARD_DAYS)
#     df["fwd_return"] = (df["future_close"] - df["Adj_Close"]) / df["Adj_Close"]
#     df["label"] = (df["fwd_return"] > RETURN_THRESHOLD).astype(int)
#     return df.dropna()


# # -----------------------------
# # Training & model persistence
# # -----------------------------
# def build_dataset(tickers: List[str]) -> pd.DataFrame:
#     rows = []
#     for t in tickers:
#         df = fetch_prices(t)
#         if df is None:
#             logger.info("no data for %s", t)
#             continue
#         df = add_indicators(df)
#         df = create_labels(df)
#         if df.empty:
#             continue
#         feature_cols = ["return_1", "return_5", "ma_5", "ma_10", "ema_10", "rsi_14", "vol_rank_10"]
#         tmp = df[feature_cols].copy()
#         tmp["ticker"] = t
#         tmp["label"] = df["label"].values
#         tmp["date"] = df.index
#         rows.append(tmp)
#     if not rows:
#         raise RuntimeError("No valid training rows - check your tickers / data source")
#     data = pd.concat(rows, ignore_index=True)
#     data = data.dropna()
#     data = data.sample(frac=1, random_state=RANDOM_STATE)  # shuffle
#     return data


# def train_and_save_model():
#     data = build_dataset(WATCHLIST)
#     feature_cols = ["return_1", "return_5", "ma_5", "ma_10", "ema_10", "rsi_14", "vol_rank_10"]
#     X = data[feature_cols].values
#     y = data["label"].values

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y)

#     clf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=RANDOM_STATE, class_weight="balanced")
#     clf.fit(X_train, y_train)

#     y_proba = clf.predict_proba(X_test)[:, 1]
#     try:
#         auc = float(roc_auc_score(y_test, y_proba))
#     except Exception:
#         auc = None

#     joblib.dump({"model": clf, "features": feature_cols}, MODEL_PATH)
#     return {"auc": auc, "n_samples": len(y), "positives": int(y.sum())}


# def load_model():
#     if not os.path.exists(MODEL_PATH):
#         return None
#     return joblib.load(MODEL_PATH)


# # -----------------------------
# # API models & endpoints
# # -----------------------------
# class PredictResponse(BaseModel):
#     ticker: str
#     date: str
#     probability: float
#     decision: str


# @app.post("/train")
# def train_endpoint():
#     try:
#         info = train_and_save_model()
#         logger.info("model trained: %s", info)
#         return {"status": "ok", "info": info}
#     except Exception as e:
#         logger.exception("training failed")
#         raise HTTPException(status_code=500, detail=str(e))


# @app.get("/predict", response_model=PredictResponse)
# def predict_endpoint(ticker: str):
#     model_obj = load_model()
#     if model_obj is None:
#         raise HTTPException(status_code=400, detail="Model not trained. Call /train first.")
#     model = model_obj["model"]
#     feature_cols = model_obj["features"]

#     df = fetch_prices(ticker, period="6mo")
#     if df is None:
#         raise HTTPException(status_code=404, detail="Ticker data not found")
#     df = add_indicators(df)
#     if df.empty:
#         raise HTTPException(status_code=404, detail="Not enough data for indicators")
#     last = df.iloc[-1]
#     x = last[feature_cols].values.reshape(1, -1)
#     proba = float(model.predict_proba(x)[0, 1])
#     decision = "Strong Buy" if proba >= 0.85 else ("Buy" if proba >= 0.65 else "No Action")
#     return PredictResponse(ticker=ticker, date=str(last.name.date()), probability=proba, decision=decision)


# @app.get("/top-picks")
# def top_picks(n: int = 20, tickers: List[str] = None):
#     model_obj = load_model()
#     if model_obj is None:
#         raise HTTPException(status_code=400, detail="Model not trained. Call /train first.")
#     if tickers is None:
#         tickers = WATCHLIST
#     model = model_obj["model"]
#     feature_cols = model_obj["features"]
#     results = []
#     for t in tickers:
#         df = fetch_prices(t, period="6mo")
#         if df is None:
#             continue
#         df = add_indicators(df)
#         if df.empty:
#             continue
#         last = df.iloc[-1]
#         x = last[feature_cols].values.reshape(1, -1)
#         proba = float(model.predict_proba(x)[0, 1])
#         decision = "Strong Buy" if proba >= 0.85 else ("Buy" if proba >= 0.65 else "No Action")
#         results.append({"ticker": t, "date": str(last.name.date()), "prob": proba, "decision": decision})
#     results = sorted(results, key=lambda r: r["prob"], reverse=True)[:n]
#     return {"results": results}


# @app.get("/health")
# def health():
#     return {"status": "ok", "model_exists": os.path.exists(MODEL_PATH)}

"""
app.py - Stock Picks MVP Backend (FastAPI) with extended indicators, sentiment, fair-value, and optional DL model
Usage:
    uvicorn app:app --reload --host 127.0.0.1 --port 8000
Endpoints:
- POST /train            -> trains RandomForest and optional LSTM (if tensorflow installed)
- GET  /predict?ticker=X -> returns RF prob, DL pred(if available), indicators, fib levels, fundamentals, fair_value, sentiment
- GET  /top-picks?n=20   -> top picks from WATCHLIST (RF probabilities)
- GET  /health
"""
import os
import joblib
import logging
from typing import List, Dict, Any, Optional

import pandas as pd
import numpy as pd  
import yfinance as yf

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# third-party indicator lib
try:
    import ta
except Exception:
    ta = None

# sentiment
# at top of file, after imports



# optional deep learning
try:
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

# config
MODEL_PATH = "model_rf.joblib"
MODEL_DL_PATH = "model_lstm.h5"
WATCHLIST = ["AAPL","MSFT","AMZN","NVDA","GOOGL","META","TSLA","JPM","V","MA","INTC","AMD"]
FORWARD_DAYS = 5
RETURN_THRESHOLD = 0.02
RANDOM_STATE = 42

# add near top with other imports/helpers
import pandas as pd
import numpy as np

def safe_float(x):
    """
    Convert x to a Python float in a safe way:
    - if x is a single-element Series -> use x.iloc[0]
    - if x is a single-element DataFrame -> take .iloc[0,0]
    - if x is scalar -> convert to float
    - if x is NaN/None -> return None
    """
    try:
        # DataFrame -> first element
        if isinstance(x, pd.DataFrame):
            if x.size == 0:
                return None
            val = x.iloc[0, 0]
        # Series -> first element
        elif isinstance(x, pd.Series):
            if x.size == 0:
                return None
            val = x.iloc[0]
        else:
            val = x

        # handle pandas/numpy missing values
        if pd.isna(val):
            return None
        return float(val)
    except Exception:
        # fallback: try to coerce with numpy
        try:
            arr = np.asarray(x)
            if arr.size == 0:
                return None
            return float(arr.flatten()[0])
        except Exception:
            return None


# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("stock-picks-backend-ext")

app = FastAPI(title="Stock Picks MVP API - extended")

# --- FinBERT (optional) integration ---
# Requires: pip install transformers torch feedparser

_FINBERT = None
_FINBERT_TOKENIZER = None
_FINBERT_MODEL_NAME = "yiyanghkust/finbert-tone"  # default; can change to another FinBERT checkpoint
_FINBERT_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch

    def _init_finbert(model_name=_FINBERT_MODEL_NAME):
        global _FINBERT, _FINBERT_TOKENIZER, _FINBERT_AVAILABLE
        if _FINBERT is not None and _FINBERT_TOKENIZER is not None:
            _FINBERT_AVAILABLE = True
            return

        try:
            _FINBERT_TOKENIZER = AutoTokenizer.from_pretrained(model_name)
            _FINBERT = AutoModelForSequenceClassification.from_pretrained(model_name)
            # Move to CPU by default. If you have GPU and want to use it, change to .to("cuda")
            _FINBERT.to("cpu")
            _FINBERT_AVAILABLE = True
        except Exception as e:
            logger.warning("FinBERT init failed: %s", e)
            _FINBERT_AVAILABLE = False

    # lazy init - don't call _init_finbert() on import; call it when needed
    _FINBERT_AVAILABLE = True  # will be validated on first call by attempting to init
except Exception as e:
    # transformers / torch not available
    logger.info("transformers/torch not installed, FinBERT disabled: %s", e)
    _FINBERT_AVAILABLE = False




try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    try:
        vader = SentimentIntensityAnalyzer()
    except Exception as e:
        # initialization failed (rare) — log and set to None
        logger.warning("VADER init failed: %s", e)
        vader = None
except Exception as e:
    # package not installed or import error
    logger.info("vaderSentiment not available: %s", e)
    vader = None

# CORS - keep your dev origins here
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:5177",
    "http://127.0.0.1:5177",
    "http://localhost:5178",
    "http://127.0.0.1:5178",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# ---------- helpers: data + indicators ----------
def score_text_finbert(text: str) -> dict:
    """
    Score `text` with FinBERT model if available.
    Returns: { available: bool, label: str|null, scores: dict|null, error: str|null }
    Example `scores`: {"positive": 0.12, "neutral": 0.55, "negative": 0.33}
    """
    out = {"available": False, "label": None, "scores": None, "error": None}
    if not text or not isinstance(text, str) or text.strip() == "":
        out["error"] = "empty text"
        return out

    # initialize if not ready
    try:
        if not _FINBERT_AVAILABLE or _FINBERT is None:
            _init_finbert()
    except Exception as e:
        out["error"] = f"finbert init failed: {e}"
        return out

    if not _FINBERT_AVAILABLE or _FINBERT is None or _FINBERT_TOKENIZER is None:
        out["error"] = "FinBERT not available"
        return out

    try:
        # prepare input
        enc = _FINBERT_TOKENIZER(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        # forward
        with torch.no_grad():
            logits = _FINBERT(**{k: v.to(_FINBERT.device) for k, v in enc.items()})[0]
            probs = torch.softmax(logits, dim=-1).cpu().numpy().flatten()
        # label mapping depends on checkpoint:
        # For 'yiyanghkust/finbert-tone' mapping is usually: [neutral, positive, negative] or different.
        # To be robust, try to read model.config.id2label if present.
        labels = None
        try:
            id2label = getattr(_FINBERT.config, "id2label", None)
            if id2label:
                # id2label is dict like {0: 'neutral', 1: 'positive', 2: 'negative'}
                labels = [id2label[i].lower() for i in range(len(probs))]
            else:
                # fallback label order - user may need to adjust depending on checkpoint used
                labels = ["neutral", "positive", "negative"] if len(probs) == 3 else [f"label_{i}" for i in range(len(probs))]
        except Exception:
            labels = ["label_%d" % i for i in range(len(probs))]

        scores = {labels[i]: float(probs[i]) for i in range(len(probs))}
        # choose the highest prob label
        best_idx = int(probs.argmax())
        out["available"] = True
        out["label"] = labels[best_idx]
        out["scores"] = scores
        return out
    except Exception as e:
        out["error"] = f"scoring error: {e}"
        logger.exception("FinBERT scoring error: %s", e)
        return out

def score_text(text: str) -> dict:
    """
    Safely score `text` using VADER if available.
    Returns a dict with guaranteed keys: { 'available': bool, 'compound': float|null, 'pos': float|null, 'neu': float|null, 'neg': float|null, 'error': str|null }
    """
    out = {"available": False, "compound": None, "pos": None, "neu": None, "neg": None, "error": None}
    if not vader:
        out["error"] = "vader not available"
        return out

    if not isinstance(text, str) or text.strip() == "":
        out["error"] = "empty or non-string text"
        return out

    try:
        scores = vader.polarity_scores(text)
        # scores should be a dict with keys: 'neg','neu','pos','compound'
        # Validate and coerce values to floats
        out["compound"] = float(scores.get("compound")) if scores.get("compound") is not None else None
        out["pos"] = float(scores.get("pos")) if scores.get("pos") is not None else None
        out["neu"] = float(scores.get("neu")) if scores.get("neu") is not None else None
        out["neg"] = float(scores.get("neg")) if scores.get("neg") is not None else None
        out["available"] = True
        return out
    except Exception as e:
        out["error"] = f"scoring failed: {e}"
        logger.debug("VADER scoring error for text [%s]: %s", text[:120], e)
        return out

def fetch_prices(ticker: str, period: str = "2y", interval: str = "1d") -> Optional[pd.DataFrame]:
    """Download OHLCV with explicit auto_adjust=False; return DataFrame with Adj_Close."""
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False)
    except Exception as e:
        logger.warning("yfinance download failed for %s: %s", ticker, e)
        return None
    if df is None or df.empty:
        return None
    if "Adj Close" not in df.columns:
        df["Adj Close"] = df.get("Close", np.nan)
    cols = ["Open","High","Low","Close","Adj Close","Volume"]
    present = [c for c in cols if c in df.columns]
    df = df[present].copy()
    if "Adj Close" in df.columns:
        df.rename(columns={"Adj Close":"Adj_Close"}, inplace=True)
    else:
        if "Close" in df.columns:
            df["Adj_Close"] = df["Close"]
        else:
            return None
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    return df

# replace compute_technical_indicators with this version
def compute_technical_indicators(df: pd.DataFrame) -> Dict[str, float]:
    """Return a dictionary of many technical indicators computed from df (must have Adj_Close, High, Low, Volume)."""
    out = {}
    d = df.copy().dropna()
    if d.empty:
        # return keys with None to keep shape consistent
        keys = ["return_1","return_5","ma_5","ma_10","ema_10","rsi_14",
                "bb_mavg","bb_h","bb_l","macd","macd_sig","atr_14",
                "stoch_k","stoch_d","vol_rank_10","last_price"]
        return {k: None for k in keys}

    price = d["Adj_Close"]
    high = d["High"] if "High" in d.columns else price
    low = d["Low"] if "Low" in d.columns else price
    vol = d["Volume"] if "Volume" in d.columns else None

    # Returns (use safe_float)
    out["return_1"] = safe_float(price.pct_change(1).iloc[-1]) if len(price) > 1 else None
    out["return_5"] = safe_float(price.pct_change(5).iloc[-1]) if len(price) > 5 else None

    # Moving averages
    out["ma_5"] = safe_float(price.rolling(5).mean().iloc[-1]) if len(price) >= 5 else None
    out["ma_10"] = safe_float(price.rolling(10).mean().iloc[-1]) if len(price) >= 10 else None
    out["ema_10"] = safe_float(price.ewm(span=10, adjust=False).mean().iloc[-1]) if len(price) >= 1 else None

    # RSI
    try:
        delta = price.diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        roll_up = up.rolling(14).mean()
        roll_down = down.rolling(14).mean()
        with np.errstate(divide="ignore", invalid="ignore"):
            rs = roll_up / (roll_down + 1e-9)
            rsi_series = 100.0 - (100.0 / (1.0 + rs))
            out["rsi_14"] = safe_float(rsi_series.iloc[-1])
    except Exception:
        out["rsi_14"] = None

    # Bollinger Bands
    try:
        m = price.rolling(20).mean()
        s = price.rolling(20).std()
        out["bb_mavg"] = safe_float(m.iloc[-1]) if len(m) >= 20 else None
        out["bb_h"] = safe_float((m + 2 * s).iloc[-1]) if len(m) >= 20 else None
        out["bb_l"] = safe_float((m - 2 * s).iloc[-1]) if len(m) >= 20 else None
    except Exception:
        out.update({"bb_mavg": None, "bb_h": None, "bb_l": None})

    # MACD (best-effort - may be None if ta not installed)
    try:
        if ta:
            macd = ta.trend.MACD(price)
            out["macd"] = safe_float(macd.macd().iloc[-1])
            out["macd_sig"] = safe_float(macd.macd_signal().iloc[-1])
        else:
            out["macd"] = None
            out["macd_sig"] = None
    except Exception:
        out["macd"] = None
        out["macd_sig"] = None

    # ATR
    try:
        if ta:
            out["atr_14"] = safe_float(ta.volatility.AverageTrueRange(high, low, price, window=14).average_true_range().iloc[-1])
        else:
            out["atr_14"] = None
    except Exception:
        out["atr_14"] = None

    # Stochastic %K/%D
    try:
        if ta:
            st = ta.momentum.StochasticOscillator(high, low, price, window=14, smooth_window=3)
            out["stoch_k"] = safe_float(st.stoch().iloc[-1])
            out["stoch_d"] = safe_float(st.stoch_signal().iloc[-1])
        else:
            out["stoch_k"] = None
            out["stoch_d"] = None
    except Exception:
        out["stoch_k"] = None
        out["stoch_d"] = None

    # Volume rank (10)
    try:
        if vol is not None and len(vol) >= 10:
            vr = vol.rolling(10).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
            out["vol_rank_10"] = safe_float(vr.iloc[-1])
        else:
            out["vol_rank_10"] = None
    except Exception:
        out["vol_rank_10"] = None

    # last price
    out["last_price"] = safe_float(price.iloc[-1]) if len(price) else None

    return out

# replace compute_fibonacci_levels with this version
def compute_fibonacci_levels(df: pd.DataFrame, lookback: int = 60) -> Dict[str, float]:
    """
    Compute Fibonacci retracement levels (uses lookback days).
    Uses safe_float to avoid deprecation warnings.
    """
    try:
        recent = df.tail(lookback)
        if recent.empty:
            return {}
        swing_high = safe_float(recent["High"].max())
        swing_low = safe_float(recent["Low"].min())
        if swing_high is None or swing_low is None:
            return {}
        diff = swing_high - swing_low
        levels = {
            "high": swing_high,
            "low": swing_low,
            "0.0": swing_high,
            "23.6": swing_high - 0.236 * diff,
            "38.2": swing_high - 0.382 * diff,
            "50.0": swing_high - 0.5 * diff,
            "61.8": swing_high - 0.618 * diff,
            "100.0": swing_low
        }
        return levels
    except Exception:
        return {}


# fundamentals & fair value
def fetch_fundamentals(ticker: str) -> Dict[str, Any]:
    """Use yfinance Ticker.info to grab some fundamentals (may be incomplete)."""
    try:
        tk = yf.Ticker(ticker)
        info = tk.get_info() if hasattr(tk, "get_info") else tk.info
    except Exception as e:
        logger.debug("fundamentals fetch error %s: %s", ticker, e)
        info = {}
    keys = ["longName","sector","industry","marketCap","trailingPE","forwardPE","epsTrailingTwelveMonths","priceToBook","beta","dividendYield"]
    out = {k: info.get(k) for k in keys}
    out["raw"] = info  # include full info if needed (careful, can be large)
    return out

def estimate_fair_value(fund: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simple fair value heuristic:
    - If EPS available, fair_price = EPS * baseline_PE (15)
    - Return both estimate and method note.
    """
    eps = fund.get("epsTrailingTwelveMonths") or fund.get("epsCurrentYear") or fund.get("earningsPerShare")
    baseline_pe = 15.0
    try:
        eps_val = float(eps) if eps is not None else None
    except Exception:
        eps_val = None
    fair = None
    note = "Heuristic: fair_price = EPS * baseline_PE (baseline_PE=15). Not a full DCF."
    if eps_val is not None:
        fair = float(eps_val * baseline_pe)
    return {"fair_value": fair, "method": note, "baseline_pe": baseline_pe, "eps": eps_val}


# sentiment
# simple in-memory per-process cache: { ticker: { 'timestamp': ts, 'result': ... } }

import feedparser
import time
_FINBERT_SENT_CACHE = {}
_FINBERT_SENT_TTL = 60 * 60  # 1 hour cache

def fetch_news_sentiment_finbert(ticker: str, max_items: int = 8, use_cache: bool = True):
    """
    Fetch headlines from Yahoo Finance RSS and score them with FinBERT.
    Returns: { available: bool, count, avg_label_confidence (dict), details: [ {title, finbert: {...}} ... ] }
    """
    now = int(time.time())
    if use_cache:
        cached = _FINBERT_SENT_CACHE.get(ticker)
        if cached and now - cached["ts"] < _FINBERT_SENT_TTL:
            return cached["val"]

    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
    try:
        feed = feedparser.parse(url)
        entries = getattr(feed, "entries", []) or []
    except Exception as e:
        logger.debug("RSS fetch failed for %s: %s", ticker, e)
        entries = []

    details = []
    for entry in entries[:max_items]:
        title = getattr(entry, "title", "") or entry.get("title", "")
        if not title:
            continue
        fin = score_text_finbert(title)
        details.append({"title": title, "finbert": fin})

    # compute aggregated scores: average of each label probability if available
    all_scores = {}
    count_scores = 0
    for d in details:
        s = d["finbert"].get("scores") if d["finbert"] else None
        if s:
            count_scores += 1
            for k, v in s.items():
                all_scores[k] = all_scores.get(k, 0.0) + float(v)
    avg_scores = {k: (v / count_scores) for k, v in all_scores.items()} if count_scores else None

    res = {"available": True, "count": len(details), "avg_scores": avg_scores, "details": details}
    if use_cache:
        _FINBERT_SENT_CACHE[ticker] = {"ts": now, "val": res}
    return res

_SENTIMENT_CACHE = {}
_SENTIMENT_TTL = 60 * 60  # 1 hour

def fetch_news_sentiment(ticker: str, max_items: int = 8, use_cache: bool = True) -> dict:
    """
    Fetch recent news via yfinance and compute VADER sentiment safely.
    Returns a dict:
    {
      available: bool,
      reason: str|null,
      avg_compound: float|null,
      count: int,
      details: [ { title, compound, pos, neu, neg, provider } ... ]
    }
    """
    # cache check
    import time
    now = int(time.time())
    if use_cache:
        cached = _SENTIMENT_CACHE.get(ticker)
        if cached and now - cached["timestamp"] < _SENTIMENT_TTL:
            return cached["result"]

    if not vader:
        res = {"available": False, "reason": "vaderSentiment not installed or failed to init", "avg_compound": None, "count": 0, "details": []}
        if use_cache:
            _SENTIMENT_CACHE[ticker] = {"timestamp": now, "result": res}
        return res

    # fetch news from yfinance safely
    try:
        tk = yf.Ticker(ticker)
        news = []
        # yfinance offers .news or .get_news depending on version — handle both
        if hasattr(tk, "news"):
            news = tk.news or []
        elif hasattr(tk, "get_news"):
            news = tk.get_news() or []
        else:
            news = []
    except Exception as e:
        logger.debug("yfinance news fetch failed for %s: %s", ticker, e)
        news = []

    details = []
    for item in (news if isinstance(news, list) else [])[:max_items]:
        # try multiple possible title keys
        title = item.get("title") or item.get("headline") or item.get("summary") or item.get("short") or ""
        provider = None
        if isinstance(item.get("publisher"), dict):
            provider = item.get("publisher").get("name")
        else:
            provider = item.get("publisher") or item.get("provider") or item.get("source")

        if not title:
            continue
        score = score_text(title)
        details.append({
            "title": title,
            "compound": score["compound"],
            "pos": score["pos"],
            "neu": score["neu"],
            "neg": score["neg"],
            "provider": provider or "",
            "error": score["error"]
        })

    compounds = [d["compound"] for d in details if d["compound"] is not None]
    avg_compound = float(np.mean(compounds)) if compounds else None

    res = {"available": True, "reason": None, "avg_compound": avg_compound, "count": len(details), "details": details}
    if use_cache:
        _SENTIMENT_CACHE[ticker] = {"timestamp": now, "result": res}
    return res


# ------------- ML: features, RF model, optional DL LSTM -------------
FEATURE_COLS = ["return_1","return_5","ma_5","ma_10","ema_10","rsi_14","vol_rank_10"]

def build_dataset(tickers: List[str]) -> pd.DataFrame:
    """
    Robust dataset builder:
    - downloads prices for each ticker
    - computes simple features per date
    - aligns forward label (FORWARD_DAYS)
    - returns concatenated DataFrame with FEATURE_COLS + label + date + ticker
    """
    rows = []
    for t in tickers:
        try:
            df = fetch_prices(t)
            if df is None:
                logger.info("no price data for %s", t)
                continue

            # Ensure numeric columns present
            if "Adj_Close" not in df.columns or "Volume" not in df.columns:
                logger.info("missing required columns for %s", t)
                continue

            # Compute features (vectorized)
            tmp = df.copy()

            # basic returns and rolling stats
            tmp["return_1"] = tmp["Adj_Close"].pct_change(1)
            tmp["return_5"] = tmp["Adj_Close"].pct_change(5)
            tmp["ma_5"] = tmp["Adj_Close"].rolling(5).mean()
            tmp["ma_10"] = tmp["Adj_Close"].rolling(10).mean()
            tmp["ema_10"] = tmp["Adj_Close"].ewm(span=10, adjust=False).mean()

            # RSI (robust)
            delta = tmp["Adj_Close"].diff()
            up = delta.clip(lower=0)
            down = -delta.clip(upper=0)
            roll_up = up.rolling(14).mean()
            roll_down = down.rolling(14).mean()
            with np.errstate(divide="ignore", invalid="ignore"):
                rs = roll_up / (roll_down + 1e-9)
                tmp["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))

            # volume rank
            tmp["vol_rank_10"] = tmp["Volume"].rolling(10).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
            )

            # drop rows that don't have full features
            tmp = tmp.dropna()
            if tmp.empty:
                logger.info("no usable feature rows for %s after dropna", t)
                continue

            # Build forward returns safely as 1D numpy arrays
            adj = tmp["Adj_Close"].astype(float)
            future_close = adj.shift(-FORWARD_DAYS)  # Series
            # Use numpy arrays to avoid accidental DataFrame assignment
            fwd_return = (future_close.values - adj.values) / (adj.values + 1e-12)  # 1D numpy array

            # Determine valid indices (we cannot use the last FORWARD_DAYS rows)
            valid_len = len(tmp) - FORWARD_DAYS
            if valid_len <= 0:
                logger.info("not enough rows after forward shift for %s", t)
                continue

            # Select only rows that have a corresponding forward label
            feat_df = tmp.iloc[:valid_len].copy()  # KEEP THE DATETIME INDEX
            # set label from numpy slice (safe)
            feat_df["label"] = (fwd_return[:valid_len] > RETURN_THRESHOLD).astype(int)
            feat_df["ticker"] = t
            # create explicit date column from the index (doesn't depend on index name)
            feat_df["date"] = feat_df.index

            # ensure FEATURE_COLS exist (if any missing, fill with zeros)
            for c in FEATURE_COLS:
                if c not in feat_df.columns:
                    feat_df[c] = 0.0

            # keep only necessary columns
            cols_keep = FEATURE_COLS + ["label", "ticker", "date"]
            rows.append(feat_df[cols_keep].reset_index(drop=True).copy())

        except Exception as e:
            logger.exception("error building rows for %s: %s", t, e)
            continue

    if not rows:
        raise RuntimeError("No training data rows. Check tickers and data availability.")

    data = pd.concat(rows, ignore_index=True)
    data = data.dropna(axis=0, how="any")
    data = data.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    return data

def train_rf(data: pd.DataFrame):
    X = data[FEATURE_COLS].values
    y = data["label"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y if len(np.unique(y))>1 else None)
    clf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=RANDOM_STATE, class_weight="balanced")
    clf.fit(X_train, y_train)
    y_proba = clf.predict_proba(X_test)[:,1]
    auc = None
    try:
        auc = float(roc_auc_score(y_test, y_proba))
    except Exception:
        auc = None
    joblib.dump({"model": clf, "features": FEATURE_COLS}, MODEL_PATH)
    return {"auc": auc, "samples": len(y), "positives": int(y.sum())}

# Optional: simple LSTM for regression of next-day return (small toy model)
def train_lstm(data: pd.DataFrame, seq_len: int = 20, epochs: int = 8, batch_size: int = 32):
    if not TF_AVAILABLE:
        return {"skipped": True, "reason": "tensorflow not installed"}
    # We'll build sequences using closing price only for simplicity
    sequences = []
    targets = []
    grouped = data.groupby("ticker")
    for t, g in grouped:
        g_sorted = g.sort_values("date")
        prices = g_sorted["ma_5"].values  # using ma_5 as a proxy feature for sequence; change as needed
        if len(prices) < seq_len + 1:
            continue
        for i in range(len(prices)-seq_len):
            seq = prices[i:i+seq_len]
            target = prices[i+seq_len]  # predict next value
            sequences.append(seq)
            targets.append(target)
    if not sequences:
        return {"skipped": True, "reason": "not enough sequence data"}
    X = np.array(sequences).reshape(-1, seq_len, 1)
    y = np.array(targets).reshape(-1, 1)
    # normalize
    mean = X.mean(); std = X.std() + 1e-9
    X = (X - mean) / std
    y = (y - mean) / std
    # simple LSTM model
    model = Sequential()
    model.add(LSTM(64, input_shape=(seq_len, 1), return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(1, activation="linear"))
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)
    model.save(MODEL_DL_PATH)
    return {"trained": True, "samples": X.shape[0]}

def load_rf_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)

def load_lstm_model():
    if TF_AVAILABLE and os.path.exists(MODEL_DL_PATH):
        try:
            return tf.keras.models.load_model(MODEL_DL_PATH)
        except Exception as e:
            logger.warning("Failed to load LSTM model: %s", e)
            return None
    return None

# -------------- API models & endpoints ----------------
class PredictResponse(BaseModel):
    ticker: str
    date: str
    probability: float
    decision: str

@app.post("/train")
def train_endpoint(train_dl: bool = False):
    """Train RF; optionally train DL model if train_dl=True and tensorflow available"""
    try:
        data = build_dataset(WATCHLIST)
    except Exception as e:
        raise HTTPException(500, detail=f"Failed to build dataset: {e}")
    rf_info = train_rf(data)
    dl_info = None
    if train_dl:
        dl_info = train_lstm(data)
    return {"status":"ok", "rf_info": rf_info, "dl_info": dl_info}

@app.get("/predict")
def predict_endpoint(ticker: str):
    """
    Returns:
    - RandomForest probability + decision
    - DL model prediction (if available)
    - indicators, fibonacci levels, fundamentals, fair_value, sentiment summary
    """
    
    model_obj = load_rf_model()
    if model_obj is None:
        raise HTTPException(status_code=400, detail="Model not trained. Call /train first.")
    model = model_obj["model"]
    features = model_obj["features"]

    finbert_sent = None
    try:
     finbert_sent = fetch_news_sentiment_finbert(ticker, max_items=8)
    except Exception as e:
     logger.debug("FinBERT sentiment fetch error: %s", e)
 

    df = fetch_prices(ticker, period="1y")
    if df is None:
        raise HTTPException(status_code=404, detail="Ticker data not found")

    indicators = compute_technical_indicators(df)
    fib = compute_fibonacci_levels(df, lookback=90)
    fund = fetch_fundamentals(ticker)
    fair = estimate_fair_value(fund)
    sentiment = fetch_news_sentiment(ticker, max_items=8)

    # build feature vector for RF from last row where features exist
    try:
        last_row = df.copy().dropna().iloc[-1]
        # compute same features as FEATURE_COLS
        last_feat = []
        for f in FEATURE_COLS:
            if f in df.columns:
                val = df[f].iloc[-1]
            else:
                # compute if needed
                val = indicators.get(f) if indicators.get(f) is not None else None
            last_feat.append(val if val is not None else 0.0)
        X = np.array(last_feat, dtype=float).reshape(1, -1)
        proba = float(model.predict_proba(X)[0,1])
    except Exception as e:
        logger.exception("rf predict failed")
        proba = None

    decision = "Strong Buy" if (proba is not None and proba >= 0.85) else ("Buy" if (proba is not None and proba >= 0.65) else "No Action")

    # dl predict (if available)
    dl_pred = None
    if TF_AVAILABLE:
        dl_model = load_lstm_model()
        if dl_model is not None:
            try:
                # prepare a small sequence using ma_5 (same heuristic as training)
                seq_len = dl_model.input_shape[1] if hasattr(dl_model.input_shape, "__len__") else None
                seq_len = int(seq_len) if seq_len else 20
                # build sequence from recent ma_5 values (fallback to adj_close)
                series = df["Adj_Close"].rolling(5).mean().dropna().values
                if len(series) >= seq_len:
                    seq = series[-seq_len:]
                else:
                    seq = np.pad(series, (seq_len-len(series),0), 'edge')  # pad with edge values
                Xdl = (np.array(seq).reshape(1, seq_len, 1) - np.mean(series)) / (np.std(series)+1e-9)
                ydl = dl_model.predict(Xdl, verbose=0)
                dl_pred = float(ydl.flatten()[0])
            except Exception as e:
                logger.warning("dl prediction failed: %s", e)
                dl_pred = None

    resp = {
        "ticker": ticker,
        "date": str(df.index[-1].date()),
        "probability": proba,
        "decision": decision,
        "dl_prediction": dl_pred,
        "indicators": indicators,
        "fibonacci": fib,
        "fundamentals": fund,
        "fair_value": fair,
        "sentiment": sentiment,
        "finbert_sentiment": finbert_sent
    }
    print(resp)
    return resp

@app.get("/top-picks")
def top_picks(n: int = 20, tickers: List[str] = None):
    model_obj = load_rf_model()
    if model_obj is None:
        raise HTTPException(status_code=400, detail="Model not trained. Call /train first.")
    if tickers is None:
        tickers = WATCHLIST
    model = model_obj["model"]
    feature_cols = model_obj["features"]
    results = []
    for t in tickers:
        df = fetch_prices(t, period="6mo")
        if df is None:
            continue
        try:
            incr = compute_technical_indicators(df)
            # build X
            last_feat = [incr.get(f, 0.0) for f in feature_cols]
            X = np.array(last_feat, dtype=float).reshape(1,-1)
            proba = float(model.predict_proba(X)[0,1])
            decision = "Strong Buy" if proba >= 0.85 else ("Buy" if proba >= 0.65 else "No Action")
            results.append({"ticker": t, "date": str(df.index[-1].date()), "prob": proba, "decision": decision})
        except Exception:
            continue
    results = sorted(results, key=lambda r: r["prob"], reverse=True)[:n]
    return {"results": results}



@app.get("/health")
def health():
    return {"status":"ok", "model_exists": os.path.exists(MODEL_PATH), "dl_installed": TF_AVAILABLE}

