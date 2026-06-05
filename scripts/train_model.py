#!/usr/bin/env python3
"""
GoldBrain-AI Training Script
Supports two modes:
  1. Candle-based OHLC (200 features = 50 candles x 4) --> exports TF.js model.json
  2. Trade-outcome (4 features) --> exports TFLite
"""
import json, os, sys, glob
import numpy as np

try:
    import tensorflow as tf
except ImportError:
    os.system("pip install tensorflow==2.13.0")
    import tensorflow as tf

try:
    import tensorflowjs as tfjs
except ImportError:
    os.system("pip install tensorflowjs==4.10.0")
    import tensorflowjs as tfjs

import pandas as pd

os.makedirs("models", exist_ok=True)

# ── Mode 1: Candle CSV trades (from web app) ───────────────────────────────
csv_files = sorted(glob.glob("data/trades_*.csv"))
if csv_files:
    print(f"Found {len(csv_files)} CSV trade files — candle-based mode")
    
    # Load or build 200-feature model
    model = None
    if os.path.exists("gold_brain_weights.keras"):
        try:
            model = tf.keras.models.load_model("gold_brain_weights.keras")
            print("Loaded existing model")
        except Exception as e:
            print(f"Load failed: {e}")
    
    if model is None:
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation="relu", input_shape=(200,)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(3, activation="softmax"),
        ])
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        print("Built new 200-feature model")

    # Build training samples from CSV
    xs, ys = [], []
    for fp in csv_files:
        try:
            df = pd.read_csv(fp)
            for _, row in df.iterrows():
                res = str(row.get("result","")).lower()
                if res not in ("win","loss"): continue
                feat = np.zeros(200, dtype=np.float32)
                feat[0] = 1.0 if str(row.get("direction","")).upper()=="BUY" else -1.0
                feat[1] = float(row.get("pnl", 0)) / 100.0
                feat[2] = {"M5":0.2,"M15":0.4,"H1":0.7,"H4":1.0}.get(str(row.get("tf","")),0.5)
                xs.append(feat)
                d = "BUY" if str(row.get("direction","")).upper()=="BUY" else "SELL"
                ys.append([1,0,0] if (res=="win" and d=="BUY") else [0,1,0] if res=="win" else [0,0,1])
        except Exception as e:
            print(f"Error reading {fp}: {e}")

    if len(xs) >= 10:
        print(f"Training on {len(xs)} samples...")
        model.fit(np.array(xs), np.array(ys, dtype=np.float32),
                  epochs=20, batch_size=min(32,len(xs)), verbose=1)
        model.save("gold_brain_weights.keras")
        print("Saved keras weights")
    else:
        print(f"Only {len(xs)} valid samples — skipping fit, exporting existing weights")

    # Export to TF.js
    tfjs.converters.save_keras_model(model, "models/")
    import datetime
    with open("models/version.json", "w") as f:
        json.dump({"updated": datetime.datetime.utcnow().isoformat()+"Z",
                   "samples": len(xs), "params": int(model.count_params())}, f, indent=2)
    print("Exported models/model.json (TF.js)")

# ── Mode 2: logs.json trade outcomes ─────────────────────────────────────────
elif os.path.exists("data/logs.json"):
    print("Found data/logs.json — outcome-based mode")
    with open("data/logs.json") as f:
        raw = json.load(f)
    trades = raw.get("trades", raw) if isinstance(raw, dict) else raw
    closed = [t for t in trades if t.get("outcome") in ("WIN","LOSS","BREAKEVEN")]
    print(f"Closed trades: {len(closed)}")
    if len(closed) < 10:
        print("Not enough trades. Skipping.")
        sys.exit(0)
    SIG = {"BUY":1.,"SELL":-1.,"WAIT":0.}
    OUT = {"WIN":1.,"BREAKEVEN":0.5,"LOSS":0.}
    X = np.array([[float(t.get("entry_price",0))/5000., float(t.get("change_percent",0))/10.,
                    SIG.get(t.get("signal",""),0.), float(t.get("confidence",0.5))]
                   for t in closed], dtype=np.float32)
    y = np.array([OUT.get(t.get("outcome",""),0.) for t in closed], dtype=np.float32)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(4,)),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(X, y, epochs=50, batch_size=max(4,len(X)//4), verbose=0)
    loss, acc = model.evaluate(X, y, verbose=0)
    print(f"Accuracy: {acc:.3f}")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite = converter.convert()
    with open("models/gold_brain_model.tflite","wb") as f: f.write(tflite)
    print(f"Saved TFLite model")

else:
    print("No data found. Building and exporting default TF.js model.")
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation="relu", input_shape=(200,)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(3, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    tfjs.converters.save_keras_model(model, "models/")
    print("Exported default models/model.json")

print("Done!")
