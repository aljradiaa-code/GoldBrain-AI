#!/usr/bin/env python3
import json, os, glob, re
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflowjs as tfjs
from datetime import datetime

def normalize_features(features_200):
    f = np.array(features_200, dtype=np.float32)
    mean = np.mean(f, axis=0)
    std = np.std(f, axis=0)
    std[std == 0] = 1
    return (f - mean) / std

def load_candles_from_csv(filepath):
    df = pd.read_csv(filepath)
    required = ['timestamp','open','high','low','close']
    if not all(c in df.columns for c in required):
        print(f"⚠️ ملف {filepath} لا يحتوي على الأعمدة المطلوبة، تخطي")
        return []
    candles = []
    for _, row in df.iterrows():
        candles.append({
            'timestamp': int(row['timestamp']),
            'open': float(row['open']),
            'high': float(row['high']),
            'low': float(row['low']),
            'close': float(row['close'])
        })
    return candles

def extract_features_50_candles(candles_50):
    if len(candles_50) != 50:
        return None
    features = []
    for c in candles_50:
        features.extend([c['open'], c['high'], c['low'], c['close']])
    return np.array(features, dtype=np.float32)

def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(200,)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    print("=== GoldBrain Auto-Trainer (OHLC-based) ===")
    os.makedirs("models", exist_ok=True)

    all_candles = []
    trades_list = []

    for f in glob.glob("data/candles_*.csv"):
        candles = load_candles_from_csv(f)
        if candles:
            all_candles.extend(candles)
            print(f"تم تحميل {len(candles)} شمعة من {f}")

    for f in glob.glob("data/training_batch_*.json"):
        try:
            with open(f, 'r') as fp:
                batch = json.load(fp)
                if 'candles' in batch:
                    for c in batch['candles']:
                        all_candles.append(c)
                if 'trades' in batch:
                    for t in batch['trades']:
                        trades_list.append(t)
                print(f"تم تحميل {len(batch.get('trades',[]))} صفقة و {len(batch.get('candles',[]))} شمعة من {f}")
        except Exception as e:
            print(f"خطأ في {f}: {e}")

    for f in glob.glob("data/trades_*.csv"):
        df = pd.read_csv(f)
        for _, row in df.iterrows():
            trade = {
                'direction': row.get('direction', ''),
                'result': row.get('result', 'closed'),
                'pnl': float(row.get('pnl', 0)),
                'tf': row.get('tf', 'M5'),
                'timestamp': row.get('openTime', None)
            }
            trades_list.append(trade)
        print(f"تم تحميل {len(df)} صفقة من {f}")

    if not all_candles or len(all_candles) < 50:
        print("❌ بيانات شموع غير كافية (يجب أن لا تقل عن 50 شمعة). سيتم تصدير نموذج افتراضي.")
        model = build_model()
        tfjs.converters.save_keras_model(model, "models/")
        print("تم تصدير نموذج افتراضي إلى models/model.json")
        return

    all_candles.sort(key=lambda x: x['timestamp'])
    print(f"إجمالي الشموع المجمعة: {len(all_candles)}")

    X = []
    Y = []
    window_size = 50
    future_bars = 4
    atr_period = 14

    for i in range(len(all_candles) - window_size - future_bars):
        window = all_candles[i:i+window_size]
        feat = extract_features_50_candles(window)
        if feat is None: continue
        atr = np.mean([c['high'] - c['low'] for c in window[-atr_period:]]) if atr_period <= len(window) else 5.0
        current_close = window[-1]['close']
        future_close = all_candles[i+window_size+future_bars-1]['close']
        future_change = future_close - current_close
        if future_change > atr * 0.5:
            label = [1,0,0]
        elif future_change < -atr * 0.5:
            label = [0,1,0]
        else:
            label = [0,0,1]
        X.append(feat)
        Y.append(label)

    print(f"تم إنشاء {len(X)} عينة تدريبية من الشموع.")

    if len(X) < 50:
        print("⚠️ عدد العينات قليل جداً (<50) لن يتم التدريب، سيتم تصدير النموذج الحالي إن وجد.")
        if os.path.exists("gold_brain_weights.keras"):
            model = tf.keras.models.load_model("gold_brain_weights.keras")
        else:
            model = build_model()
        tfjs.converters.save_keras_model(model, "models/")
        return

    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)
    X_norm = np.array([normalize_features(x) for x in X])

    if os.path.exists("gold_brain_weights.keras"):
        model = tf.keras.models.load_model("gold_brain_weights.keras")
        print("تم تحميل النموذج الموجود")
    else:
        model = build_model()
        print("تم بناء نموذج جديد")

    print(f"بدء التدريب على {X_norm.shape[0]} عينة...")
    history = model.fit(X_norm, Y, epochs=30, batch_size=min(32, len(X_norm)),
                        validation_split=0.2, verbose=1)
    acc = history.history['accuracy'][-1]
    print(f"✅ دقة التدريب النهائية: {acc:.3f}")

    model.save("gold_brain_weights.keras")
    tfjs.converters.save_keras_model(model, "models/")
    with open("models/version.json", "w") as f:
        json.dump({"updated": datetime.utcnow().isoformat()+"Z", "samples": len(X), "accuracy": float(acc)}, f, indent=2)
    print("✅ تم تصدير models/model.json و model_weights.bin")

if __name__ == "__main__":
    main()
