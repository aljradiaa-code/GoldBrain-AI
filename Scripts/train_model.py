#!/usr/bin/env python3
"""
MVT-VRL Auto Training Script v3.0
- Reads: data/trades_*.csv, data/training_batch_*.json, data/logs.json
- Exports: models/model.json + models/model_weights.bin (real TF.js format)
"""

import os, sys, json, glob, traceback, csv, datetime
import numpy as np

print("=" * 55)
print("MVT-VRL Training Script v3.0")
print("=" * 55)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
try:
    import tensorflow as tf
    from tensorflow import keras
    print(f"✓ TensorFlow {tf.__version__}")
    TF_OK = True
except ImportError as e:
    print(f"✗ TensorFlow: {e}"); TF_OK = False

try:
    import tensorflowjs as tfjs
    print(f"✓ TensorFlowJS {tfjs.__version__}")
    TFJS_OK = True
except ImportError as e:
    print(f"✗ TensorFlowJS: {e}"); TFJS_OK = False


# ── Load all data sources ───────────────────────────────────────
def load_all_trades():
    """Load from CSV trades + JSON training batches + logs.json"""
    trades = []

    # 1. CSV trades (from web app: data/trades_*.csv)
    csv_files = sorted(glob.glob("data/trades_*.csv"))
    print(f"  CSV files: {len(csv_files)}")
    for fp in csv_files:
        try:
            with open(fp, encoding='utf-8', errors='replace') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        entry = float(row.get('entry', 0) or 0)
                        tp    = float(row.get('tp', entry) or entry)
                        sl    = float(row.get('sl', entry) or entry)
                        pnl   = float(row.get('pnl', 0) or 0)
                        direction = str(row.get('direction', 'BUY')).upper().strip()
                        result    = str(row.get('result', 'closed')).lower().strip()
                        # Determine win/loss: use pnl if result="closed"
                        if result == 'closed':
                            result = 'win' if pnl > 0 else ('loss' if pnl < 0 else 'neutral')
                        trades.append({
                            'entry': entry, 'tp': tp, 'sl': sl,
                            'direction': direction, 'result': result,
                            'pnl': pnl, 'prob': 0.65
                        })
                    except (ValueError, TypeError): continue
        except Exception as e:
            print(f"  ✗ {fp}: {e}")
    print(f"  CSV records loaded: {len(trades)}")

    # 2. JSON training batches (data/training_batch_*.json)
    batch_files = sorted(glob.glob("data/training_batch_*.json"))
    print(f"  JSON batch files: {len(batch_files)}")
    batch_count = 0
    for fp in batch_files:
        try:
            with open(fp, encoding='utf-8') as f:
                data = json.load(f)
            records = data if isinstance(data, list) else data.get('trades', [])
            for r in records:
                if not isinstance(r, dict): continue
                trades.append({
                    'entry': float(r.get('entry', 0) or 0),
                    'tp':    float(r.get('tp', 0) or 0),
                    'sl':    float(r.get('sl', 0) or 0),
                    'direction': str(r.get('direction', 'BUY')).upper(),
                    'result':    str(r.get('result', 'closed')).lower(),
                    'pnl':       float(r.get('pnl', 0) or 0),
                    'prob':      float(r.get('prob', 0.65) or 0.65),
                })
                batch_count += 1
        except Exception as e:
            print(f"  ✗ {fp}: {e}")
    print(f"  JSON batch records: {batch_count}")

    # 3. logs.json (outcome-based)
    for lf in ['data/logs.json']:
        if os.path.exists(lf):
            try:
                with open(lf) as f: raw = json.load(f)
                records = raw if isinstance(raw, list) else raw.get('trades', [])
                SIG = {'BUY':1.,'SELL':-1.,'WAIT':0.}
                for r in records:
                    if not isinstance(r, dict): continue
                    outcome = str(r.get('outcome', '')).upper()
                    result  = 'win' if outcome=='WIN' else ('loss' if outcome=='LOSS' else 'neutral')
                    trades.append({
                        'entry':     float(r.get('entry_price', 0) or 0),
                        'tp':        float(r.get('entry_price', 0) or 0),
                        'sl':        float(r.get('entry_price', 0) or 0),
                        'direction': str(r.get('direction', 'BUY')).upper(),
                        'result':    result,
                        'pnl':       float(r.get('pnl', 0) or 0),
                        'prob':      float(r.get('confidence', 0.65) or 0.65),
                    })
                print(f"  logs.json records: {len(records)}")
            except Exception as e:
                print(f"  ✗ logs.json: {e}")

    return trades


def trades_to_xy(trades):
    """Convert trade list → (X[n,200], y[n,3])"""
    xs, ys = [], []
    for t in trades:
        try:
            entry = float(t.get('entry', 0) or 0)
            if entry <= 0: continue
            tp  = float(t.get('tp', entry) or entry)
            sl  = float(t.get('sl', entry) or entry)
            prob = float(t.get('prob', 0.65) or 0.65)
            direction = str(t.get('direction', 'BUY')).upper()
            result    = str(t.get('result', 'neutral')).lower()
            if result == 'closed':
                pnl = float(t.get('pnl', 0) or 0)
                result = 'win' if pnl > 0 else ('loss' if pnl < 0 else 'neutral')
            sl_d = abs(entry - sl) + 1e-8
            tp_d = abs(tp - entry) + 1e-8
            rr   = tp_d / sl_d
            feat = [
                (entry - 2000) / 500,
                rr,
                tp_d / entry,
                sl_d / entry,
                prob,
                1.0 if direction == 'BUY' else -1.0,
            ] + [0.0] * 194
            xs.append(feat[:200])
            if result == 'win':
                ys.append([1,0,0] if direction == 'BUY' else [0,1,0])
            elif result == 'loss':
                ys.append([0,1,0] if direction == 'BUY' else [1,0,0])
            else:
                ys.append([0,0,1])
        except (ValueError, TypeError): continue
    if not xs: return None, None
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def generate_synthetic(n=300):
    print(f"  Generating {n} synthetic XAUUSD samples...")
    xs, ys = [], []
    for _ in range(n):
        price = 2000 + np.random.uniform(0, 800)
        rr    = np.random.uniform(1.5, 4.0)
        sl_p  = np.random.uniform(0.001, 0.008)
        prob  = np.random.uniform(0.4, 0.95)
        d     = np.random.choice([1.0, -1.0])
        feat  = [(price-2000)/500, rr, sl_p*rr, sl_p, prob, d] +                 [np.random.normal(0, 0.08) for _ in range(194)]
        xs.append(feat[:200])
        wc = prob * 0.6 + (rr / 4.0) * 0.4
        r  = np.random.random()
        if r < wc:        ys.append([1,0,0] if d>0 else [0,1,0])
        elif r < wc+0.2:  ys.append([0,0,1])
        else:             ys.append([0,1,0] if d>0 else [1,0,0])
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def build_model():
    m = keras.Sequential([
        keras.layers.Dense(128, activation='relu', input_shape=(200,), name='dense_1'),
        keras.layers.BatchNormalization(name='bn_1'),
        keras.layers.Dropout(0.2, name='dropout_1'),
        keras.layers.Dense(64, activation='relu', name='dense_2'),
        keras.layers.BatchNormalization(name='bn_2'),
        keras.layers.Dropout(0.15, name='dropout_2'),
        keras.layers.Dense(32, activation='relu', name='dense_3'),
        keras.layers.Dense(3, activation='softmax', name='output'),
    ], name='mvt_vrl_model')
    m.compile(optimizer=keras.optimizers.Adam(0.001),
              loss='categorical_crossentropy', metrics=['accuracy'])
    return m


def save_all(model, acc, n_samples, n_epochs):
    os.makedirs('models', exist_ok=True)

    # 1. Keras full model
    try:
        model.save('gold_brain_weights.keras')
        print("  ✓ gold_brain_weights.keras")
    except Exception as e:
        print(f"  ⚠ keras save: {e}")

    # 2. TF.js format with real weights (requires tensorflowjs package)
    if TFJS_OK:
        try:
            tfjs.converters.save_keras_model(model, 'models/')
            print("  ✓ models/model.json + model_weights.bin (TF.js)")
        except Exception as e:
            print(f"  ⚠ tfjs save: {e}")
    else:
        # Fallback: save topology-only model.json
        cfg = model.get_config()
        tfjs_model = {
            "modelTopology": {
                "class_name": "Sequential",
                "config": cfg,
                "keras_version": tf.__version__,
                "backend": "tensorflow"
            },
            "format": "layers-model",
            "generatedBy": f"MVT-VRL Trainer v3.0",
            "convertedBy": "train_model.py",
            "weightsManifest": []
        }
        with open('models/model.json', 'w') as f:
            json.dump(tfjs_model, f, indent=2)
        print("  ✓ models/model.json (topology only — install tensorflowjs for weights)")

    # 3. Training report
    report = {
        "timestamp":  datetime.datetime.utcnow().isoformat() + 'Z',
        "accuracy":   round(float(acc), 4),
        "samples":    int(n_samples),
        "epochs":     int(n_epochs),
        "version":    "3.0",
        "model_arch": "200->128->BN->64->BN->32->3",
        "status":     "ready"
    }
    with open('models/training_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    print(f"  ✓ training_report.json  acc={acc:.1%}  samples={n_samples}")


def main():
    if not TF_OK:
        print("✗ No TensorFlow"); sys.exit(1)

    print("\n[1] Loading data...")
    trades = load_all_trades()
    print(f"    Total trade records: {len(trades)}")

    print("\n[2] Building feature matrix...")
    xs, ys = trades_to_xy(trades)
    real_n = len(xs) if xs is not None else 0
    print(f"    Real samples: {real_n}")

    if real_n < 50:
        sx, sy = generate_synthetic(max(300, 300 - real_n))
        xs = np.vstack([xs, sx]) if real_n > 0 else sx
        ys = np.vstack([ys, sy]) if real_n > 0 else sy
        print(f"    Total after augment: {len(xs)}")

    print("\n[3] Building model...")
    model = build_model()

    # Try loading existing keras weights
    for wpath in ['gold_brain_weights.keras', 'models/model_weights.h5']:
        if os.path.exists(wpath):
            try:
                if wpath.endswith('.keras'):
                    old_m = tf.keras.models.load_model(wpath)
                    model.set_weights(old_m.get_weights())
                else:
                    model.load_weights(wpath)
                print(f"  ✓ Loaded weights from {wpath}")
                break
            except Exception as e:
                print(f"  ⚠ Load {wpath}: {e}")

    model.summary(print_fn=lambda s: print(f"    {s}"))

    print(f"\n[4] Training on {len(xs)} samples...")
    val_s = 0.15 if len(xs) >= 30 else 0.0
    history = model.fit(
        xs, ys, epochs=60,
        batch_size=min(32, max(8, len(xs)//8)),
        validation_split=val_s, shuffle=True, verbose=1,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor='val_loss' if val_s>0 else 'loss',
                patience=10, restore_best_weights=True, verbose=1),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss' if val_s>0 else 'loss',
                factor=0.5, patience=5, verbose=1)
        ]
    )
    final_acc  = history.history.get('val_accuracy', history.history.get('accuracy',[0]))[-1]
    epochs_done = len(history.history['loss'])
    print(f"\n  Final accuracy: {final_acc:.1%}  ({epochs_done} epochs)")

    print("\n[5] Saving...")
    save_all(model, final_acc, len(xs), epochs_done)

    print("\n" + "="*55)
    print(f"✅ Done! acc={final_acc:.1%} samples={len(xs)} epochs={epochs_done}")
    print("="*55)


if __name__ == '__main__':
    try: main()
    except Exception as e:
        print(f"\n✗ FATAL: {e}"); traceback.print_exc(); sys.exit(1)
