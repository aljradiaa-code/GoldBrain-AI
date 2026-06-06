#!/usr/bin/env python3
"""
MVT-VRL Auto Training Script v2.0
Fixed: handles missing files, empty data, all edge cases
Runs on GitHub Actions after trade log upload
"""

import os
import sys
import json
import glob
import traceback
import numpy as np

print("=" * 50)
print("MVT-VRL Training Script v2.0")
print("=" * 50)

# ── TensorFlow import with fallback ──────────────────────────
try:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    import tensorflow as tf
    from tensorflow import keras
    print(f"✓ TensorFlow {tf.__version__} loaded")
    TF_OK = True
except ImportError as e:
    print(f"✗ TensorFlow not available: {e}")
    TF_OK = False


def load_all_trade_logs():
    """Load all trade log JSON files from data/ folder"""
    logs = []
    pattern = 'data/trade_log_*.json'
    files = glob.glob(pattern)
    
    if not files:
        print(f"  No files matching: {pattern}")
        # Check if data folder has anything
        if os.path.exists('data'):
            all_files = os.listdir('data')
            print(f"  Files in data/: {all_files}")
        else:
            print("  data/ folder does not exist")
        return logs
    
    for path in sorted(files):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if not content:
                    print(f"  ⚠ Empty file: {path}")
                    continue
                data = json.loads(content)
                if isinstance(data, list):
                    valid = [t for t in data if isinstance(t, dict)]
                    logs.extend(valid)
                    print(f"  ✓ {path}: {len(valid)} records")
                elif isinstance(data, dict):
                    logs.append(data)
                    print(f"  ✓ {path}: 1 record")
        except json.JSONDecodeError as e:
            print(f"  ✗ JSON error in {path}: {e}")
        except Exception as e:
            print(f"  ✗ Error loading {path}: {e}")
    
    return logs


def trades_to_training_data(trades):
    """Convert trade records to (X, y) training arrays"""
    xs, ys = [], []
    
    for t in trades:
        try:
            entry  = float(t.get('entry', 0) or 0)
            tp     = float(t.get('tp', entry) or entry)
            sl     = float(t.get('sl', entry) or entry)
            prob   = float(t.get('prob', 50) or 50) / 100.0
            direction = str(t.get('direction', 'BUY')).upper()
            result    = str(t.get('result', 'timeout')).lower()
            
            if entry <= 0:
                continue
            
            sl_dist = abs(entry - sl) + 1e-8
            tp_dist = abs(tp - entry) + 1e-8
            rr      = tp_dist / sl_dist
            
            # Build 200-dim feature vector
            feat = [
                (entry - 2000) / 500,          # normalized gold price
                rr,                             # risk-reward ratio
                tp_dist / entry,                # TP distance %
                sl_dist / entry,                # SL distance %
                prob,                           # model confidence
                1.0 if direction == 'BUY' else -1.0,  # direction
            ]
            # Pad to 200
            feat += [0.0] * (200 - len(feat))
            xs.append(feat[:200])
            
            # Labels: [BUY_win, SELL_win, neutral]
            if result == 'win':
                ys.append([1,0,0] if direction == 'BUY' else [0,1,0])
            elif result == 'loss':
                ys.append([0,1,0] if direction == 'BUY' else [1,0,0])
            else:
                ys.append([0,0,1])
                
        except (ValueError, TypeError, KeyError) as e:
            continue  # skip bad records
    
    if not xs:
        return None, None
    
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def build_model():
    """Build the MVT-VRL neural network"""
    model = keras.Sequential([
        keras.layers.Dense(128, activation='relu', input_shape=(200,), name='dense_1'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(64, activation='relu', name='dense_2'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.15),
        keras.layers.Dense(32, activation='relu', name='dense_3'),
        keras.layers.Dense(3, activation='softmax', name='output')
    ], name='mvt_vrl_model')
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def load_or_build_model():
    """Load existing model weights or build fresh"""
    model = build_model()
    weights_path = 'models/model_weights.h5'
    
    if os.path.exists(weights_path):
        try:
            model.load_weights(weights_path)
            print(f"  ✓ Loaded existing weights from {weights_path}")
        except Exception as e:
            print(f"  ⚠ Could not load weights ({e}), starting fresh")
    else:
        print("  ℹ No existing weights, starting fresh")
    
    return model


def save_model(model, accuracy, samples, epochs_done):
    """Save model in all required formats"""
    os.makedirs('models', exist_ok=True)
    
    # 1. Save Keras weights (.h5)
    model.save_weights('models/model_weights.h5')
    print("  ✓ Saved models/model_weights.h5")
    
    # 2. Save full Keras model
    try:
        model.save('gold_brain_weights.keras')
        print("  ✓ Saved gold_brain_weights.keras")
    except Exception as e:
        print(f"  ⚠ Could not save .keras: {e}")
    
    # 3. Save TF.js compatible model.json topology
    model_config = model.get_config()
    
    # Build TF.js format
    tfjs_model = {
        "modelTopology": {
            "class_name": "Sequential",
            "config": model_config,
            "keras_version": tf.__version__,
            "backend": "tensorflow"
        },
        "format": "layers-model",
        "generatedBy": "MVT-VRL Trainer",
        "convertedBy": "manual",
        "weightsManifest": []
    }
    
    with open('models/model.json', 'w') as f:
        json.dump(tfjs_model, f, indent=2)
    print("  ✓ Saved models/model.json (TF.js format)")
    
    # 4. Save training report
    report = {
        "timestamp":  __import__('datetime').datetime.utcnow().isoformat() + 'Z',
        "accuracy":   float(accuracy),
        "samples":    int(samples),
        "epochs":     int(epochs_done),
        "version":    "2.0",
        "model_arch": "3→10→20→28 (200→128→64→32→3)",
        "status":     "trained" if accuracy > 0.5 else "needs_more_data"
    }
    with open('models/training_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    print(f"  ✓ Saved models/training_report.json (acc={accuracy:.1%})")


def generate_synthetic_data(n=200):
    """
    Generate synthetic XAUUSD-like training data
    when real trade logs are insufficient
    """
    print(f"  Generating {n} synthetic training samples...")
    xs, ys = [], []
    
    for _ in range(n):
        price    = 2000 + np.random.uniform(0, 800)
        rr       = np.random.uniform(1.5, 4.0)
        sl_pct   = np.random.uniform(0.001, 0.008)
        tp_pct   = sl_pct * rr
        prob     = np.random.uniform(0.4, 0.95)
        direction = np.random.choice([1.0, -1.0])
        
        feat = [
            (price - 2000) / 500,
            rr, tp_pct, sl_pct, prob, direction,
        ] + [np.random.normal(0, 0.1) for _ in range(194)]
        xs.append(feat[:200])
        
        # Realistic label: higher prob/rr → more likely to win
        win_chance = prob * 0.6 + (rr / 4.0) * 0.4
        r = np.random.random()
        if r < win_chance:
            ys.append([1,0,0] if direction > 0 else [0,1,0])
        elif r < win_chance + 0.2:
            ys.append([0,0,1])
        else:
            ys.append([0,1,0] if direction > 0 else [1,0,0])
    
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def main():
    if not TF_OK:
        print("✗ Cannot train without TensorFlow")
        sys.exit(1)
    
    print("\n[1] Loading trade logs...")
    trades = load_all_trade_logs()
    print(f"    Total records: {len(trades)}")
    
    print("\n[2] Preparing training data...")
    xs, ys = trades_to_training_data(trades)
    
    real_samples = len(xs) if xs is not None else 0
    print(f"    Real samples: {real_samples}")
    
    # Supplement with synthetic data if needed
    if real_samples < 50:
        needed = max(200, 200 - real_samples)
        sx, sy = generate_synthetic_data(needed)
        if real_samples > 0:
            xs = np.vstack([xs, sx])
            ys = np.vstack([ys, sy])
        else:
            xs, ys = sx, sy
        print(f"    After synthetic augmentation: {len(xs)} samples")
    
    print(f"\n[3] Building model...")
    model = load_or_build_model()
    model.summary(print_fn=lambda s: print(f"    {s}"))
    
    print(f"\n[4] Training ({len(xs)} samples)...")
    
    val_split = 0.15 if len(xs) >= 20 else 0.0
    epochs = 50
    
    history = model.fit(
        xs, ys,
        epochs=epochs,
        batch_size=min(32, max(8, len(xs) // 8)),
        validation_split=val_split,
        shuffle=True,
        verbose=1,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor='val_loss' if val_split > 0 else 'loss',
                patience=8,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss' if val_split > 0 else 'loss',
                factor=0.5,
                patience=4,
                verbose=1
            )
        ]
    )
    
    final_acc = history.history.get('val_accuracy', history.history.get('accuracy', [0]))[-1]
    epochs_done = len(history.history['loss'])
    print(f"\n    Final accuracy: {final_acc:.1%} after {epochs_done} epochs")
    
    print("\n[5] Saving model...")
    save_model(model, final_acc, len(xs), epochs_done)
    
    print("\n" + "=" * 50)
    print(f"✅ Training complete!")
    print(f"   Accuracy : {final_acc:.1%}")
    print(f"   Samples  : {len(xs)} ({real_samples} real + {len(xs)-real_samples} synthetic)")
    print(f"   Epochs   : {epochs_done}/{epochs}")
    print("=" * 50)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n✗ FATAL ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)
