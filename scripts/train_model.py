import json, os, sys
import numpy as np

try:
    import tensorflow as tf
except ImportError:
    print('Installing tensorflow...'); os.system('pip install tensorflow')
    import tensorflow as tf

LOGS_PATH  = os.environ.get('LOGS_PATH', 'data/logs.json')
MODEL_OUT  = os.environ.get('MODEL_OUT',  'models/gold_brain_model.tflite')
MIN_TRADES = int(os.environ.get('MIN_TRADES', '10'))

# ─── Load logs ────────────────────────────────────────────────────────────────
with open(LOGS_PATH) as f:
    data = json.load(f)

trades = data.get('trades', [])
print(f'Loaded {len(trades)} trades')

if len(trades) < MIN_TRADES:
    print(f'Not enough trades ({len(trades)} < {MIN_TRADES}). Skipping retrain.')
    sys.exit(0)

# ─── Feature engineering ──────────────────────────────────────────────────────
def extract_features(trade):
    price      = float(trade.get('price', 0))
    change     = float(trade.get('changePercent', 0) or 0)
    sig_map    = {'BUY': 1.0, 'SELL': -1.0, 'WAIT': 0.0}
    signal     = sig_map.get(trade.get('signal', 'WAIT'), 0.0)
    confidence = float(trade.get('confidence', 0.5))
    outcome    = 1.0 if trade.get('outcome', 'WIN') == 'WIN' else 0.0
    return [price / 5000.0, change / 10.0, signal, confidence, outcome]

rows = [extract_features(t) for t in trades]
X    = np.array([[r[0], r[1], r[2], r[3]] for r in rows], dtype=np.float32)
y    = np.array([r[4] for r in rows], dtype=np.float32)
print(f'Training on {len(X)} samples...')

# ─── Build & train model ──────────────────────────────────────────────────────
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(4,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1,  activation='sigmoid'),
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=50, batch_size=8, verbose=0)

loss, acc = model.evaluate(X, y, verbose=0)
print(f'Accuracy: {acc:.3f}  Loss: {loss:.4f}')

# ─── Export TFLite ────────────────────────────────────────────────────────────
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
with open(MODEL_OUT, 'wb') as f:
    f.write(tflite_model)

print(f'Saved {len(tflite_model)} bytes -> {MODEL_OUT}')
