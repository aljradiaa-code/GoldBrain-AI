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

with open(LOGS_PATH) as f:
    raw = json.load(f)

# Support both {trades:[]} and plain [] formats
trades = raw.get('trades', raw) if isinstance(raw, dict) else raw
print(f'Loaded {len(trades)} trades')

if len(trades) < MIN_TRADES:
    print(f'Not enough trades ({len(trades)} < {MIN_TRADES}). Skipping retrain.')
    sys.exit(0)

# Filter only closed trades with outcome
closed = [t for t in trades if t.get('outcome') in ('WIN','LOSS','BREAKEVEN')]
print(f'Closed trades with outcome: {len(closed)}')

if len(closed) < MIN_TRADES:
    print(f'Not enough closed trades. Skipping retrain.')
    sys.exit(0)

SIG_MAP = {'BUY': 1.0, 'SELL': -1.0, 'WAIT': 0.0}
OUT_MAP = {'WIN': 1.0, 'BREAKEVEN': 0.5, 'LOSS': 0.0}

def extract(t):
    price      = float(t.get('entry_price') or t.get('entryPrice') or 0) / 5000.0
    change     = float(t.get('change_percent') or t.get('changePercent') or 0) / 10.0
    signal     = SIG_MAP.get(t.get('signal') or '', 0.0)
    confidence = float(t.get('confidence') or 0.5)
    outcome    = OUT_MAP.get(t.get('outcome',''), 0.0)
    return [price, change, signal, confidence, outcome]

rows = [extract(t) for t in closed]
X    = np.array([[r[0], r[1], r[2], r[3]] for r in rows], dtype=np.float32)
y    = np.array([r[4] for r in rows], dtype=np.float32)
print(f'Training on {len(X)} samples (X shape: {X.shape})')

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(4,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1,  activation='sigmoid'),
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=50, batch_size=max(4, len(X)//4), verbose=0)

loss, acc = model.evaluate(X, y, verbose=0)
print(f'Accuracy: {acc:.3f}  Loss: {loss:.4f}')

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
with open(MODEL_OUT, 'wb') as f:
    f.write(tflite_model)

print(f'Saved {len(tflite_model)} bytes -> {MODEL_OUT}')
