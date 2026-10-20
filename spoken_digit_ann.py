# =====================================================
# Spoken Digit Recognition using ANN (Free Spoken Digit Dataset)
# Pipeline: WAV audio -> MFCC features -> ANN (Dense layers) -> digit 0-9
# =====================================================
import os, subprocess, glob
import numpy as np
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

np.random.seed(42)
tf.random.set_seed(42)

# ---------- STEP 1: Download dataset (3000 wav files, ~27 MB) ----------
if not os.path.exists("free-spoken-digit-dataset"):
    subprocess.run(["git", "clone", "--depth", "1",
                    "https://github.com/Jakobovski/free-spoken-digit-dataset.git"], check=True)
DATA_DIR = "free-spoken-digit-dataset/recordings"

# ---------- STEP 2: Feature extraction (MFCC) ----------
N_MFCC = 20          # number of MFCC coefficients per frame

def extract_features(path):
    signal, sr = librosa.load(path, sr=8000)             # all files are 8 kHz
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=N_MFCC)   # shape (20, frames)
    # Every recording has a different length -> summarise over time
    # so each audio becomes a FIXED-size vector of 40 numbers.
    return np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])

files = sorted(glob.glob(os.path.join(DATA_DIR, "*.wav")))
X, y = [], []
for f in files:
    X.append(extract_features(f))
    y.append(int(os.path.basename(f).split("_")[0]))     # file name "7_jackson_3.wav" -> label 7
X, y = np.array(X), np.array(y)
print("Feature matrix:", X.shape, " Labels:", y.shape)

# ---------- STEP 3: Split + Normalise ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler().fit(X_train)                   # fit ONLY on train data
X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)

# ---------- STEP 4: Build the ANN ----------
model = keras.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(10, activation="softmax"),              # 10 digits
])
model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
model.summary()

# ---------- STEP 5: Train ----------
early = keras.callbacks.EarlyStopping(monitor="val_loss", patience=15,
                                      restore_best_weights=True)
history = model.fit(X_train, y_train, validation_split=0.15,
                    epochs=150, batch_size=32, callbacks=[early], verbose=2)

# ---------- STEP 6: Evaluate ----------
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTEST ACCURACY: {test_acc*100:.2f}%")
y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
print(classification_report(y_test, y_pred))

# ---------- STEP 7: Graphs for the presentation ----------
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="train")
plt.plot(history.history["val_accuracy"], label="validation")
plt.title("Accuracy"); plt.xlabel("Epoch"); plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="validation")
plt.title("Loss"); plt.xlabel("Epoch"); plt.legend()
plt.tight_layout(); plt.savefig("training_curves.png", dpi=150)

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
plt.imshow(cm, cmap="Blues"); plt.colorbar()
plt.xticks(range(10)); plt.yticks(range(10))
plt.xlabel("Predicted digit"); plt.ylabel("Actual digit"); plt.title("Confusion Matrix")
for i in range(10):
    for j in range(10):
        plt.text(j, i, cm[i, j], ha="center", va="center",
                 color="white" if cm[i, j] > cm.max()/2 else "black")
plt.tight_layout(); plt.savefig("confusion_matrix.png", dpi=150)

# ---------- STEP 8: Save model and test on ONE audio file ----------
model.save("digit_model.keras")

def predict_digit(wav_path):
    feats = scaler.transform(extract_features(wav_path).reshape(1, -1))
    probs = model.predict(feats, verbose=0)[0]
    return int(np.argmax(probs)), float(probs.max())

sample = files[1234]
digit, conf = predict_digit(sample)
print(f"\nDemo file: {os.path.basename(sample)} -> predicted {digit} (confidence {conf*100:.1f}%)")
