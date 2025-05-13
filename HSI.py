# hsi_fruit_classifier.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

# === CONFIG ===
HSI_DATA_PATH = 'fruit_sample.npy'      # Hyperspectral cube (HxWxBands)
LABELS_PATH = 'fruit_labels.npy'        # Labels for each pixel (0 = unripe, 1 = ripe)
MODEL_SAVE_PATH = 'fruit_classifier.pkl'

# === 1. Load Hyperspectral Cube ===
print("[INFO] Loading hyperspectral cube...")
hsi_cube = np.load(HSI_DATA_PATH)  # shape: (H, W, Bands)
h, w, b = hsi_cube.shape

# Optional: View RGB Approximation (select 3 bands)
rgb_image = hsi_cube[:, :, [60, 30, 10]]
plt.imshow(rgb_image / np.max(rgb_image))
plt.title("RGB Approximation of HSI Cube")
plt.axis("off")
plt.show()

# === 2. Preprocess Data ===
print("[INFO] Preprocessing data...")
reshaped_cube = hsi_cube.reshape(-1, b)  # (Pixels, Bands)

scaler = StandardScaler()
normalized_cube = scaler.fit_transform(reshaped_cube)

# === 3. Load Labels and Split ===
labels = np.load(LABELS_PATH)  # shape: (Pixels, )
X_train, X_test, y_train, y_test = train_test_split(normalized_cube, labels, test_size=0.2, random_state=42)

# === 4. PCA Dimensionality Reduction ===
print("[INFO] Applying PCA...")
pca = PCA(n_components=10)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# === 5. Train Classifier ===
print("[INFO] Training Random Forest classifier...")
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_pca, y_train)

# Save model for reuse
joblib.dump((clf, scaler, pca), MODEL_SAVE_PATH)
print(f"[INFO] Model saved to {MODEL_SAVE_PATH}")

# === 6. Evaluate ===
y_pred = clf.predict(X_test_pca)
print("[INFO] Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# === 7. Predict on Full Image ===
print("[INFO] Predicting full image...")
full_pca = pca.transform(normalized_cube)
predicted_full = clf.predict(full_pca)
predicted_image = predicted_full.reshape(h, w)

# === 8. Visualize Prediction ===
plt.imshow(predicted_image, cmap='viridis')
plt.title("Predicted Ripeness Map")
plt.colorbar(label="Class (0 = Unripe, 1 = Ripe)")
plt.axis("off")
plt.show()
