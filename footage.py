import cv2
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load model components
clf, scaler, pca = joblib.load('fruit_classifier.pkl')

# --------- Spectrum Feature Extraction Functions --------- #

def pixel_to_wavelength(pixel_index, pixel_range=150, wavelength_range=(400, 800)):
    start_wl, end_wl = wavelength_range
    return start_wl + (pixel_index / pixel_range) * (end_wl - start_wl)

def get_band_indices(wavelength_range, pixel_range=150, sensor_range=(400, 800)):
    indices = []
    for wl in wavelength_range:
        index = int(((wl - sensor_range[0]) / (sensor_range[1] - sensor_range[0])) * pixel_range)
        indices.append(index)
    return indices

# Define bands (for simulated 150-pixel spectrum)
chlorophyll_band = get_band_indices([440, 680])
carotenoid_band = get_band_indices([470])
anthocyanin_band = get_band_indices([530])
ta_band = get_band_indices([720])
starch_band = get_band_indices([950])  # outside range

def extract_feature_intensities(spectrum):
    features = {
        "Chlorophyll": np.mean(spectrum[chlorophyll_band[0]:chlorophyll_band[1]]),
        "Carotenoids": spectrum[carotenoid_band[0]],
        "Anthocyanins": spectrum[anthocyanin_band[0]],
        "Titratable Acidity (Est.)": spectrum[ta_band[0]] if ta_band[0] < len(spectrum) else -1,
        "Starch (Est.)": spectrum[starch_band[0]] if starch_band[0] < len(spectrum) else -1
    }
    return features

# --------- Webcam + Prediction Loop --------- #

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Webcam not accessible.")
    exit()

print("[INFO] Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for speed and normalize
    resized = cv2.resize(frame, (100, 100)).astype(np.float32) / 255.0
    h, w, c = resized.shape

    # Simulate hyperspectral cube (150 bands, red channel repeated)
    cube = np.repeat(resized[:, :, :1], 150, axis=2)

    reshaped = cube.reshape(-1, 150)
    normalized = scaler.transform(reshaped)
    transformed = pca.transform(normalized)
    predicted = clf.predict(transformed)
    prediction_image = predicted.reshape(h, w).astype(np.uint8) * 255

    # Feature estimation from average spectrum of center pixels
    spectrum = np.mean(cube[40:60, 40:60, :], axis=(0, 1))  # Average small ROI
    features = extract_feature_intensities(spectrum)

    print("\n[INFO] Estimated Fruit Quality Features:")
    for k, v in features.items():
        print(f"{k}: {v:.2f}")

    # Show result
    cv2.imshow("Original", frame)
    cv2.imshow("Predicted Ripeness", prediction_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
