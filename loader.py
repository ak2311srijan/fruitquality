# predict_fruit.py

import numpy as np
import matplotlib.pyplot as plt
import joblib

# Load model, scaler, PCA
clf, scaler, pca = joblib.load('fruit_classifier.pkl')

# Load new hyperspectral cube
cube = np.load('fruit_sample.npy')  # Use your test cube
h, w, b = cube.shape

# Flatten and preprocess
reshaped = cube.reshape(-1, b)
normalized = scaler.transform(reshaped)
pca_transformed = pca.transform(normalized)

# Predict
predicted = clf.predict(pca_transformed)
predicted_image = predicted.reshape(h, w)

# Visualize
plt.imshow(predicted_image, cmap='viridis')
plt.title("Predicted Ripeness Map")
plt.colorbar(label="Class")
plt.axis("off")
plt.show()
