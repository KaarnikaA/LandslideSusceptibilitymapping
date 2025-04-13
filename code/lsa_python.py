
#!pip install numpy rasterio scikit-image scikit-learn tensorflow matplotlib joblib

import numpy as np
import rasterio
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import joblib

# Define file paths
files = ['aspect.tif', 'curvature.tif', 'distanceFromRoads.tif',
         'distanceFromStreams.tif', 'elevation.tif', 'lulc.tif',
         'ndvi.tif', 'planCurvature.tif', 'precipitation.tif',
         'profileCurvature.tif', 'relativeRelief.tif', 'slope.tif']

# Load target (landslide inventory)
with rasterio.open('landslideInventory.tif') as src:
    target = src.read(1).astype(float)
    profile = src.profile
    ref_shape = target.shape

# Load feature rasters
X = []
for f in files:
    with rasterio.open(f) as src:
        data = src.read(1).astype(float)
        if data.shape != ref_shape:
            from skimage.transform import resize
            data = resize(data, ref_shape, mode='constant', preserve_range=True)
        # Normalize to [0, 1]
        data = (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data))
        X.append(data.flatten())

X = np.stack(X, axis=1)
Y = target.flatten()

# Remove rows with NaNs
valid_idx = ~np.any(np.isnan(X), axis=1) & ~np.isnan(Y)
X = X[valid_idx]
Y = Y[valid_idx]

# Split into train, val, test
X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.3, random_state=42)
X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# Convert to numeric labels
encoder = LabelEncoder()
Y_train_enc = encoder.fit_transform(Y_train)
Y_val_enc = encoder.transform(Y_val)
Y_test_enc = encoder.transform(Y_test)

from tensorflow.keras.utils import to_categorical

# Even though class 2 is missing, we manually set num_classes=4
Y_train_cat = to_categorical(Y_train_enc, num_classes=4)
Y_val_cat   = to_categorical(Y_val_enc, num_classes=4)
Y_test_cat  = to_categorical(Y_test_enc, num_classes=4)

model = Sequential([
    Dense(64, input_dim=X_train.shape[1], activation='tanh'),
    Dense(32, activation='tanh'),
    Dense(16, activation='tanh'),
    Dense(4, activation='softmax')  # 4 classes
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train, Y_train_cat,
                    validation_data=(X_test, Y_test_cat),
                    epochs=15,
                    batch_size=64)

from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_score, recall_score, f1_score

# ... (rest of your code) ...

Y_pred_probs = model.predict(X_test)
Y_pred = np.argmax(Y_pred_probs, axis=1)  # Predict the class labels
Y_true = np.argmax(Y_test_cat, axis=1)    # True labels in class format

print(classification_report(Y_true, Y_pred))



conf_mat = confusion_matrix(Y_true, Y_pred)
print(conf_mat)



# Import ListedColormap
from matplotlib.colors import ListedColormap

# Generate full susceptibility map
all_preds = np.full(target.shape, np.nan)  # target.shape refers to the output shape (e.g., image size)
flat_probs = model.predict(X)
flat_preds = np.argmax(flat_probs, axis=1)  # Get predicted class for each sample
all_preds_flat = np.full(Y.shape, np.nan)   # Y.shape refers to the grid of predictions
all_preds_flat[valid_idx] = flat_preds     # valid_idx is the set of indices to insert predictions into
all_preds = all_preds_flat.reshape(ref_shape)  # ref_shape is the reference shape for your map

# Visualize the map (Optional: Color-coded map)
cmap = ListedColormap(['green', 'yellow', 'red'])  # Example color map: adjust according to the classes
plt.imshow(all_preds, cmap=cmap)
plt.title("Landslide Susceptibility Map")
plt.colorbar()
plt.show()

with rasterio.open('landslide_susceptibility_map.tif', 'w', **profile) as dst:
    dst.write(all_preds.astype(rasterio.float32), 1)

model.save('landslide_model.h5')