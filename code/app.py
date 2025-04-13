import numpy as np
import tensorflow as tf
import rasterio
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from io import BytesIO
from tensorflow.keras.models import load_model # type: ignore

app = FastAPI()
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 👈 You can specify exact frontend origin here
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model
model = load_model('landslide_model.h5')

def preprocess_tif(file: UploadFile):
    with rasterio.open(file.file) as src:
        data = src.read(1)  # Read the first band
        transform = src.transform
        crs = src.crs
        profile = src.profile

        # Reduce the image to a single feature (mean pixel value)
        mean_value = np.nanmean(data)  # Use nanmean in case of NaNs

        return mean_value, transform, crs, profile

@app.get("/")
def read_root():
    return {"message": "Welcome to the Landslide Prediction API!"}

@app.post("/predict/")
async def predict(files: list[UploadFile] = File(...)):
    input_data = []
    transform, crs, profile = None, None, None

    for file in files:
        value, transform, crs, profile = preprocess_tif(file)
        input_data.append(value)

    X_input = np.array(input_data, dtype=np.float32).reshape(1, -1)
    pred_probs = model.predict(X_input)
    predicted_class = np.argmax(pred_probs, axis=1)[0]

    # Create a dummy 2D susceptibility map with the predicted class
    height, width = 100, 100
    predicted_map = np.full((height, width), predicted_class, dtype=np.uint8)

    # Optional: create a color RGB map for visualization
    colormap = np.array([
        [0, 255, 0],     # Green → Low
        [255, 255, 0],   # Yellow → Medium
        [255, 0, 0],     # Red → High
    ], dtype=np.uint8)
    colored_map = colormap[predicted_map]  # shape (100, 100, 3)

    # Write colored map to a TIFF in memory
    output_buffer = BytesIO()
    with rasterio.open(
        output_buffer,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=3,
        dtype='uint8',
        crs=crs,
        transform=transform
    ) as dst:
        for i in range(3):
            dst.write(colored_map[:, :, i], i + 1)

    output_buffer.seek(0)
    return StreamingResponse(output_buffer, media_type="image/tiff")
