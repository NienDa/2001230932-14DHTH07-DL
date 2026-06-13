import os
import json
import numpy as np
from PIL import Image

import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from flask import Flask, render_template, request, url_for


# =========================
# 1. KHOI TAO FLASK
# =========================
app = Flask(__name__)


# =========================
# 2. CAU HINH
# =========================
MODEL_PATH = "efficientnetb0_102flowers_final.keras"
CLASS_NAME_PATH = "flower_display_names_efficientnet.json"

UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

IMG_SIZE = (224, 224)

model = None
class_names = []


# =========================
# 3. LOAD MODEL VA CLASS
# =========================
def load_model_and_classes():
    global model, class_names

    if not os.path.exists(MODEL_PATH):
        print("Chua tim thay model:", MODEL_PATH)
        print("Hay train xong truoc khi chay giao dien.")
        return

    model = tf.keras.models.load_model(MODEL_PATH)
    print("Da load model thanh cong:", MODEL_PATH)

    if os.path.exists(CLASS_NAME_PATH):
        with open(CLASS_NAME_PATH, "r", encoding="utf-8") as f:
            class_names = json.load(f)
    else:
        class_names = [str(i) for i in range(102)]

    print("So class:", len(class_names))


load_model_and_classes()


# =========================
# 4. TIEN XU LY ANH
# =========================
def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize(IMG_SIZE)

    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Quan trong voi EfficientNetB0
    img_array = preprocess_input(img_array)

    return img_array


# =========================
# 5. ROUTE TRANG CHU
# =========================
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    confidence = None
    image_url = None
    top_predictions = []

    if request.method == "POST":
        if model is None:
            result = "Chưa tìm thấy model. Hãy train xong trước."
            return render_template(
                "index.html",
                result=result,
                confidence=confidence,
                image_url=image_url,
                top_predictions=top_predictions
            )

        file = request.files.get("image")

        if file and file.filename != "":
            filename = file.filename
            image_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(image_path)

            image_url = url_for("static", filename=f"uploads/{filename}")

            img_array = preprocess_image(image_path)

            predictions = model.predict(img_array, verbose=0)[0]

            predicted_index = int(np.argmax(predictions))
            confidence = float(predictions[predicted_index]) * 100

            result = class_names[predicted_index]

            # Lay top 5 ket qua cao nhat
            top_indices = predictions.argsort()[-5:][::-1]

            for idx in top_indices:
                top_predictions.append({
                    "name": class_names[int(idx)],
                    "prob": round(float(predictions[int(idx)]) * 100, 2)
                })

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        image_url=image_url,
        top_predictions=top_predictions
    )


# =========================
# 6. CHAY APP
# =========================
if __name__ == "__main__":
    app.run(debug=True)