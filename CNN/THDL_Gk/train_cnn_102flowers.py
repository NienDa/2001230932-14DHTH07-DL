import os
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


# =========================================================
# TRAIN CNN THUAN - OXFORD 102 FLOWERS
# Phan loai 102 loai hoa bang CNN tu xay dung
#
# Khong dung:
# - MobileNetV2
# - EfficientNet
# - ResNet
# - Transfer Learning
# - Pretrained ImageNet
# =========================================================


# =========================
# 1. CAU HINH DATASET
# =========================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Neu thu muc flower_data nam cung cap voi file code
DATASET_DIR = os.path.join(CURRENT_DIR, "flower_data")

# Neu anh muon dung duong dan tuyet doi thi bo comment dong duoi va sua lai duong dan:
# DATASET_DIR = r"N:\Nam3\HK2\THDeepLearn\LamLai\DoAn_102Flowers_CNN\flower_data"

TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VALID_DIR = os.path.join(DATASET_DIR, "valid")

# Do thu muc test bi rong nen tam dung valid lam tap danh gia
TEST_DIR = os.path.join(DATASET_DIR, "valid")

CAT_TO_NAME_PATH = os.path.join(DATASET_DIR, "cat_to_name.json")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.0005

print("========================================")
print("DATASET_DIR:", DATASET_DIR)
print("TRAIN_DIR  :", TRAIN_DIR)
print("VALID_DIR  :", VALID_DIR)
print("TEST_DIR   :", TEST_DIR)
print("IMG_SIZE   :", IMG_SIZE)
print("BATCH_SIZE :", BATCH_SIZE)
print("EPOCHS     :", EPOCHS)
print("LR         :", LEARNING_RATE)
print("========================================")

if not os.path.exists(TRAIN_DIR):
    raise FileNotFoundError(f"Khong tim thay TRAIN_DIR: {TRAIN_DIR}")

if not os.path.exists(VALID_DIR):
    raise FileNotFoundError(f"Khong tim thay VALID_DIR: {VALID_DIR}")

if not os.path.exists(TEST_DIR):
    raise FileNotFoundError(f"Khong tim thay TEST_DIR: {TEST_DIR}")


# =========================
# 2. LOAD DATASET
# =========================
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="int",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42
)

valid_ds = tf.keras.utils.image_dataset_from_directory(
    VALID_DIR,
    labels="inferred",
    label_mode="int",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    labels="inferred",
    label_mode="int",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = train_ds.class_names
num_classes = len(class_names)

print("\nSo luong class:", num_classes)
print("Danh sach class folder:", class_names[:10], "...")

with open("class_names.json", "w", encoding="utf-8") as f:
    json.dump(class_names, f, ensure_ascii=False, indent=4)


# =========================
# 3. DOC TEN HOA TU cat_to_name.json
# =========================
display_names = class_names

if os.path.exists(CAT_TO_NAME_PATH):
    with open(CAT_TO_NAME_PATH, "r", encoding="utf-8") as f:
        cat_to_name = json.load(f)

    display_names = [cat_to_name.get(str(c), str(c)) for c in class_names]

    with open("flower_display_names.json", "w", encoding="utf-8") as f:
        json.dump(display_names, f, ensure_ascii=False, indent=4)

    print("\nVi du ten lop:")
    for i in range(min(10, len(class_names))):
        print(f"{i}: folder={class_names[i]} -> name={display_names[i]}")
else:
    print("\nKhong tim thay cat_to_name.json, se dung ten folder lam ten class.")


# =========================
# 4. TOI UU LOAD DATA
# =========================
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
valid_ds = valid_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)


# =========================
# 5. XAY DUNG MODEL CNN THUAN
# =========================
# CNN thuan gom:
# - Rescaling
# - Conv2D
# - MaxPooling2D
# - GlobalAveragePooling2D
# - Dense
# - Softmax

model = models.Sequential([
    layers.Input(shape=(224, 224, 3)),

    # Chuan hoa pixel tu [0,255] ve [0,1]
    layers.Rescaling(1.0 / 255.0),

    # Block 1
    layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),

    # Block 2
    layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),

    # Block 3
    layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),

    # Block 4
    layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),

    # Giam so tham so so voi Flatten
    layers.GlobalAveragePooling2D(),

    # Fully Connected
    layers.Dense(256, activation="relu"),

    # Neu thay cho phep Dropout thi co the mo dong nay:
    # layers.Dropout(0.3),

    # Output 102 class
    layers.Dense(num_classes, activation="softmax")
])


# =========================
# 6. COMPILE MODEL
# =========================
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# =========================
# 7. LUU CAU HINH TRAIN
# =========================
config = {
    "dataset": "Oxford 102 Flowers",
    "dataset_dir": DATASET_DIR,
    "model": "Pure CNN",
    "num_classes": num_classes,
    "image_size": IMG_SIZE,
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "learning_rate": LEARNING_RATE,
    "optimizer": "Adam",
    "loss": "Sparse Categorical Crossentropy",
    "note": "CNN thuan, khong dung pretrained model, khong Transfer Learning",
    "architecture": [
        "Input(224, 224, 3)",
        "Rescaling(1/255)",
        "Conv2D(32) + MaxPooling2D",
        "Conv2D(64) + MaxPooling2D",
        "Conv2D(128) + MaxPooling2D",
        "Conv2D(256) + MaxPooling2D",
        "GlobalAveragePooling2D",
        "Dense(256, ReLU)",
        f"Dense({num_classes}, Softmax)"
    ]
}

with open("training_config.json", "w", encoding="utf-8") as f:
    json.dump(config, f, ensure_ascii=False, indent=4)


# =========================
# 8. CALLBACKS
# =========================
callbacks = [
    ModelCheckpoint(
        "cnn_102flowers_best.keras",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    ),

    EarlyStopping(
        monitor="val_loss",
        patience=6,
        restore_best_weights=True,
        verbose=1
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
]


# =========================
# 9. TRAIN MODEL
# =========================
history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)


# =========================
# 10. LUU MODEL
# =========================
model.save("cnn_102flowers_final.keras")


# =========================
# 11. DANH GIA MODEL
# =========================
test_loss, test_acc = model.evaluate(test_ds)

print("\n========================================")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Loss    : {test_loss:.4f}")
print("========================================")

with open("test_result.txt", "w", encoding="utf-8") as f:
    f.write(f"Test Accuracy: {test_acc:.4f}\n")
    f.write(f"Test Loss: {test_loss:.4f}\n")
    f.write(f"Num Classes: {num_classes}\n")
    f.write(f"Image Size: {IMG_SIZE}\n")
    f.write(f"Batch Size: {BATCH_SIZE}\n")
    f.write(f"Epochs: {EPOCHS}\n")
    f.write(f"Learning Rate: {LEARNING_RATE}\n")
    f.write("Model: Pure CNN\n")


# =========================
# 12. VE BIEU DO ACCURACY
# =========================
plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.savefig("accuracy_chart.png", dpi=300, bbox_inches="tight")
plt.show()


# =========================
# 13. VE BIEU DO LOSS
# =========================
plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig("loss_chart.png", dpi=300, bbox_inches="tight")
plt.show()


# =========================
# 14. DU DOAN TAP TEST / VALID
# =========================
y_true = []
y_pred = []

for images, labels in test_ds:
    preds = model.predict(images, verbose=0)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

y_true = np.array(y_true)
y_pred = np.array(y_pred)


# =========================
# 15. CLASSIFICATION REPORT
# =========================
report = classification_report(
    y_true,
    y_pred,
    target_names=display_names,
    zero_division=0
)

print("\nClassification Report:")
print(report)

with open("classification_report.txt", "w", encoding="utf-8") as f:
    f.write(report)


# =========================
# 16. CONFUSION MATRIX
# =========================
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(24, 20))
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=display_names
)

disp.plot(
    cmap="Blues",
    xticks_rotation=90,
    values_format="d",
    colorbar=True
)

plt.title("Confusion Matrix - CNN - Oxford 102 Flowers")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.show()


print("\nDa train xong va luu cac file:")
print("- cnn_102flowers_best.keras")
print("- cnn_102flowers_final.keras")
print("- class_names.json")
print("- flower_display_names.json")
print("- training_config.json")
print("- test_result.txt")
print("- accuracy_chart.png")
print("- loss_chart.png")
print("- classification_report.txt")
print("- confusion_matrix.png")