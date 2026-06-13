import os
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


# =========================================================
# TRAIN MODEL NANG CAP - OXFORD 102 FLOWERS
# Model: EfficientNetB0 + Transfer Learning
#
# Ban nay KHONG fine-tuning de bieu do muot hon.
# EfficientNetB0 duoc dung nhu bo trich xuat dac trung.
# Chi train cac lop phan loai moi phia sau.
# =========================================================


# =========================
# 1. CAU HINH DATASET
# =========================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = r"N:\Nam3\HK2\THDeepLearn\THDL_CK\flower_data"

if not os.path.exists(DATASET_DIR):
    DATASET_DIR = os.path.join(CURRENT_DIR, "flower_data")

TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VALID_DIR = os.path.join(DATASET_DIR, "valid")

# Neu thu muc test bi rong, dung valid de danh gia
TEST_DIR = os.path.join(DATASET_DIR, "valid")

CAT_TO_NAME_PATH = os.path.join(DATASET_DIR, "cat_to_name.json")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 0.001

print("========================================")
print("DATASET_DIR:", DATASET_DIR)
print("TRAIN_DIR  :", TRAIN_DIR)
print("VALID_DIR  :", VALID_DIR)
print("TEST_DIR   :", TEST_DIR)
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

with open("class_names_efficientnet.json", "w", encoding="utf-8") as f:
    json.dump(class_names, f, ensure_ascii=False, indent=4)


# =========================
# 3. DOC TEN HOA TU cat_to_name.json
# =========================
display_names = class_names

if os.path.exists(CAT_TO_NAME_PATH):
    with open(CAT_TO_NAME_PATH, "r", encoding="utf-8") as f:
        cat_to_name = json.load(f)

    display_names = [cat_to_name.get(str(c), str(c)) for c in class_names]

    with open("flower_display_names_efficientnet.json", "w", encoding="utf-8") as f:
        json.dump(display_names, f, ensure_ascii=False, indent=4)

    print("\nVi du ten lop:")
    for i in range(min(10, len(class_names))):
        print(f"{i}: folder={class_names[i]} -> name={display_names[i]}")
else:
    print("\nKhong tim thay cat_to_name.json, se dung ten folder lam ten class.")


# Tang toc load data
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
valid_ds = valid_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)


# =========================
# 4. XAY DUNG MODEL EFFICIENTNETB0
# =========================

# Data Augmentation giup mo hinh tong quat tot hon
data_augmentation = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.15),
    layers.RandomContrast(0.1),
], name="data_augmentation")


# EfficientNetB0 pretrained tren ImageNet
# include_top=False: bo lop phan loai 1000 class goc
base_model = EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3)
)

# KHONG fine-tuning:
# Dong bang toan bo EfficientNetB0
base_model.trainable = False


inputs = layers.Input(shape=(224, 224, 3))

x = data_augmentation(inputs)

# Tien xu ly dung chuan EfficientNet
x = preprocess_input(x)

# Dung EfficientNetB0 lam bo trich xuat dac trung
x = base_model(x, training=False)

# Giam so tham so, thay cho Flatten
x = layers.GlobalAveragePooling2D()(x)

# Giam overfitting
x = layers.Dropout(0.3)(x)

# Lop phan loai moi
x = layers.Dense(256, activation="relu")(x)

x = layers.Dropout(0.2)(x)

# Output 102 class
outputs = layers.Dense(num_classes, activation="softmax")(x)

model = models.Model(inputs, outputs, name="EfficientNetB0_102Flowers_FeatureExtractor")


# =========================
# 5. COMPILE MODEL
# =========================
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# =========================
# 6. LUU CAU HINH TRAIN
# =========================
config = {
    "dataset": "Oxford 102 Flowers",
    "dataset_dir": DATASET_DIR,
    "model": "EfficientNetB0 + Transfer Learning",
    "training_type": "Feature Extraction only, no fine-tuning",
    "num_classes": num_classes,
    "image_size": IMG_SIZE,
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "learning_rate": LEARNING_RATE,
    "optimizer": "Adam",
    "loss": "Sparse Categorical Crossentropy",
    "note": "EfficientNetB0 duoc dong bang, chi train cac lop classifier moi de bieu do muot hon"
}

with open("training_config_efficientnet.json", "w", encoding="utf-8") as f:
    json.dump(config, f, ensure_ascii=False, indent=4)


# =========================
# 7. CALLBACKS
# =========================
callbacks = [
    ModelCheckpoint(
        "efficientnetb0_102flowers_best.keras",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    ),

    EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True,
        verbose=1
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )
]


# =========================
# 8. TRAIN MODEL
# =========================
history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)


# Luu model cuoi
model.save("efficientnetb0_102flowers_final.keras")


# =========================
# 9. DANH GIA MODEL
# =========================
test_loss, test_acc = model.evaluate(test_ds)

print("\n========================================")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Loss    : {test_loss:.4f}")
print("========================================")

with open("test_result_efficientnet.txt", "w", encoding="utf-8") as f:
    f.write(f"Test Accuracy: {test_acc:.4f}\n")
    f.write(f"Test Loss: {test_loss:.4f}\n")
    f.write(f"Num Classes: {num_classes}\n")
    f.write(f"Image Size: {IMG_SIZE}\n")
    f.write(f"Batch Size: {BATCH_SIZE}\n")
    f.write(f"Epochs: {EPOCHS}\n")
    f.write(f"Learning Rate: {LEARNING_RATE}\n")
    f.write("Model: EfficientNetB0 + Transfer Learning\n")
    f.write("Training Type: Feature Extraction only, no fine-tuning\n")


# =========================
# 10. VE BIEU DO ACCURACY
# =========================
plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Training and Validation Accuracy - EfficientNetB0")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.savefig("accuracy_chart_efficientnet.png", dpi=300, bbox_inches="tight")
plt.show()


# =========================
# 11. VE BIEU DO LOSS
# =========================
plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Training and Validation Loss - EfficientNetB0")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig("loss_chart_efficientnet.png", dpi=300, bbox_inches="tight")
plt.show()


# =========================
# 12. PREDICT TEST SET
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
# 13. CLASSIFICATION REPORT
# =========================
report = classification_report(
    y_true,
    y_pred,
    target_names=display_names,
    zero_division=0
)

print("\nClassification Report:")
print(report)

with open("classification_report_efficientnet.txt", "w", encoding="utf-8") as f:
    f.write(report)


# =========================
# 14. CONFUSION MATRIX
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

plt.title("Confusion Matrix - EfficientNetB0 - Oxford 102 Flowers")
plt.tight_layout()
plt.savefig("confusion_matrix_efficientnet.png", dpi=300, bbox_inches="tight")
plt.show()


print("\nDa train xong va luu cac file:")
print("- efficientnetb0_102flowers_best.keras")
print("- efficientnetb0_102flowers_final.keras")
print("- class_names_efficientnet.json")
print("- flower_display_names_efficientnet.json")
print("- training_config_efficientnet.json")
print("- test_result_efficientnet.txt")
print("- accuracy_chart_efficientnet.png")
print("- loss_chart_efficientnet.png")
print("- classification_report_efficientnet.txt")
print("- confusion_matrix_efficientnet.png")