import os
import xml.etree.ElementTree as ET
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import cv2

# ==============================
# DATASET PATH
# ==============================

BASE_PATH = r"D:\NUST\Task2.3\dataset"

train_images = os.path.join(BASE_PATH, "train", "images")
train_ann = os.path.join(BASE_PATH, "train", "annotations")

val_images = os.path.join(BASE_PATH, "val", "images")
val_ann = os.path.join(BASE_PATH, "val", "annotations")

# ==============================
# CLASS LABELS
# ==============================

class_map = {
    "dog": 0,
    "car": 1,
    "person": 2
}

IMG_SIZE = 224

# ==============================
# XML PARSER
# ==============================

def parse_xml(xml_file):

    tree = ET.parse(xml_file)
    root = tree.getroot()

    obj = root.find("object")

    label = obj.find("name").text.lower().strip()

    bbox = obj.find("bndbox")

    xmin = int(bbox.find("xmin").text)
    ymin = int(bbox.find("ymin").text)
    xmax = int(bbox.find("xmax").text)
    ymax = int(bbox.find("ymax").text)

    size = root.find("size")

    width = int(size.find("width").text)
    height = int(size.find("height").text)

    bbox = [
        xmin / width,
        ymin / height,
        xmax / width,
        ymax / height
    ]

    return bbox, class_map[label]


# ==============================
# LOAD DATASET
# ==============================

def load_dataset(img_folder, ann_folder):

    images = []
    boxes = []
    labels = []

    for file in os.listdir(img_folder):

        img_path = os.path.join(img_folder, file)
        xml_path = os.path.join(ann_folder, file.replace(".jpg", ".xml"))

        if not os.path.exists(xml_path):
            continue

        img = cv2.imread(img_path)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255.0

        bbox, label = parse_xml(xml_path)

        images.append(img)
        boxes.append(bbox)
        labels.append(label)

    return np.array(images), np.array(boxes), np.array(labels)


print("Loading dataset...")

X_train, y_box_train, y_cls_train = load_dataset(train_images, train_ann)
X_val, y_box_val, y_cls_val = load_dataset(val_images, val_ann)

print("Train:", X_train.shape)
print("Val:", X_val.shape)


# ==============================
# DATA AUGMENTATION
# ==============================

data_aug = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.1)
])


# ==============================
# PRETRAINED MODEL
# ==============================

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False


inputs = layers.Input(shape=(224, 224, 3))

x = data_aug(inputs)

x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

x = base_model(x)

x = layers.GlobalAveragePooling2D()(x)

x = layers.BatchNormalization()(x)

x = layers.Dense(
    256,
    activation='relu',
    kernel_regularizer=tf.keras.regularizers.l2(0.001)
)(x)

x = layers.Dropout(0.5)(x)


# ==============================
# OUTPUTS
# ==============================

bbox_output = layers.Dense(4, activation="sigmoid", name="bbox")(x)

class_output = layers.Dense(
    3,
    activation="softmax",
    name="class"
)(x)

model = models.Model(inputs, [bbox_output, class_output])


# ==============================
# COMPILE
# ==============================

model.compile(

    optimizer=tf.keras.optimizers.Adam(1e-4),

    loss={
        "bbox": tf.keras.losses.Huber(),
        "class": "sparse_categorical_crossentropy"
    },

    loss_weights={
        "bbox": 1.0,
        "class": 2.0
    },

    metrics={
        "class": "accuracy"
    }
)

model.summary()


# ==============================
# CALLBACKS
# ==============================

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=6,
    restore_best_weights=True
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.3,
    patience=3,
    min_lr=1e-6
)


# ==============================
# TRAIN
# ==============================

history = model.fit(

    X_train,
    {"bbox": y_box_train, "class": y_cls_train},

    validation_data=(
        X_val,
        {"bbox": y_box_val, "class": y_cls_val}
    ),

    epochs=30,
    batch_size=8,

    callbacks=[early_stop, reduce_lr]
)


# ==============================
# SAVE MODEL
# ==============================

model.save("object_detector.keras")

print("Model saved")


# ==============================
# PLOTS
# ==============================

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)

plt.plot(history.history['class_accuracy'], label="Train Acc")
plt.plot(history.history['val_class_accuracy'], label="Val Acc")

plt.title("Classification Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.legend()


plt.subplot(1, 2, 2)

plt.plot(history.history['loss'], label="Train Loss")
plt.plot(history.history['val_loss'], label="Val Loss")

plt.title("Total Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend()

plt.show()