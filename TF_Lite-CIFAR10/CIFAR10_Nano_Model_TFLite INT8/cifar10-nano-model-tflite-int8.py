#!/usr/bin/env python
# coding: utf-8

# # Setup

# In[1]:


import os
import logging
import warnings
import numpy as np
import random
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


logging.getLogger("tensorflow").setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# In[2]:


EPOCHS       = 100
BATCH_SIZE   = 64
PATIENCE_ES  = 10
PATIENCE_REDUCELR_ON_PLATEAU = 5

MODEL_PATH   = "best_model.keras"
TFLITE_PATH  = "model_simple_int8.tflite"
MONITOR_MET  = "val_accuracy"
LR = 0.01

IMG_SIZE = (32, 32)
INPUT_SHAPE = (*IMG_SIZE, 3)

AUTOTUNE = tf.data.AUTOTUNE

# # Dataset

# In[3]:


def get_datasets():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    y_train = y_train.flatten()
    y_test = y_test.flatten()

    indices = np.arange(len(x_train))
    np.random.shuffle(indices)
    x_train = x_train[indices]
    y_train = y_train[indices]

    val_size = int(len(x_train) * 0.1)
    x_val, y_val = x_train[:val_size], y_train[:val_size]
    x_train, y_train = x_train[val_size:], y_train[val_size:]

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(BATCH_SIZE)
    val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(BATCH_SIZE)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(BATCH_SIZE)

    print(f"Training samples: {len(x_train)}, Validation samples: {len(x_val)}, Test samples: {len(x_test)}")

    return train_ds, val_ds, test_ds

def preprocess(image, label):
    return tf.cast(image, tf.float32) / 255.0, label

def prepare_dataset(ds, is_training=False):
    ds = ds.cache()
    if is_training:
        ds = ds.shuffle(buffer_size=1000)
    ds = ds.map(preprocess, num_parallel_calls=AUTOTUNE)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds

raw_train_ds, raw_val_ds, raw_test_ds = get_datasets()
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
train_ds = prepare_dataset(raw_train_ds, is_training=True)
val_ds = prepare_dataset(raw_val_ds)
test_ds = prepare_dataset(raw_test_ds)

# In[15]:


def plot_samples(dataset, class_names, num_samples=9):
    plt.figure(figsize=(10, 10))
    for images, labels in dataset.take(1):
        for i in range(num_samples):
            plt.subplot(3, 3, i + 1)
            img_array = images[i].numpy().astype("uint8")
            label_index = int(labels[i].numpy())
            plt.imshow(img_array)
            plt.title(class_names[label_index])
            plt.axis("off")
    plt.show()

plot_samples(raw_train_ds, CLASS_NAMES)

# # Helpers

# In[5]:


def get_callbacks():
    return [
        EarlyStopping(monitor=MONITOR_MET, patience=PATIENCE_ES, verbose=0, restore_best_weights=True),
        ModelCheckpoint(MODEL_PATH, monitor=MONITOR_MET, save_best_only=True, verbose=0),
        ReduceLROnPlateau(monitor=MONITOR_MET, factor=0.1, patience=PATIENCE_REDUCELR_ON_PLATEAU, min_lr=1e-6, verbose=1)
    ]

def plot_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="val")
    plt.title("Loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history["accuracy"], label="train")
    plt.plot(history.history["val_accuracy"], label="val")
    plt.title("Accuracy")
    plt.legend()
    plt.show()

def evaluate_model(model, test_data):
    loss, acc = model.evaluate(test_data, verbose=0)
    print(f"Test set accuracy: {acc:.4f}")
    return loss, acc

# # Model

# In[6]:


def create_balanced_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(32, 32, 3)),

        tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'), # Conv2D
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'), # Conv2D
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'), # Conv2D
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'), # Conv2D
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model

# In[7]:


model = create_balanced_model()

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
model.summary()

# # Train

# In[8]:


print(f"\n Start training - {EPOCHS} epochs (batch size={BATCH_SIZE})")
history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
    callbacks=get_callbacks(),
    verbose=1
)

# In[9]:


print("\n Evaluating on the test set")
evaluate_model(model, test_ds)
print(f"Number of classes {len(CLASS_NAMES)}"  )

# In[10]:


print("\n Plotting training curve")
plot_history(history)

# In[17]:


def predict_random_samples(model, dataset, class_names, num_samples=10):
    subset = dataset.unbatch().shuffle(buffer_size=1000, seed=None).take(num_samples)
    images, true_labels = [], []
    for img, lbl in subset:
        images.append(img.numpy())
        true_labels.append(int(lbl.numpy()))
    images_arr = np.array(images)
    pred_labels = np.argmax(model.predict(images_arr, verbose=0), axis=1)
    rows = int(np.ceil(num_samples / 5))
    cols = min(num_samples, 5)
    plt.figure(figsize=(15, rows * 3))
    for i in range(num_samples):
        plt.subplot(rows, cols, i + 1)
        img_display = (images_arr[i] * 255).astype("uint8")
        plt.imshow(img_display)
        color = 'green' if true_labels[i] == pred_labels[i] else 'red'
        plt.title(f"T: {class_names[true_labels[i]]}\nP: {class_names[pred_labels[i]]}", color=color, fontsize=9)
        plt.axis("off")
    plt.tight_layout()
    plt.show()

predict_random_samples(model, test_ds, CLASS_NAMES, num_samples=20)

# # TFLite Export

# In[12]:


%%time

def create_representative_dataset_generator(dataset, num_samples=20):
    def representative_dataset():
        for images, _ in dataset.take(num_samples).unbatch().batch(1):
            yield [tf.cast(images, tf.float32)]
    return representative_dataset

def export_quantized_tflite(keras_model, representative_dataset_gen, tflite_path):
    print(f"\n🛠️ Converting and quantizing to TFLite (INT8) in `{tflite_path}`...")

    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.int8

    tflite_model_quant = converter.convert()

    with open(tflite_path, "wb") as f:
        f.write(tflite_model_quant)

    keras_size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    tflite_size_kb = len(tflite_model_quant) / 1024
    print(f"✅Export completed.  Keras: {keras_size_mb:.2f} MB |  TFLite: {tflite_size_kb:.1f} KB")



representative_data_gen = create_representative_dataset_generator(train_ds)
export_quantized_tflite(model, representative_data_gen, TFLITE_PATH)

# In[13]:


import numpy as np
import tensorflow as tf

def evaluate_tflite_universal(tflite_path, dataset):
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    input_scale, input_zero = input_details["quantization"]
    output_scale, output_zero = output_details["quantization"]

    correct = 0
    total = 0

    print(f"Testing TFLite Universal: {tflite_path}")

    for images, labels in dataset.unbatch().batch(1):
        img = images[0].numpy()
        lbl_raw = labels[0].numpy()

        if lbl_raw.size == 1:
            label = int(lbl_raw)
        else:
            label = np.argmax(lbl_raw)

        if input_details['dtype'] == np.uint8:
            img = img / input_scale + input_zero
            img = np.clip(img, 0, 255).astype(np.uint8)

        interpreter.set_tensor(input_details["index"], [img])
        interpreter.invoke()
        output = interpreter.get_tensor(output_details["index"])[0]

        if output_details['dtype'] != np.float32:
             output = (output.astype(np.float32) - output_zero) * output_scale

        pred = np.argmax(output)

        if pred == label:
            correct += 1
        total += 1

    acc = correct / total if total > 0 else 0
    print(f"Accuracy TFLite: {acc:.4f}")
    return acc

# In[14]:


evaluate_tflite_universal(TFLITE_PATH, test_ds)

# In[18]:


def predict_random_samples_tfliteint8(tflite_path, dataset, class_names, num_samples=20):
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    input_scale, input_zero = input_details["quantization"]
    output_scale, output_zero = output_details["quantization"]

    subset = dataset.unbatch().shuffle(buffer_size=1000, seed=None).take(num_samples)
    images, true_labels = [], []
    for img, lbl in subset:
        images.append(img.numpy())
        true_labels.append(int(lbl.numpy()))

    pred_labels = []
    for img in images:
        inp = img.copy()
        if input_details["dtype"] == np.uint8:
            inp = inp / input_scale + input_zero
            inp = np.clip(inp, 0, 255).astype(np.uint8)
        interpreter.set_tensor(input_details["index"], inp[np.newaxis])
        interpreter.invoke()
        output = interpreter.get_tensor(output_details["index"])[0]
        if output_details["dtype"] != np.float32:
            output = (output.astype(np.float32) - output_zero) * output_scale
        pred_labels.append(np.argmax(output))

    rows = int(np.ceil(num_samples / 5))
    cols = min(num_samples, 5)
    plt.figure(figsize=(15, rows * 3))
    for i in range(num_samples):
        plt.subplot(rows, cols, i + 1)
        plt.imshow((images[i] * 255).astype("uint8"))
        color = 'green' if true_labels[i] == pred_labels[i] else 'red'
        plt.title(f"T: {class_names[true_labels[i]]}\nP: {class_names[pred_labels[i]]}", color=color, fontsize=9)
        plt.axis("off")
    plt.tight_layout()
    plt.show()

predict_random_samples_tfliteint8(TFLITE_PATH, test_ds, CLASS_NAMES, num_samples=20)
