

import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Dropout
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
import os
import matplotlib.pyplot as plt

# --- Configurações ---
INPUT_SHAPE = (224, 224, 3)
NUM_CLASSES = 10
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
INITIAL_EPOCHS = 10
FINE_TUNE_EPOCHS = 10
FINE_TUNE_AT = -20

# --- Callbacks ---
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=3,
    min_lr=1e-6
)

# --- Caminhos dos Arquivos ---
H5_MODEL_PATH = "cifar10_EfficientNetB0_small_finetuned.keras"
TFLITE_MODEL_PATH = "cifar10_EfficientNetB0_small_finetuned.tflite"
TFLITE_INT8_MODEL_PATH = "cifar10_EfficientNetB0_small_finetuned_int8.tflite"

def load_and_preprocess_data():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    y_train = to_categorical(y_train, NUM_CLASSES)
    y_test = to_categorical(y_test, NUM_CLASSES)

    def format_image(image, label):
        image = tf.cast(image, tf.float32)
        image = tf.image.resize(image, (INPUT_SHAPE[0], INPUT_SHAPE[1]))
        image = preprocess_input(image)
        return image, label

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_ds = (
        train_ds.shuffle(buffer_size=1024)
        .map(format_image, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_ds = (
        test_ds.map(format_image, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )
    return train_ds, test_ds

def build_model():
    base_model = EfficientNetB0(
        input_shape=INPUT_SHAPE,
        include_top=False,
        weights='imagenet',
        pooling='avg'  # Equivalente ao GlobalAveragePooling2D
    )
    base_model.trainable = False

    inputs = Input(shape=INPUT_SHAPE)
    x = preprocess_input(inputs)  # Pré-processamento específico do EfficientNet
    x = base_model(x, training=False)
    x = Dropout(0.2)(x)  # Mesmo dropout usado no cats/dogs
    outputs = Dense(NUM_CLASSES, activation='softmax')(x)

    model = Model(inputs, outputs)
    return model, base_model


def plot_training_history(history, history_fine):
    """
    Plota as métricas de acurácia e perda do treinamento inicial e do fine-tuning.
    """
    acc = history.history['accuracy'] + history_fine.history['accuracy']
    val_acc = history.history['val_accuracy'] + history_fine.history['val_accuracy']
    loss = history.history['loss'] + history_fine.history['loss']
    val_loss = history.history['val_loss'] + history_fine.history['val_loss']

    initial_epochs = len(history.history['accuracy'])
    total_epochs = len(acc)

    plt.figure(figsize=(14, 6))

    # Gráfico de Acurácia
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Acurácia de Treino')
    plt.plot(val_acc, label='Acurácia de Validação')
    plt.axvline(initial_epochs - 1, color='gray', linestyle='--', label='Início do Fine-Tuning')
    plt.title('Acurácia do Modelo')
    plt.xlabel('Época')
    plt.ylabel('Acurácia')
    plt.ylim([min(plt.ylim()), 1])
    plt.legend(loc='lower right')

    # Gráfico de Perda
    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Perda de Treino')
    plt.plot(val_loss, label='Perda de Validação')
    plt.axvline(initial_epochs - 1, color='gray', linestyle='--', label='Início do Fine-Tuning')
    plt.title('Perda do Modelo')
    plt.xlabel('Época')
    plt.ylabel('Perda')
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.show()




def save_tflite_model(model, path, is_quantized=False, representative_ds=None):
    """
    Converte um modelo Keras para TFLite, com configuração estrita para
    quantização full-integer.
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if is_quantized:
        if representative_ds is None:
            raise ValueError("representative_dataset é necessário para quantização.")

        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_ds
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        converter._experimental_new_quantizer = True

    tflite_model = converter.convert()

    with open(path, 'wb') as f:
        f.write(tflite_model)
    print(f"Modelo salvo em {path}")

# --- 1. Carregar Dados ---
train_ds, test_ds = load_and_preprocess_data()

# --- 2. Construir Modelo ---
model, base_model = build_model()
model.summary()

# --- 3. Treinamento Inicial (Feature Extraction) ---
print("\n--- Iniciando Treinamento Inicial ---")
model.compile(
    optimizer=AdamW(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_ds,
    epochs=INITIAL_EPOCHS,
    validation_data=test_ds,
    callbacks=[early_stopping, reduce_lr]
)

loss, accuracy = model.evaluate(test_ds)
print(f"\nAcurácia (Transfer Learning): {accuracy * 100:.2f}%")

# --- 4. Fine-Tuning ---
print("\n--- Iniciando Fine-Tuning ---")
base_model.trainable = True
for layer in base_model.layers[:FINE_TUNE_AT]:
    layer.trainable = False

model.compile(
    optimizer=AdamW(learning_rate=LEARNING_RATE / 10, weight_decay=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

fine_tune_start_epoch = history.epoch[-1] + 1
total_epochs = fine_tune_start_epoch + FINE_TUNE_EPOCHS +1

history_fine = model.fit(
    train_ds,
    epochs=total_epochs,
    initial_epoch=fine_tune_start_epoch,
    validation_data=test_ds,
    callbacks=[early_stopping, reduce_lr]
)

loss, accuracy = model.evaluate(test_ds)
print(f"\nAcurácia Final: {accuracy * 100:.2f}%")

plot_training_history(history, history_fine)

# --- 5. Salvar Modelos ---
model.save(H5_MODEL_PATH)
print(f"Modelo Keras salvo em {H5_MODEL_PATH}")

save_tflite_model(model, TFLITE_MODEL_PATH)

# --- 6. Quantização INT8 ---
def representative_dataset_gen():
    """
    Gera um dataset representativo usando uma amostra aleatória do
    conjunto de treinamento para uma calibração mais robusta.
    """
    (x_train, _), _ = cifar10.load_data()

    num_calibration_samples = 20 # Hiperparâmetro: um valor robusto

    # Gera índices aleatórios para garantir uma amostra diversificada
    indices = np.random.choice(x_train.shape[0], num_calibration_samples, replace=False)

    for i in indices:
        image = x_train[i].astype(np.float32)
        image_resized = tf.image.resize(image, (INPUT_SHAPE[0], INPUT_SHAPE[1]))
        image_preprocessed = preprocess_input(image_resized)
        yield [np.expand_dims(image_preprocessed, axis=0)]

save_tflite_model(
    model,
    TFLITE_INT8_MODEL_PATH,
    is_quantized=True,
    representative_ds=representative_dataset_gen,

)

print("\n--- Comparação de Tamanhos ---")
print(f"  Keras (.keras): {os.path.getsize(H5_MODEL_PATH) / (1024*1024):.2f} MB")
print(f"  TFLite (FP32):  {os.path.getsize(TFLITE_MODEL_PATH) / (1024*1024):.2f} MB")
print(f"  TFLite (INT8):  {os.path.getsize(TFLITE_INT8_MODEL_PATH) / (1024*1024):.2f} MB")

import tensorflow as tf
import numpy as np


def evaluate_tflite_int8_model(model_path, num_images_to_test=None):
    print(f"\nAvaliando modelo INT8: {model_path}")

    (_, _), (x_test, y_test) = cifar10.load_data()

    interpreter = tf.lite.Interpreter(
    model_path=model_path,
    experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
)


    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    input_scale, input_zero_point = input_details["quantization"]
    output_scale, output_zero_point = output_details["quantization"]

    input_shape = input_details['shape']
    height, width = input_shape[1], input_shape[2]

    correct_predictions = 0
    total_images = len(x_test) if num_images_to_test is None else num_images_to_test

    for i in range(total_images):
        image_float32 = x_test[i].astype(np.float32)
        image_resized = tf.image.resize(image_float32, [height, width])
        image_preprocessed = preprocess_input(image_resized)

        image_quantized = (image_preprocessed / input_scale) + input_zero_point
        image_quantized = np.expand_dims(image_quantized.numpy().astype(input_details["dtype"]), axis=0)

        interpreter.set_tensor(input_details['index'], image_quantized)
        interpreter.invoke()

        output_data = interpreter.get_tensor(output_details['index'])
        output_dequantized = (output_data.astype(np.float32) - output_zero_point) * output_scale

        predicted_label = np.argmax(output_dequantized)
        true_label = y_test[i][0]

        if predicted_label == true_label:
            correct_predictions += 1

    accuracy = (correct_predictions / total_images) * 100
    print(f"Acurácia do modelo TFLite INT8: {accuracy:.2f}% ({correct_predictions}/{total_images})")
    return accuracy

evaluate_tflite_int8_model(TFLITE_INT8_MODEL_PATH)