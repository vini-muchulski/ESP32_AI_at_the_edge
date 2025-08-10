import requests
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import logging
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')
logging.getLogger('tensorflow').setLevel(logging.ERROR)

ESP32_IP = "192.168.0.116"
PREDICT_URL = f"http://{ESP32_IP}/predict"
REQUEST_TIMEOUT = 10

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

def get_cifar10_sample(index):
    (_, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    image = x_test[index]
    true_label_index = y_test[index][0]
    true_label_name = CLASS_NAMES[true_label_index]
    return image, true_label_index, true_label_name

def send_image_for_inference(url, image_data):
    try:
        pixel_list = image_data.flatten().tolist()
        payload = json.dumps({"pixels": pixel_list})
        headers = {"Content-Type": "application/json"}

        response = requests.post(
            url,
            data=payload,
            headers=headers,
            timeout=REQUEST_TIMEOUT
        )

        response.raise_for_status()
        return response.json()

    except requests.exceptions.RequestException as e:
        print(f"Error connecting to ESP32: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

def plot_prediction(image, true_name, predicted_name, confidence):
    plt.figure(figsize=(4, 4))
    plt.imshow(image)
    plt.title(f"True: {true_name}\nPredicted: {predicted_name} ({confidence:.2f})")
    plt.axis('off')
    plt.show()

def main():
    image_index = 1
    
    image, true_label_index, true_label_name = get_cifar10_sample(image_index)

    print(f"Sending image {image_index} ({true_label_name}) to {PREDICT_URL}...")
    
    result = send_image_for_inference(PREDICT_URL, image)

    if not result:
        print("Inference failed.")
        return

    if result.get("success"):
        predicted_index = result.get("predicted_class", -1)
        confidence = result.get("confidence", 0.0)
        predicted_name = CLASS_NAMES[predicted_index] if predicted_index != -1 else "Unknown"

        print("\n--- Inference Result ---")
        print(f"True Class:      {true_label_index} ({true_label_name})")
        print(f"Predicted Class: {predicted_index} ({predicted_name})")
        print(f"Confidence:      {confidence:.4f}")
        print(f"ESP32 Heap Free: {result.get('heap_free', 'N/A')} bytes")
        print("------------------------\n")

        plot_prediction(image, true_label_name, predicted_name, confidence)
    else:
        error_msg = result.get("error_message", "No error message provided.")
        print(f"Inference failed on ESP32: {error_msg}")

if __name__ == "__main__":
    main()