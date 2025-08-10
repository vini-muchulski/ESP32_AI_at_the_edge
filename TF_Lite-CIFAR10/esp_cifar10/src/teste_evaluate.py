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

ESP32_IP = "192.168.0.111"
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

def evaluate(n=100):
    correct = 0
    for idx in np.random.choice(10000, n, replace=False):
        img, true_idx, _ = get_cifar10_sample(idx)
        r = send_image_for_inference(PREDICT_URL, img)
        if r and r.get("success") and r["predicted_class"] == true_idx:
            correct += 1
    print(f"Accuracy: {correct}/{n} ({correct/n:.2%})")

if __name__ == "__main__":
    evaluate(100)

# Accuracy: 78/100 (78.00%) no teste dia 06 08 2025
