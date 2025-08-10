import requests
import json
import numpy as np
import time
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


def main():
    NUM_SAMPLES_TO_TEST = 100
    correct_predictions = 0
    failed_requests = 0

    print("Loading CIFAR-10 test dataset...")
    (_, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    if NUM_SAMPLES_TO_TEST > len(x_test):
        print(f"Error: Number of samples to test ({NUM_SAMPLES_TO_TEST}) is greater than dataset size ({len(x_test)}).")
        return

    print(f"\nStarting evaluation for {NUM_SAMPLES_TO_TEST} samples...")

    for i in range(NUM_SAMPLES_TO_TEST):
        original_image = x_test[i]
        true_label_index = y_test[i][0]
        true_label_name = CLASS_NAMES[true_label_index]

        resized_image_tensor = tf.image.resize(original_image, [96, 96], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        resized_image_uint8 = tf.cast(resized_image_tensor, tf.uint8).numpy()
        
        print(f"  Testing image {i+1:3d}/{NUM_SAMPLES_TO_TEST} (True: {true_label_name:<11}) ... ", end="", flush=True)
        
        result = send_image_for_inference(PREDICT_URL, resized_image_uint8)

        if not result or not result.get("success"):
            error_msg = result.get("error_message", "Request failed") if result else "Request failed"
            print(f"Failed. Error: {error_msg}")
            failed_requests += 1
            continue

        predicted_index = result.get("predicted_class", -1)
        
        if predicted_index == true_label_index:
            correct_predictions += 1
            print("Correct.")
        else:
            predicted_name = CLASS_NAMES[predicted_index] if 0 <= predicted_index < len(CLASS_NAMES) else "Unknown"
            print(f"Incorrect. (Predicted: {predicted_name})")

        time.sleep(1)

    print("\n--- Evaluation Summary ---")
    
    successful_tests = NUM_SAMPLES_TO_TEST - failed_requests
    
    if successful_tests > 0:
        accuracy = (correct_predictions / successful_tests) * 100
        print(f"Successful Tests:    {successful_tests}")
        print(f"Correct Predictions: {correct_predictions}")
        print(f"Accuracy:            {accuracy:.2f}%")
    else:
        print("No tests were completed successfully.")
        
    if failed_requests > 0:
        print(f"Failed Requests:     {failed_requests}")
    print("--------------------------\n")
    
 
        
if __name__ == "__main__":
    main()