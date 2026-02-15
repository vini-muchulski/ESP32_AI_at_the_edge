import json
from pathlib import Path

import numpy as np
import requests
import tensorflow as tf

ESP32_IP = "192.168.0.111"
ESP32_PORT = 80
IMAGE_INDEX = 5
REQUEST_TIMEOUT = 15.0
SAVE_PLOT_PATH = Path(__file__).resolve().parent / "inference_result.png"
SHOW_PLOT = False


def get_mnist_image(index: int) -> tuple[np.ndarray, int]:
    (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    if index < 0 or index >= len(x_test):
        raise IndexError(
            f"Index {index} is out of range for test dataset (0-{len(x_test) - 1})."
        )
    return x_test[index], int(y_test[index])


def send_image_for_inference(api_url: str, image_data: np.ndarray, timeout: float) -> dict:
    payload = {"pixels": image_data.flatten().astype(np.uint8).tolist()}
    print(f"Sending request to {api_url}...")
    response = requests.post(api_url, json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


def print_results(image_index: int, true_label: int, prediction_data: dict) -> None:
    predicted_digit = prediction_data.get("predicted_digit", "N/A")
    confidence = float(prediction_data.get("confidence", 0.0))
    success = bool(prediction_data.get("success", False))
    error_message = prediction_data.get("error_message", "")

    print("\n--- Inference Results ---")
    print(f"Image index: {image_index}")
    print(f"True label: {true_label}")
    print(f"ESP32 prediction: {predicted_digit}")
    print(f"Confidence: {confidence:.4f}")
    print(f"Success: {success}")
    if error_message:
        print(f"Error message: {error_message}")
    print("-------------------------\n")
    print("Full API response:")
    print(json.dumps(prediction_data, indent=2))


def maybe_save_plot(
    image: np.ndarray,
    true_label: int,
    prediction_data: dict,
    output_path: Path,
    show_plot: bool,
) -> None:
    if not output_path and not show_plot:
        return

    import matplotlib.pyplot as plt

    predicted_digit = prediction_data.get("predicted_digit", "N/A")
    plt.imshow(image, cmap="gray")
    plt.title(f"Sent to ESP32\nTrue label: {true_label} | Prediction: {predicted_digit}")
    plt.axis("off")

    if output_path:
        plt.savefig(output_path)
        print(f"Plot saved to: {output_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()


def main() -> None:
    api_url = f"http://{ESP32_IP}:{ESP32_PORT}/predict"
    image, true_label = get_mnist_image(IMAGE_INDEX)
    result = send_image_for_inference(api_url, image, REQUEST_TIMEOUT)
    print_results(IMAGE_INDEX, true_label, result)

    output_path = None if str(SAVE_PLOT_PATH).strip() == "" else SAVE_PLOT_PATH
    maybe_save_plot(image, true_label, result, output_path, SHOW_PLOT)


if __name__ == "__main__":
    main()
