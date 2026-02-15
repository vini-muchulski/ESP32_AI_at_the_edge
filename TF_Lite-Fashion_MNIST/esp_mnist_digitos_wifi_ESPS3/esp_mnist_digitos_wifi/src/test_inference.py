import json
import os
import warnings
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import requests
import tensorflow as tf

# Configuration
ESP32_IP = "192.168.3.22"
ESP32_PORT = 80

REQUEST_TIMEOUT = 15.0
SAVE_PLOT_PATH = Path(__file__).resolve().parent / "inference_result.png"
INPUT_INFO_PATH = Path(__file__).resolve().parent / "input_info.json"
SHOW_PLOT = True


IMAGE_INDEX = 55
CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def class_name_from_index(index: int | str) -> str:
    if isinstance(index, int) and 0 <= index < len(CLASS_NAMES):
        return CLASS_NAMES[index]
    return str(index)


def get_prediction_info(prediction_data: dict) -> tuple[int | str, str]:
    predicted_class = prediction_data.get("predicted_class", prediction_data.get("predicted_digit", "N/A"))
    predicted_name = class_name_from_index(predicted_class)
    return predicted_class, predicted_name


def get_fashion_image(index: int) -> tuple[np.ndarray, int]:
    (_, _), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    if index < 0 or index >= len(x_test):
        raise IndexError(
            f"Index {index} is out of range for test dataset (0-{len(x_test) - 1})."
        )
    return x_test[index], int(y_test[index])


def load_input_info(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Input info file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def quantize_image(image_data: np.ndarray, input_info: dict) -> np.ndarray:
    scale = float(input_info["scale"])
    zero_point = int(input_info["zero_point"])
    dtype = str(input_info["dtype"])

    if scale <= 0:
        raise ValueError(f"Invalid quantization scale: {scale}")

    normalized = image_data.astype(np.float32) / 255.0
    quantized = np.round(normalized / scale + zero_point)

    if dtype == "int8":
        return np.clip(quantized, -128, 127).astype(np.int8)
    if dtype == "uint8":
        return np.clip(quantized, 0, 255).astype(np.uint8)

    raise ValueError(f"Unsupported input dtype in input_info.json: {dtype}")


def send_image_for_inference(api_url: str, image_data: np.ndarray, timeout: float) -> dict:
    input_info = load_input_info(INPUT_INFO_PATH)
    quantized_image = quantize_image(image_data, input_info)
    payload = {"q_pixels": quantized_image.flatten().astype(int).tolist()}
    print(f"Sending request to {api_url}...")
    response = requests.post(api_url, json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


def print_results(image_index: int, true_label: int, prediction_data: dict) -> None:
    predicted_class, pred_label_name = get_prediction_info(prediction_data)
    confidence = float(prediction_data.get("confidence", 0.0))
    success = bool(prediction_data.get("success", False))
    error_message = prediction_data.get("error_message", "")
    true_label_name = class_name_from_index(true_label)

    print("\n--- Inference Results ---")
    print(f"Image index: {image_index}")
    print(f"True label: {true_label} ({true_label_name})")
    print(f"ESP32 prediction: {predicted_class} ({pred_label_name})")
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

    
    warnings.filterwarnings(
        "ignore",
        message="Unable to import Axes3D.*",
        category=UserWarning,
    )

    import matplotlib

    if show_plot:
        # Avoid Qt backends (xcb plugin issues in many Linux setups).
        try:
            matplotlib.use("TkAgg", force=True)
        except Exception:
            matplotlib.use("Agg", force=True)
            show_plot = False
            print("Interactive backend unavailable, falling back to image-only mode.")
    else:
        matplotlib.use("Agg", force=True)

    import matplotlib.pyplot as plt

    predicted_class, pred_label_name = get_prediction_info(prediction_data)
    true_label_name = class_name_from_index(true_label)

    plt.imshow(image, cmap="gray")
    plt.title(
        f"Sent to ESP32\nTrue: {true_label_name} ({true_label}) | Pred: {pred_label_name} ({predicted_class})"
    )
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
    image, true_label = get_fashion_image(IMAGE_INDEX)
    result = send_image_for_inference(api_url, image, REQUEST_TIMEOUT)
    print_results(IMAGE_INDEX, true_label, result)

    output_path = None if str(SAVE_PLOT_PATH).strip() == "" else SAVE_PLOT_PATH
    maybe_save_plot(image, true_label, result, output_path, SHOW_PLOT)


if __name__ == "__main__":
    main()
