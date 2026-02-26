import json
import os
import statistics
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import requests
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Configuration
BASE_DIR = Path(__file__).resolve().parent
ESP32_IP = "192.168.3.22"
ESP32_PORT = 80
PREDICT_URL = f"http://{ESP32_IP}:{ESP32_PORT}/predict"
STATUS_URL = f"http://{ESP32_IP}:{ESP32_PORT}/status"
INPUT_INFO_PATH = BASE_DIR / "input_info.json"
REQUEST_TIMEOUT = 45.0
REQUEST_RETRIES = 2
INTER_REQUEST_DELAY = 0.2

N_SAMPLES = 100
START_INDEX = 0
SHUFFLE = True
SEED = 42

SAVE_CSV = True
CSV_PATH = BASE_DIR / "eval_results.csv"
PLOT_CM = True
warnings.filterwarnings("ignore", message="Unable to import Axes3D.*", category=UserWarning)

CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def load_input_info(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Input info file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def quantize_image(image_data: np.ndarray, input_info: dict) -> np.ndarray:
    scale = float(input_info["scale"])
    zero_point = int(input_info["zero_point"])
    dtype = str(input_info["dtype"])
    target_h, target_w = map(int, input_info["size"])

    if scale <= 0:
        raise ValueError(f"Invalid quantization scale: {scale}")

    image_f32 = image_data.astype(np.float32)
    if image_f32.ndim == 2:
        image_f32 = np.expand_dims(image_f32, axis=-1)
    if image_f32.shape[-1] == 1:
        image_f32 = np.repeat(image_f32, 3, axis=-1)

    resized = tf.image.resize(image_f32, (target_h, target_w)).numpy()
    preprocessed = preprocess_input(resized)
    quantized = np.round(preprocessed / scale + zero_point)

    if dtype == "int8":
        return np.clip(quantized, -128, 127).astype(np.int8)
    if dtype == "uint8":
        return np.clip(quantized, 0, 255).astype(np.uint8)

    raise ValueError(f"Unsupported input dtype in input_info.json: {dtype}")


def get_cifar10_test_set() -> Tuple[np.ndarray, np.ndarray]:
    (_, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    return x_test, y_test.reshape(-1)


def pick_indices(total: int, n: int, start: int, shuffle: bool, seed: int) -> np.ndarray:
    indices = np.arange(total)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)
    else:
        indices = indices[start:]

    if n <= 0 or n >= len(indices):
        return indices
    return indices[:n]


def check_status(session: requests.Session) -> bool:
    try:
        response = session.get(STATUS_URL, timeout=(3, 10))
        response.raise_for_status()
        status = response.json()
        return bool(status.get("model_initialized", False))
    except Exception:
        return False


def send_qpixels(session: requests.Session, qimg: np.ndarray) -> Tuple[dict, float]:
    payload = {"q_pixels": qimg.flatten().astype(int).tolist()}
    last_exc: Exception | None = None

    for attempt in range(REQUEST_RETRIES + 1):
        t0 = time.perf_counter()
        try:
            response = session.post(
                PREDICT_URL,
                json=payload,
                timeout=(8, REQUEST_TIMEOUT),
                headers={"Connection": "close"},
            )
            response.raise_for_status()
            dt_ms = (time.perf_counter() - t0) * 1000.0
            return response.json(), dt_ms
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
            last_exc = exc
            if attempt < REQUEST_RETRIES:
                time.sleep(0.5)
                continue
            raise

    # Defensive fallback (should not be reached because the loop raises on final failure).
    raise RuntimeError(f"Request failed: {last_exc}")


def write_csv(rows: List[Dict], path: Path) -> None:
    import csv

    if not rows:
        return

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(results: List[Dict]) -> Tuple[float, float, float, float, float]:
    corrects = [r["correct"] for r in results]
    lats = [r["lat_ms"] for r in results if r["lat_ms"] >= 0.0]

    accuracy = sum(corrects) / len(results) if results else 0.0
    mean_lat = statistics.fmean(lats) if lats else float("nan")
    p50 = float(np.percentile(lats, 50)) if lats else float("nan")
    p95 = float(np.percentile(lats, 95)) if lats else float("nan")
    p99 = float(np.percentile(lats, 99)) if lats else float("nan")
    return accuracy, mean_lat, p50, p95, p99


def summarize_server_timings(results: List[Dict]) -> Tuple[float, float, float, float]:
    recv = [r["receive_ms"] for r in results if r["receive_ms"] >= 0]
    parse = [r["parse_ms"] for r in results if r["parse_ms"] >= 0]
    infer = [r["inference_ms"] for r in results if r["inference_ms"] >= 0]
    total = [r["server_total_ms"] for r in results if r["server_total_ms"] >= 0]
    return (
        statistics.fmean(recv) if recv else float("nan"),
        statistics.fmean(parse) if parse else float("nan"),
        statistics.fmean(infer) if infer else float("nan"),
        statistics.fmean(total) if total else float("nan"),
    )


def plot_confusion_matrix(cm: np.ndarray, labels: List[str]) -> None:
    warnings.filterwarnings("ignore", message="Unable to import Axes3D.*", category=UserWarning)

    import matplotlib

    try:
        matplotlib.use("TkAgg", force=True)
    except Exception:
        matplotlib.use("Agg", force=True)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    ax.set_xlabel("Pred")
    ax.set_ylabel("True")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()


def main() -> None:
    input_info = load_input_info(INPUT_INFO_PATH)
    x_test, y_test = get_cifar10_test_set()
    idxs = pick_indices(len(x_test), N_SAMPLES, START_INDEX, SHUFFLE, SEED)

    cm = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=int)
    results: List[Dict] = []

    with requests.Session() as session:
        if not check_status(session):
            print("status:false")

        for k, idx in enumerate(idxs, 1):
            image = x_test[idx]
            true_label = int(y_test[idx])

            try:
                qimg = quantize_image(image, input_info)
                prediction, lat_ms = send_qpixels(session, qimg)

                ok = bool(prediction.get("success", False))
                pred_label = int(prediction.get("predicted_class", -1)) if ok else -1
                conf = float(prediction.get("confidence", 0.0)) if ok else 0.0
                heap = int(prediction.get("heap_free", -1)) if "heap_free" in prediction else -1
                err = "" if ok else str(prediction.get("error_message", ""))
                receive_ms = float(prediction.get("receive_ms", -1))
                parse_ms = float(prediction.get("parse_ms", -1))
                inference_ms = float(prediction.get("inference_ms", -1))
                server_total_ms = float(prediction.get("total_ms", -1))
            except Exception as exc:
                ok = False
                pred_label = -1
                conf = 0.0
                lat_ms = -1.0
                heap = -1
                err = str(exc)
                receive_ms = -1.0
                parse_ms = -1.0
                inference_ms = -1.0
                server_total_ms = -1.0

            correct = int(ok and pred_label == true_label)
            if ok and 0 <= true_label < len(CLASS_NAMES) and 0 <= pred_label < len(CLASS_NAMES):
                cm[true_label, pred_label] += 1

            results.append(
                {
                    "idx": int(idx),
                    "true_id": true_label,
                    "true": CLASS_NAMES[true_label],
                    "pred_id": pred_label,
                    "pred": CLASS_NAMES[pred_label] if 0 <= pred_label < len(CLASS_NAMES) else "unknown",
                    "ok": int(ok),
                    "correct": correct,
                    "conf": round(conf, 6),
                    "lat_ms": round(lat_ms, 3),
                    "receive_ms": round(receive_ms, 3),
                    "parse_ms": round(parse_ms, 3),
                    "inference_ms": round(inference_ms, 3),
                    "server_total_ms": round(server_total_ms, 3),
                    "heap_free": heap,
                    "error": err,
                }
            )

            print(
                f"{k}/{len(idxs)} idx={idx} true={CLASS_NAMES[true_label]} "
                f"pred={results[-1]['pred']} ok={int(ok)} "
                f"lat_ms={round(lat_ms, 3)} srv_total_ms={round(server_total_ms, 3)} "
                f"{'err=' + err[:120] if not ok and err else ''}"
            )
            time.sleep(INTER_REQUEST_DELAY)

    acc, mean_lat, p50, p95, p99 = summarize(results)
    recv_mean, parse_mean, infer_mean, total_mean = summarize_server_timings(results)
    print("")
    print(
        f"samples={len(results)} accuracy={acc:.4f} "
        f"mean_ms={mean_lat:.1f} p50_ms={p50:.1f} p95_ms={p95:.1f} p99_ms={p99:.1f}"
    )
    print(
        "server_mean_ms="
        f"receive:{recv_mean:.1f} parse:{parse_mean:.1f} "
        f"inference:{infer_mean:.1f} total:{total_mean:.1f}"
    )

    if SAVE_CSV:
        write_csv(results, CSV_PATH)
        print(f"csv={CSV_PATH}")

    if PLOT_CM:
        plot_confusion_matrix(cm, CLASS_NAMES)


if __name__ == "__main__":
    main()
