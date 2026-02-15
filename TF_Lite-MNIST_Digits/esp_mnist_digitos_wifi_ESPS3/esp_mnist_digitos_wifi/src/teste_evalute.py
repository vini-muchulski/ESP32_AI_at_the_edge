import csv
import json
import os
import statistics
import time
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import requests
import tensorflow as tf

# Basic configuration
BASE_DIR = Path(__file__).resolve().parent
ESP32_IP = "192.168.3.22"
ESP32_PORT = 80
PREDICT_URL = f"http://{ESP32_IP}:{ESP32_PORT}/predict"
STATUS_URL = f"http://{ESP32_IP}:{ESP32_PORT}/status"
INPUT_INFO_PATH = BASE_DIR / "input_info.json"
REQUEST_TIMEOUT = 20.0

N_SAMPLES = 50
START_INDEX = 0
SAVE_CSV = True
CSV_PATH = BASE_DIR / "eval_results.csv"

CLASS_NAMES = [str(i) for i in range(10)]


def get_mnist_test_set() -> tuple[np.ndarray, np.ndarray]:
    (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    return x_test, y_test


def pick_indices(total: int, n: int, start: int) -> np.ndarray:
    end = min(total, start + n)
    return np.arange(start, end)


def quantize_image(image: np.ndarray, input_info: dict) -> np.ndarray:
    scale = float(input_info["scale"])
    zero_point = int(input_info["zero_point"])
    dtype = str(input_info["dtype"])
    q = np.round((image.astype(np.float32) / 255.0) / scale + zero_point)
    if dtype == "int8":
        return np.clip(q, -128, 127).astype(np.int8)
    if dtype == "uint8":
        return np.clip(q, 0, 255).astype(np.uint8)
    raise ValueError(f"Unsupported input dtype: {dtype}")


def check_status(session: requests.Session) -> bool:
    try:
        response = session.get(STATUS_URL, timeout=(3, 10))
        response.raise_for_status()
        status = response.json()
        return bool(status.get("model_initialized", False))
    except Exception:
        return False


def send_pixels(session: requests.Session, image: np.ndarray, input_info: dict) -> tuple[dict, float]:
    dtype = str(input_info["dtype"])
    if dtype == "int8":
        qimg = quantize_image(image, input_info)
        payload = {"q_pixels": qimg.flatten().astype(int).tolist()}
    else:
        payload = {"pixels": image.flatten().astype(np.uint8).tolist()}
    t0 = time.perf_counter()
    response = session.post(PREDICT_URL, json=payload, timeout=(5, REQUEST_TIMEOUT))
    response.raise_for_status()
    dt_ms = (time.perf_counter() - t0) * 1000.0
    return response.json(), dt_ms


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(results: list[dict]) -> tuple[float, float, float, float, float]:
    corrects = [r["correct"] for r in results]
    lats = [r["lat_ms"] for r in results if r["lat_ms"] >= 0.0]
    accuracy = sum(corrects) / len(results) if results else 0.0
    mean_lat = statistics.fmean(lats) if lats else float("nan")
    p50 = float(np.percentile(lats, 50)) if lats else float("nan")
    p95 = float(np.percentile(lats, 95)) if lats else float("nan")
    p99 = float(np.percentile(lats, 99)) if lats else float("nan")
    return accuracy, mean_lat, p50, p95, p99


def main() -> None:
    input_info = json.loads(INPUT_INFO_PATH.read_text(encoding="utf-8"))
    x_test, y_test = get_mnist_test_set()
    idxs = pick_indices(len(x_test), N_SAMPLES, START_INDEX)

    results: list[dict] = []

    with requests.Session() as session:
        if not check_status(session):
            print("status:false")

        for k, idx in enumerate(idxs, 1):
            image = x_test[idx]
            true_label = int(y_test[idx])

            try:
                prediction, lat_ms = send_pixels(session, image, input_info)
                ok = bool(prediction.get("success", False))
                pred_label = int(prediction.get("predicted_digit", -1)) if ok else -1
                conf = float(prediction.get("confidence", 0.0)) if ok else 0.0
                heap = int(prediction.get("heap_free", -1))
                err = "" if ok else str(prediction.get("error_message", ""))
            except Exception as exc:
                ok = False
                pred_label = -1
                conf = 0.0
                lat_ms = -1.0
                heap = -1
                err = str(exc)

            correct = int(ok and pred_label == true_label)
            pred_name = CLASS_NAMES[pred_label] if 0 <= pred_label < 10 else "unknown"

            row = {
                "idx": int(idx),
                "true_id": true_label,
                "true": CLASS_NAMES[true_label],
                "pred_id": pred_label,
                "pred": pred_name,
                "ok": int(ok),
                "correct": correct,
                "conf": round(conf, 6),
                "lat_ms": round(lat_ms, 3),
                "heap_free": heap,
                "error": err,
            }
            results.append(row)

            print(
                f"{k}/{len(idxs)} idx={idx} true={row['true']} "
                f"pred={row['pred']} ok={row['ok']} lat_ms={row['lat_ms']}"
            )

    acc, mean_lat, p50, p95, p99 = summarize(results)
    print("")
    print(
        f"samples={len(results)} accuracy={acc:.4f} "
        f"mean_ms={mean_lat:.1f} p50_ms={p50:.1f} p95_ms={p95:.1f} p99_ms={p99:.1f}"
    )

    if SAVE_CSV:
        write_csv(results, CSV_PATH)
        print(f"csv={CSV_PATH}")

    print("last_response:")
    if results:
        print(json.dumps(results[-1], indent=2))


if __name__ == "__main__":
    main()
