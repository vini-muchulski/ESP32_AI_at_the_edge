import os, warnings, logging, time, statistics, requests, numpy as np, tensorflow as tf
import matplotlib.pyplot as plt
from typing import List, Dict

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')
logging.getLogger('tensorflow').setLevel(logging.ERROR)

ESP32_IP = "192.168.0.115"
PREDICT_URL = f"http://{ESP32_IP}/predict_bin"
STATUS_URL = f"http://{ESP32_IP}/status"
REQUEST_TIMEOUT = 120
TFLITE_PATH = "cifar10_mobilenetv2_finetuned_int8.tflite"

N_SAMPLES = 50
START_INDEX = 0
SHUFFLE = True
SEED = 42
SAVE_CSV = True
CSV_PATH = "eval_results.csv"
PLOT_CM = True

CLASS_NAMES = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

def get_qparams(tflite_path):
    it = tf.lite.Interpreter(model_path=tflite_path, experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
    it.allocate_tensors()
    d = it.get_input_details()[0]
    return d["quantization"]

def quantize_mobilenet(img, s, zp):
    x = tf.convert_to_tensor(img, dtype=tf.float32)
    x = tf.image.resize(x, [96, 96])
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    q = tf.cast(tf.round(x / s) + zp, tf.int8)
    return q.numpy()

def load_test_set():
    (_, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    y_test = y_test.reshape(-1).astype(int)
    return x_test, y_test

def pick_indices(total, n, start, shuffle, seed):
    rng = np.random.default_rng(seed) if shuffle else None
    idx = np.arange(total)
    if shuffle:
        rng.shuffle(idx)
    if not shuffle and start > 0:
        idx = idx[start:]
    return idx[:n]

def send_bin(session, qimg):
    data = qimg.tobytes()
    t0 = time.perf_counter()
    r = session.post(PREDICT_URL, data=data, headers={"Content-Type": "application/octet-stream"}, timeout=(5, REQUEST_TIMEOUT))
    r.raise_for_status()
    dt = (time.perf_counter() - t0) * 1000.0
    return r.json(), dt

def check_status(session):
    try:
        r = session.get(STATUS_URL, timeout=(3, 10))
        r.raise_for_status()
        j = r.json()
        return bool(j.get("model_initialized", False))
    except Exception:
        return False

def write_csv(rows: List[Dict], path: str):
    import csv
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow(row)

def percentiles(values, ps):
    return [np.percentile(values, p) if values else float("nan") for p in ps]

def plot_cm(cm: np.ndarray, labels: List[str]):
    fig, ax = plt.subplots(figsize=(6,6))
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

def summarize(results: List[Dict]):
    corrects = [r["correct"] for r in results]
    acc = sum(corrects) / len(results) if results else 0.0
    lats = [r["lat_ms"] for r in results if r["lat_ms"] >= 0.0]
    p50, p95, p99 = percentiles(lats, [50, 95, 99])
    mean_lat = statistics.fmean(lats) if lats else float("nan")
    return acc, mean_lat, p50, p95, p99



def main():
    x_test, y_test = load_test_set()
    s, zp = get_qparams(TFLITE_PATH)
    idxs = pick_indices(len(x_test), N_SAMPLES, START_INDEX, SHUFFLE, SEED)
    cm = np.zeros((10, 10), dtype=int)
    results = []
    with requests.Session() as session:
        if not check_status(session):
            print("status:false")
        for k, i in enumerate(idxs, 1):
            img = x_test[i]
            y = int(y_test[i])
            try:
                qimg = quantize_mobilenet(img, s, zp)
                res, lat = send_bin(session, qimg)
                ok = bool(res.get("success", False))
                pred = int(res.get("predicted_class", -1)) if ok else -1
                conf = float(res.get("confidence", 0.0)) if ok else 0.0
                heap = int(res.get("heap_free", -1)) if "heap_free" in res else -1
                err = "" if ok else str(res.get("error_message", ""))
            except Exception as ex:
                ok = False
                pred = -1
                conf = 0.0
                lat = -1.0
                heap = -1
                err = str(ex)
            if ok and 0 <= y < 10 and 0 <= pred < 10:
                cm[y, pred] += 1
            tn = CLASS_NAMES[y]
            pn = CLASS_NAMES[pred] if 0 <= pred < 10 else "unknown"
            correct = int(ok and 0 <= y < 10 and 0 <= pred < 10 and pred == y)
            results.append({
                "idx": i,
                "true_id": y,
                "true": tn,
                "pred_id": pred,
                "pred": pn,
                "ok": int(ok),
                "correct": correct,
                "conf": round(conf, 6),
                "lat_ms": round(lat, 3),
                "heap_free": heap,
                "error": err
            })
            print(f"{k}/{len(idxs)} idx={i} true={tn} pred={pn} ok={int(ok)} lat_ms={round(lat,3)}")
    acc, mean_lat, p50, p95, p99 = summarize(results)
    print("")
    print(f"samples={len(results)} accuracy={acc:.4f} mean_ms={mean_lat:.1f} p50_ms={p50:.1f} p95_ms={p95:.1f} p99_ms={p99:.1f}")
    if SAVE_CSV:
        write_csv(results, CSV_PATH)
        print(f"csv={CSV_PATH}")
    if PLOT_CM:
        plot_cm(cm, CLASS_NAMES)


if __name__ == "__main__":
    main()
