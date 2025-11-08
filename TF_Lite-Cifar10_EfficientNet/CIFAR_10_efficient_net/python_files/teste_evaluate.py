import os, warnings, logging, requests, numpy as np, tensorflow as tf, time, random
from tensorflow.keras.datasets import cifar10

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')
logging.getLogger('tensorflow').setLevel(logging.ERROR)

ESP32_IP = "192.168.0.115"
TFLITE_PATH = "cifar10_EfficientNetB0_small_finetuned_int8.tflite"
REQUEST_TIMEOUT = 120
N_IMAGES = 20
SEED = 42
RANDOM_PICK = True

CLASS_NAMES = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

def get_qparams(tflite_path):
    it = tf.lite.Interpreter(model_path=tflite_path, experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
    it.allocate_tensors()
    d = it.get_input_details()[0]
    return d["quantization"]

def quantize(img, s, zp):
    x = tf.convert_to_tensor(img, dtype=tf.float32)
    x = tf.image.resize(x, [224, 224])
    x = tf.keras.applications.efficientnet.preprocess_input(x)
    q = tf.cast(tf.round(x / s) + zp, tf.int8)
    return q.numpy()

def send_image(url, data, timeout):
    r = requests.post(url, data=data.tobytes(), headers={"Content-Type": "application/octet-stream","Content-Length": str(data.nbytes)}, timeout=timeout)
    r.raise_for_status()
    return r.json()

def confusion_matrix(y_true, y_pred, n=10):
    m = np.zeros((n, n), dtype=np.int32)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < n and 0 <= p < n:
            m[t, p] += 1
    return m

def per_class_acc(cm):
    acc = []
    for i in range(cm.shape[0]):
        s = cm[i].sum()
        acc.append((cm[i, i] / s) if s > 0 else 0.0)
    return np.array(acc)

def evaluate(esp_ip, tflite_path, n_images, timeout, seed, random_pick):
    (_, _), (x_test, y_test) = cifar10.load_data()
    y_test = y_test.reshape(-1)
    s, zp = get_qparams(tflite_path)
    url = f"http://{esp_ip}/predict_bin"
    rng = random.Random(seed)
    pool = list(range(len(x_test)))
    idxs = pool[:n_images] if not random_pick else rng.sample(pool, n_images)
    y_true, y_pred, confs, lats, heaps = [], [], [], [], []
    ok = 0
    fail = 0
    for k, idx in enumerate(idxs, 1):
        img = x_test[idx]
        qimg = quantize(img, s, zp)
        t0 = time.perf_counter()
        try:
            res = send_image(url, qimg, timeout)
            lat = (time.perf_counter() - t0) * 1000.0
            if res.get("success"):
                pred = int(res.get("predicted_class", -1))
                conf = float(res.get("confidence", 0.0))
                heap = int(res.get("heap_free", -1)) if "heap_free" in res else -1
                y_true.append(int(y_test[idx]))
                y_pred.append(pred)
                confs.append(conf)
                lats.append(lat)
                heaps.append(heap)
                ac = int(pred == y_test[idx])
                ok += 1
                cname_t = CLASS_NAMES[y_test[idx]]
                cname_p = CLASS_NAMES[pred] if 0 <= pred < 10 else "unknown"
                print(f"[{k:3d}/{n_images:3d}] idx={idx:5d} true={y_test[idx]:1d}({cname_t:10s}) pred={pred:1d}({cname_p:10s}) ok={ac} conf={conf:7.4f} lat={lat:7.1f}ms heap={heap}")
            else:
                fail += 1
                print(f"[{k:3d}/{n_images:3d}] idx={idx:5d} falha: {res.get('error_message','erro')}")
        except requests.exceptions.RequestException as e:
            fail += 1
            print(f"[{k:3d}/{n_images:3d}] idx={idx:5d} exceção: {e}")
    if ok == 0:
        print("Sem amostras válidas.")
        return
    cm = confusion_matrix(np.array(y_true), np.array(y_pred), n=10)
    acc_global = (cm.trace() / cm.sum()) if cm.sum() > 0 else 0.0
    acc_cls = per_class_acc(cm)
    mean_conf = float(np.mean(confs)) if confs else 0.0
    mean_lat = float(np.mean(lats)) if lats else 0.0
    print("\nResumo")
    print(f"Total={n_images} Sucesso={ok} Falhas={fail} Acuracia={acc_global*100:.2f}% ConfMedia={mean_conf:.4f} LatMedia={mean_lat:.1f}ms")
    print("\nAcuracia por classe")
    for i, a in enumerate(acc_cls):
        print(f"{i} {CLASS_NAMES[i]:10s}: {a*100:6.2f}%")
    print("\nMatriz de confusao (linhas=verdade, colunas=predito)")
    header = "       " + " ".join([f"{i:4d}" for i in range(10)])
    print(header)
    for i in range(10):
        row = " ".join([f"{cm[i,j]:4d}" for j in range(10)])
        print(f"{i:2d} {row}")

def main():
    evaluate(ESP32_IP, TFLITE_PATH, N_IMAGES, REQUEST_TIMEOUT, SEED, RANDOM_PICK)

if __name__ == "__main__":
    main()
