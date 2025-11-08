import os, warnings, logging, requests, numpy as np, tensorflow as tf
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')
logging.getLogger('tensorflow').setLevel(logging.ERROR)

ESP32_IP = "192.168.0.115"
PREDICT_URL = f"http://{ESP32_IP}/predict_bin"
REQUEST_TIMEOUT = 120
TFLITE_PATH = "cifar10_EfficientNetB0_small_finetuned_int8.tflite"

CLASS_NAMES = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

_qparams = None
def get_qparams():
    global _qparams
    if _qparams is None:
        it = tf.lite.Interpreter(model_path=TFLITE_PATH, experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
        it.allocate_tensors()
        d = it.get_input_details()[0]
        _qparams = d["quantization"]
    return _qparams

def quantize_for_esp(img):
    s, zp = get_qparams()
    x = tf.convert_to_tensor(img, dtype=tf.float32)
    x = tf.image.resize(x, [224, 224])
    x = tf.keras.applications.efficientnet.preprocess_input(x)
    q = tf.cast(tf.round(x / s) + zp, tf.int8)
    return q.numpy()

def load_sample(idx):
    (_, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    return x_test[idx], y_test[idx][0]

def send_image(url, data):
    try:
        r = requests.post(
            url,
            data=data.tobytes(),
            headers={"Content-Type": "application/octet-stream","Content-Length": str(data.nbytes)},
            timeout=REQUEST_TIMEOUT
        )
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        print(f"Erro ao conectar ao ESP32: {e}")
        return None

def plot_result(img, true_idx, pred_idx, conf):
    plt.imshow(img)
    plt.axis('off')
    true_name = CLASS_NAMES[true_idx]
    pred_name = CLASS_NAMES[pred_idx] if pred_idx != -1 else "unknown"
    plt.title(f"True: {true_name}\nPred: {pred_name} ({conf:.2f})")
    plt.show()

def main():
    idx = 18
    img32, true_idx = load_sample(idx)
    qimg = quantize_for_esp(img32)
    print(f"Enviando imagem {idx} ({CLASS_NAMES[true_idx]}) para {PREDICT_URL} ...")
    result = send_image(PREDICT_URL, qimg)
    if not result:
        print("Inferência falhou.")
        return
    if result.get("success"):
        pred_idx = result.get("predicted_class", -1)
        conf = result.get("confidence", 0.0)
        print("\n--- Resultado ---")
        print(f"Classe real:      {true_idx} ({CLASS_NAMES[true_idx]})")
        print(f"Classe predita:   {pred_idx} ({CLASS_NAMES[pred_idx]})")
        print(f"Confiança:        {conf:.4f}")
        print(f"Heap livre ESP32: {result.get('heap_free', 'N/A')} bytes\n")
        print(result)
        plot_result(img32, true_idx, pred_idx, conf)
    else:
        print(f"Inferência falhou no ESP32: {result.get('error_message', 'Sem mensagem.')}")

if __name__ == "__main__":
    main()
