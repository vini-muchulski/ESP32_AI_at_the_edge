import os, warnings, logging, requests, numpy as np, tensorflow as tf
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')
logging.getLogger('tensorflow').setLevel(logging.ERROR)

INDICE_SAMPLE = 13

ESP32_IP = "192.168.0.115"
PREDICT_URL = f"http://{ESP32_IP}/predict_bin"
REQUEST_TIMEOUT = 60

TFLITE_PATH = "cifar10_mobilenetv2_finetuned_int8.tflite"

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

def load_sample(idx):
    (_, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    return x_test[idx], int(y_test[idx][0])

def send_bin(qimg):
    data = qimg.tobytes()
    r = requests.post(
        PREDICT_URL,
        data=data,
        headers={"Content-Type": "application/octet-stream"},
        timeout=(5, REQUEST_TIMEOUT)
    )
    r.raise_for_status()
    return r.json()


def plot_prediction(image, true_name, predicted_name, confidence):
    plt.figure(figsize=(4, 4))
    plt.imshow(image)
    plt.title(f"True: {true_name}\nPredicted: {predicted_name} ({confidence:.2f})")
    plt.axis('off')
    plt.show()



def main():
    idx = INDICE_SAMPLE
    img32, y = load_sample(idx)
    s, zp = get_qparams(TFLITE_PATH)
    qimg = quantize_mobilenet(img32, s, zp)
    res = send_bin(qimg)
    if not res or not res.get("success"):
        print(f"Falha: {res.get('error_message','sem mensagem') if res else 'sem resposta'}")
        return
    pred = int(res.get("predicted_class", -1))
    conf = float(res.get("confidence", 0.0))
    tn = CLASS_NAMES[y]
    pn = CLASS_NAMES[pred] if 0 <= pred < 10 else "unknown"
    match = int(pred == y)
    print("\n--- Resultado ---")
    print(f"True: {y} ({tn})")
    print(f"Pred: {pred} ({pn})")
    print(f"Conf: {conf:.4f}")
    print(f"Correto: {match}")
    print(f"Heap: {res.get('heap_free','N/A')}")
    plot_prediction(img32, tn, pn, conf)




if __name__ == "__main__":
    main()
