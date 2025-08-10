import requests
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

ESP32_IP = "192.168.0.116"
API_URL = f"http://{ESP32_IP}/predict"
IMAGE_INDEX = 3

def get_mnist_image(index: int):
    try:
        (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        if index >= len(x_test):
            raise IndexError(f"Índice {index} fora do alcance para o dataset de teste (0-{len(x_test)-1}).")
        
        image = x_test[index]
        label = y_test[index]
        return image, label
    except Exception as e:
        print(f"Erro ao carregar dados do MNIST: {e}")
        return None, None

def send_image_for_inference(url: str, image_data: np.ndarray):
    if image_data is None:
        return None

    pixels = image_data.flatten().tolist()
    payload = {"pixels": pixels}

    headers = {
        'Content-Type': 'application/json'
    }

    try:
        print(f"Enviando requisição para {url}...")
        response = requests.post(url, data=json.dumps(payload), headers=headers, timeout=15)
        response.raise_for_status() 
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Erro na comunicação com o ESP32: {e}")
        return None

def display_results(image: np.ndarray, true_label: int, prediction_data: dict):
    if image is None or prediction_data is None:
        print("Não foi possível processar os resultados.")
        return

    predicted_digit = prediction_data.get('predicted_digit', 'N/A')
    confidence = prediction_data.get('confidence', 0.0)
    
    print("\n--- Resultados ---")
    print(f"Índice da Imagem: {IMAGE_INDEX}")
    print(f"Rótulo Verdadeiro: {true_label}")
    print(f"Predição do ESP32: {predicted_digit}")
    print(f"Confiança: {confidence:.4f}")
    print("------------------\n")
    print("Resposta completa da API:")
    print(json.dumps(prediction_data, indent=2))

    plt.imshow(image, cmap='gray')
    plt.title(f"Enviado para ESP32\nRótulo Real: {true_label} | Predição: {predicted_digit}")
    plt.axis('off')
    
    output_filename = 'inference_result.png'
    plt.savefig(output_filename)
    print(f"\nGráfico salvo como '{output_filename}'.")
    
    plt.show()


if __name__ == "__main__":
    sample_image, true_label = get_mnist_image(IMAGE_INDEX)
    
    if sample_image is not None:
        prediction_result = send_image_for_inference(API_URL, sample_image)
        display_results(sample_image, true_label, prediction_result)