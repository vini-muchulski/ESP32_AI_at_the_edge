import socket
import os
import io
import json
from PIL import Image, ImageDraw

# --- CONFIGURAÇÃO ---
ESP_IP = "192.168.0.111"
ESP_PORT = 3333
#IMAGE_FILENAME_TO_SEND = "imagens/human_face_0.jpg"
#IMAGE_FILENAME_TO_SEND = "imagens/jesse_ww_crop_resized.jpg"
#IMAGE_FILENAME_TO_SEND = "imagens/jesse_ww_crop.jpg"
IMAGE_FILENAME_TO_SEND = "imagens/human_group_photo.jpg"

TARGET_DIMENSIONS = (320, 240)
JPEG_QUALITY = 100
OUTPUT_IMAGE_DIR = "imagens/results_exemplos"
# --------------------

def process_and_prepare_image(filepath: str, target_size: tuple, quality: int) -> bytes | None:
    if not os.path.exists(filepath):
        print(f"Erro: Arquivo não encontrado em '{filepath}'")
        return None
    try:
        print(f"Processando imagem '{filepath}'...")
        with Image.open(filepath) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            resized_img = img.resize(target_size, Image.Resampling.LANCZOS)
            print(f"Imagem redimensionada para {target_size[0]}x{target_size[1]}.")
            buffer = io.BytesIO()
            resized_img.save(buffer, format="JPEG", quality=quality)
            image_data = buffer.getvalue()
            print(f"Imagem convertida para JPEG. Tamanho final: {len(image_data)} bytes.")
            return image_data
    except Exception as e:
        print(f"Erro ao processar a imagem: {e}")
        return None

def send_and_receive(host: str, port: int, data: bytes) -> str | None:
    print(f"Conectando a {host}:{port}...")
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(30)
            sock.connect((host, port))
            
            print(f"Enviando {len(data)} bytes...")
            sock.sendall(data)
            sock.shutdown(socket.SHUT_WR) # Informa que não enviará mais dados
            print("Envio concluído. Aguardando resposta...")

            response_parts = []
            while True:
                chunk = sock.recv(1024)
                if not chunk:
                    break
                response_parts.append(chunk)
            
            response = b''.join(response_parts).decode('utf-8')
            print(f"Resposta recebida: {response}")
            return response

    except socket.timeout:
        print("Erro: Timeout na conexão ou na resposta.")
    except ConnectionRefusedError:
        print("Erro: Conexão recusada.")
    except Exception as e:
        print(f"Ocorreu um erro de socket: {e}")
    finally:
        print("Conexão fechada.")
    return None

def plot_results_on_image(filepath: str, target_size: tuple, results_json: str, output_dir: str):
    try:
        results = json.loads(results_json)
            
        with Image.open(filepath) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            resized_img = img.resize(target_size, Image.Resampling.LANCZOS)
            draw = ImageDraw.Draw(resized_img)

            if not results:
                print("Nenhum rosto detectado para plotar.")
            else:
                for face in results:
                    box = face.get("box")
                    score = face.get("score")
                    if box:
                        draw.rectangle(box, outline="red", width=2)
                        if score is not None:
                            draw.text((box[0], box[1] - 10), f"{score:.2f}", fill="red")
            
            os.makedirs(output_dir, exist_ok=True)
            base_filename = os.path.basename(filepath)
            output_filename = f"result_{base_filename}"
            output_filepath = os.path.join(output_dir, output_filename)
            
            resized_img.save(output_filepath)
            print(f"Imagem com resultados salva em: '{output_filepath}'")
            resized_img.show()
            
    except json.JSONDecodeError:
        print("Erro: A resposta recebida não é um JSON válido.")
    except Exception as e:
        print(f"Erro ao plotar e salvar resultados: {e}")


if __name__ == "__main__":
    jpeg_data = process_and_prepare_image(
        IMAGE_FILENAME_TO_SEND, 
        TARGET_DIMENSIONS, 
        JPEG_QUALITY
    )

    if jpeg_data:
        results_str = send_and_receive(ESP_IP, ESP_PORT, jpeg_data)
        if results_str:
            plot_results_on_image(
                IMAGE_FILENAME_TO_SEND, 
                TARGET_DIMENSIONS, 
                results_str,
                OUTPUT_IMAGE_DIR
            )
    else:
        print("Envio cancelado devido a erro no processamento da imagem.")