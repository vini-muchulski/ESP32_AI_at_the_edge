import socket
import os
import io
from PIL import Image

# --- CONFIGURAÇÃO ---
ESP_IP = "192.168.0.111"  # << MUDE PARA O IP DO SEU ESP32
ESP_PORT = 3333
IMAGE_FILENAME_TO_SEND = "imagens/jesse_ww_crop.jpg"  # << MUDE PARA O NOME DO SEU ARQUIVO DE IMAGEM
TARGET_DIMENSIONS = (320, 240)
JPEG_QUALITY = 100
# --------------------

def process_and_prepare_image(filepath: str, target_size: tuple, quality: int) -> bytes | None:
    """
    Carrega uma imagem, a redimensiona, converte para JPEG em memória
    e retorna os bytes resultantes.
    """
    if not os.path.exists(filepath):
        print(f"Erro: Arquivo não encontrado em '{filepath}'")
        return None

    try:
        print(f"Processando imagem '{filepath}'...")
        with Image.open(filepath) as img:
            # Garante que a imagem esteja no modo RGB (necessário para salvar como JPEG)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Redimensiona a imagem para as dimensões alvo
            resized_img = img.resize(target_size, Image.Resampling.LANCZOS)
            print(f"Imagem redimensionada para {target_size[0]}x{target_size[1]}.")

            # Salva a imagem processada em um buffer de memória no formato JPEG
            buffer = io.BytesIO()
            resized_img.save(buffer, format="JPEG", quality=quality)
            
            # Obtém os bytes do buffer
            image_data = buffer.getvalue()
            print(f"Imagem convertida para JPEG. Tamanho final: {len(image_data)} bytes.")
            return image_data

    except Exception as e:
        print(f"Erro ao processar a imagem: {e}")
        return None

def send_data_to_esp(host: str, port: int, data: bytes):
    """
    Conecta-se a um servidor TCP, envia o conteúdo de um buffer de bytes
    e fecha a conexão.
    """
    print(f"Conectando a {host}:{port}...")
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(10)
            sock.connect((host, port))
            
            print(f"Enviando {len(data)} bytes...")
            sock.sendall(data)
            print("Envio concluído.")

    except socket.timeout:
        print("Erro: Timeout na conexão. Verifique o IP e se o ESP32 está na rede.")
    except ConnectionRefusedError:
        print("Erro: Conexão recusada. O servidor no ESP32 está rodando?")
    except Exception as e:
        print(f"Ocorreu um erro de socket: {e}")
    finally:
        print("Conexão fechada.")

if __name__ == "__main__":
    # 1. Processa a imagem para garantir formato e dimensões
    jpeg_data = process_and_prepare_image(
        IMAGE_FILENAME_TO_SEND, 
        TARGET_DIMENSIONS, 
        JPEG_QUALITY
    )

    # 2. Se o processamento foi bem-sucedido, envia os dados
    if jpeg_data:
        send_data_to_esp(ESP_IP, ESP_PORT, jpeg_data)
    else:
        print("Envio cancelado devido a erro no processamento da imagem.")