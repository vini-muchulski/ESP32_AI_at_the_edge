from PIL import Image

# Abre a imagem
IMAGE_FILENAME_TO_SEND = "imagens/jesse_ww_crop.jpg" 
img = Image.open(IMAGE_FILENAME_TO_SEND)


TARGET_WIDTH = 320  # era 320
TARGET_HEIGHT = 240  # era 240

resized = img.resize((TARGET_WIDTH, TARGET_HEIGHT), Image.LANCZOS)

# Garante RGB (se tiver canal alfa)
rgb_img = resized.convert("RGB")

# Salva em JPEG com qualidade 60
rgb_img.save("imagens/jesse_ww_crop_resized.jpg", "JPEG", quality=60)
