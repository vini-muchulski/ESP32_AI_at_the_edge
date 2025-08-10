import cv2

# Caminho da imagem original
image_path = 'main/human_face.jpg'

# Carrega a imagem
img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError(f"Imagem não encontrada em: {image_path}")

# Coordenadas da detecção facial
score = 0.880797
x1, y1, x2, y2 = 99, 68, 192, 190

# Pontos faciais
left_eye = (117, 113)
right_eye = (156, 113)
nose = (131, 144)
left_mouth = (121, 158)
right_mouth = (153, 158)

# Desenha a bounding box do rosto
cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
# Opcional: exibe a pontuação
cv2.putText(img, f'{score:.2f}', (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# Desenha os landmarks como círculos
landmarks = [left_eye, right_eye, nose, left_mouth, right_mouth]
for (x, y) in landmarks:
    cv2.circle(img, (x, y), radius=3, color=(0, 0, 255), thickness=-1)

# Exibe resultado
cv2.imshow('Deteccao Facial', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Se quiser salvar a imagem resultante:
output_path = "main/human_face_save.jpg"
cv2.imwrite(output_path, img)
print(f'Imagem anotada salva em: {output_path}')
