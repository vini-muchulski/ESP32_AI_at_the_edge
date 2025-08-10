import re
import cv2

def plot_faces_from_log(log_str: str, img_path: str, output_path: str = None):
    """
    Extrai detecções faciais (bbox + score) de uma string de log e plota sobre a imagem.
    Landmarks são opcionais: se não presentes, apenas as caixas são desenhadas.
    """
    # Regex para todas as bbox
    bbox_re = re.compile(
        r'\[score:\s*([0-9]*\.?[0-9]+),\s*'
        r'x1:\s*(\d+),\s*y1:\s*(\d+),\s*'
        r'x2:\s*(\d+),\s*y2:\s*(\d+)\]'
    )
    # Regex para landmarks (opcional)
    lm_re = re.compile(
        r'left_eye:\s*\[(\d+),\s*(\d+)\],\s*'
        r'left_mouth:\s*\[(\d+),\s*(\d+)\],\s*'
        r'nose:\s*\[(\d+),\s*(\d+)\],\s*'
        r'right_eye:\s*\[(\d+),\s*(\d+)\],\s*'
        r'right_mouth:\s*\[(\d+),\s*(\d+)\]'
    )

    # Encontra todas as caixas
    bboxes = list(bbox_re.finditer(log_str))
    if not bboxes:
        raise ValueError("Nenhuma detecção de face encontrada no log.")

    # Carrega imagem
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Imagem não encontrada em: {img_path}")

    # Para cada detecção, desenha caixa, score e (opcionalmente) landmarks
    for bb in bboxes:
        score, x1, y1, x2, y2 = bb.groups()
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        score = float(score)

        # Desenha bounding box e score
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f'{score:.2f}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Se houver landmarks após essa bbox, desenha-os
        lm_match = lm_re.search(log_str, bb.end())
        if lm_match:
            lm_vals = list(map(int, lm_match.groups()))
            landmarks = [
                tuple(lm_vals[0:2]),   # left_eye
                tuple(lm_vals[2:4]),   # left_mouth
                tuple(lm_vals[4:6]),   # nose
                tuple(lm_vals[6:8]),   # right_eye
                tuple(lm_vals[8:10]),  # right_mouth
            ]
            for (x, y) in landmarks:
                cv2.circle(img, (x, y), 3, (0, 0, 255), -1)

    # Exibe resultado
    cv2.imshow('Deteccoes Faciais', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Salva, se desejado
    if output_path:
        cv2.imwrite(output_path, img)
        print(f'Imagem anotada salva em: {output_path}')


if __name__ == '__main__':
    # Exemplo de uso com o seu log que só tem bbox:
    log = """
I (136451) FACE_DETECT_WIFI: [score: 0.78, x1: 86, y1: 44, x2: 128, y2: 111]
I (136451) FACE_DETECT_WIFI: [score: 0.59, x1: 203, y1: 44, x2: 249, y2: 114]
    """
    plot_faces_from_log(log, 'imagens/jesse_ww_crop_resized.jpg', 'imagens/results_exemplos/resultado.jpg')
