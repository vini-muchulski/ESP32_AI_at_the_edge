import numpy as np
import tensorflow as tf 

#from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# 1. Carrega MNIST
(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


# 2. Seleciona a imagem e o rótulo
idx = 41



# 2. Recupera a imagem e o rótulo
img = x_test[idx].squeeze()      # squeeze() transforma (28,28,1) em (28,28)
label = y_test[idx]

# 3. Plota
plt.figure(figsize=(4,4))
plt.imshow(img, cmap='gray')
plt.title(f"Rótulo: {label}")
plt.axis('off')
plt.show()



img = x_test[idx]
label = y_test[idx]
print(f"Classe verdadeira: {label}")



# 3. Achata e imprime os pixels no formato C
flat = img.flatten().astype(int)
lines = []
for i in range(0, len(flat), 16):
    chunk = flat[i:i+16]
    line = ", ".join(f"{p}" for p in chunk)
    lines.append("  " + line)
print("uint8_t mnist_sample[28*28] = {")
print(",\n".join(lines))
print("};")
