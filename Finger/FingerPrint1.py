import cv2
import numpy as np

# Função para contar o número de cruzamentos em uma imagem
def count_crossings(image):
    # Use um kernel para encontrar cruzamentos
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.int16)
    crossings = cv2.filter2D(image, -1, kernel)
    
    # Conte o número de cruzamentos
    num_crossings = np.sum(crossings > 0)
    
    return crossings, num_crossings

# Carregue a imagem
image = cv2.imread('imgs/Fingerprint2R.jpeg', cv2.IMREAD_GRAYSCALE) 

# Converta a imagem para binário
_, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# Conte o número de cruzamentos de linha
crossings, num_crossings = count_crossings(binary_image)

# Crie uma cópia da imagem original em cores para desenhar os cruzamentos
color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

# Desenhe os cruzamentos em vermelho
color_image[crossings > 0] = [0, 0, 255]

# Mostre a imagem original e a imagem com cruzamentos
cv2.imshow('Original', image)
cv2.imshow('Crossings', color_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(f'O número de cruzamentos de linha é {num_crossings}')
