import cv2
from skimage.morphology import skeletonize
from skimage import img_as_ubyte
import fingerprint_feature_extractor as Finger
import fingerprint_enhancer as EnchanceFinger


# Carregue a imagem
image = cv2.imread('imgs/FingerprintR.jpeg', cv2.IMREAD_GRAYSCALE)  # Substitua 'fingerprint.jpg' pelo caminho para a sua imagem

# Converta a imagem para binário
_, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# Inverta a imagem, pois a esqueletização é normalmente realizada em fundo preto
# binary_image = cv2.bitwise_not(binary_image)

# melhora a imagem
binary_image = EnchanceFinger.enhance_Fingerprint(binary_image)

# Realize uma busca por Birfurcaçoes e Terminaçoes e exiba os pontos achados
FeaturesTerminations, FeaturesBifurcations = Finger.extract_minutiae_features(binary_image, spuriousMinutiaeThresh=10, invertImage=False, showResult=True, saveResult=True)

print('Birfucação: ', len(FeaturesBifurcations), 'Terminaçoes :', len(FeaturesTerminations))

# Mostre a imagem original e a imagem com cruzamentos
cv2.imshow('Original', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

