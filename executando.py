import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Carregando Modelo
model = load_model('modelo_de_animal.h5')

img_width, img_height = 150, 150 # Tamanho das imagens
batch_size = 32 #Tamanho do lote

# Processando Modelo
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(img_width, img_height))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalização dos pixels para o intervalo [0, 1]
    return img

# Carregando e imagem
img_path = 'cachorro_teste.jpg'  # Substitua pelo caminho da sua imagem
img = preprocess_image(img_path)

# Definindo as etiquetas das classes
class_labels = ['cachorro', 'cavalo', 'gato', 'passaro']

# Previsão
predictions = model.predict(img)

# Obtendo previção e probabilidade
predicted_class = np.argmax(predictions)
probability = predictions[0][predicted_class]

# Mapeando a classe prevista de volta para o rótulo real
predicted_label = class_labels[predicted_class]

print("Animal identificado: ", predicted_label)
print("Probabilidade de: ", probability)