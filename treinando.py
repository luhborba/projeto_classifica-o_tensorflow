from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras import layers, models


img_width, img_height = 150, 150 # Tamanho das imagens
batch_size = 20 #Tamanho do lote

#Criando geradores de dados para pré-processamneto das imagens
train_datagen = ImageDataGenerator(
    rescale = 1./255, # Normalização dos pixels para o intervalo [0,1]
    rotation_range = 40, # Rotação aleatrória das imagens
    width_shift_range = 0.3, # Deslocamento Hoprizontal aleatório
    height_shift_range = 0.3, # Deslocamento vertical aleatório
    shear_range = 0.2, # Cisalhamento aleatório
    zoom_range = 0.2, # Zoom aleatório
    horizontal_flip = True, # Espelho horizontal aleatório
    fill_mode = 'nearest' # Preenchimento de pixels
)

train_generator = train_datagen.flow_from_directory(
    'data',
    target_size = (img_width, img_height),
    batch_size = batch_size,
    class_mode = 'categorical'
)

# Criando modelo
model = models.Sequential()

# Adicionando camadas convulcionais
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(img_width, img_height, 3)))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(128, (3,3), activation='relu'))          
model.add(layers.MaxPooling2D(2,2))


# Adicionando camadas conectadas
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(4, activation='softmax'))


# Compilando
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Treinando Modelo
model.fit(
    train_generator,
    epochs=100,
    steps_per_epoch=len(train_generator)
)

# Salvando Modelo
model.save('modelo_de_animal.h5')