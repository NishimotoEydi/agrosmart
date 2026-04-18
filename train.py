import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import os

IMG_WIDTH = 128
IMG_HEIGHT = 128
BATCH_SIZE = 16
EPOCHS = 10

BASE_DIR = "data/dataset"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
VALID_DIR = os.path.join(BASE_DIR, "valid")

print(f"Buscando imagens de Treinamento em: {TRAIN_DIR}")
print(f"Buscando imagens de Validação em: {VALID_DIR}")

train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=20,
    zoom_range=0.15,
    horizontal_flip=True
)

valid_datagen = ImageDataGenerator(rescale=1.0/255.0)

print("\n[Carregando Imagens de Treino...]")
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

print("\n[Carregando Imagens de Validação...]")
valid_generator = valid_datagen.flow_from_directory(
    VALID_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

modelo = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid') 
])

modelo.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("\n==============================================")
print("INICIANDO O TREINAMENTO DA INTELIGÊNCIA ARTIFICIAL")
print("==============================================\n")

try:
    historico = modelo.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=valid_generator
    )

    print("\n✅ Treinamento finalizado com sucesso!")

    caminho_modelo = "models/modelo_agrosmart.h5"
    modelo.save(caminho_modelo)
    print(f"✅ Modelo salvo em: {caminho_modelo}")

except Exception as e:
    print("\n❌ Ocorreu um erro durante o treinamento!")
    print(f"Detalhes do erro: {str(e)}")
