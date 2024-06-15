import json
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import seaborn as sns
import random

# Descargar recursos de NLTK (ejecutar solo una vez)
nltk.download('punkt')
nltk.download('wordnet')

# Cargar datos desde el archivo JSON
with open('archive/intentos.json', 'r', encoding='utf-8') as file:
    intents = json.load(file)

# Preparar datos
words = []
classes = []
documents = []
ignore_words = ['?', '!']
lemmatizer = WordNetLemmatizer()

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

classes = sorted(list(set(classes)))

print(f"Palabras únicas lematizadas: {len(words)}")
print(f"Clases únicas: {len(classes)}")

training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    word_patterns = doc[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]

    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

random.shuffle(training)

train_x = np.array([i[0] for i in training])
train_y = np.array([i[1] for i in training])

print("Datos de entrenamiento creados.")

# Construir el modelo mejorado
model = Sequential()
model.add(Dense(512, input_dim=len(train_x[0]), activation='relu'))  # Aumentar a 512 neuronas
model.add(Dropout(0.6))  # Aumentar el dropout a 0.6
model.add(Dense(256, activation='relu'))  # Añadir una capa oculta con 256 neuronas
model.add(Dropout(0.6))  # Aumentar el dropout a 0.6
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compilar el modelo con el optimizador Adam y métricas adicionales
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

# Entrenar el modelo y capturar historial de entrenamiento
history = model.fit(train_x, train_y, epochs=800, batch_size=16, verbose=1)

# Guardar el modelo
model.save('chatbot_model.h5')
print("Modelo guardado como 'chatbot_model.h5'.")

# Predecir las clases para los datos de entrenamiento
predictions = model.predict(train_x)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(train_y, axis=1)

# Reporte de clasificación
print("Reporte de Clasificación:")
print(classification_report(true_classes, predicted_classes, target_names=classes))

# Obtener precisión y pérdida del historial de entrenamiento
accuracy = history.history['accuracy']
loss = history.history['loss']

# Gráfico de Precisión
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(accuracy) + 1), accuracy, color='blue', label='Training Accuracy')
plt.title('Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# Gráfico de Pérdida
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(loss) + 1), loss, color='red', label='Training Loss')
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# Matriz de Confusión
conf_matrix = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Matriz de Confusión')
plt.xlabel('Clases Predichas')
plt.ylabel('Clases Verdaderas')
plt.show()

# Curva ROC y AUC
fpr, tpr, thresholds = roc_curve(true_classes, predictions[:, 1])
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos (FPR)')
plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
plt.title('Curva ROC')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

# Ejemplos de Predicciones Erróneas
incorrect_predictions = np.where(predicted_classes != true_classes)[0]
print(f'\nEjemplos de predicciones incorrectas (solo los primeros 5 ejemplos):')
for i in incorrect_predictions[:5]:
    print(f'Predicción incorrecta: {classes[predicted_classes[i]]}, Clase verdadera: {classes[true_classes[i]]}')
