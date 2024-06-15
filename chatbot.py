import json
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model # type: ignore
import random
import time  # Importamos la libreria time para simular la espera

# Descargar recursos de NLTK (ejecutar solo una vez)
nltk.download('punkt')
nltk.download('wordnet')

# Cargar el modelo del archivo chatbot_model.h5
model = load_model('chatbot_model.h5')

# Cargar datos de intents desde el archivo JSON
with open('archive/intentos.json', 'r', encoding='utf-8') as file:
    intents = json.load(file)

lemmatizer = WordNetLemmatizer()

# Preprocesamiento de texto
def preprocess_text(text):
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word.lower()) for word in tokens]
    return tokens

# Función para predecir la clase de la entrada del usuario
def predict_class(text):
    tokens = preprocess_text(text)
    bag = [0]*len(words)
    for w in tokens:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    
    result = model.predict(np.array([bag]))[0]
    threshold = 0.25
    results = [[i, r] for i, r in enumerate(result) if r > threshold]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

# Función para obtener una respuesta del chatbot
def get_response(intents_list, intents_json):
    if intents_list:
        tag = intents_list[0]['intent']
        list_of_intents = intents_json['intents']
        for i in list_of_intents:
            if i['tag'] == tag:
                result = random.choice(i['responses'])
                break
    else:
        # Simular espera y luego derivar al especialista
        print("Bot: Espera un momento, te estamos derivando a una especialista...")
        time.sleep(3)  # Simulamos una espera de 3 segundos
        result = "Bot: Te estamos derivando a una especialista dentro de unos minutos. Por favor, espera."
    return result

# Cargar palabras y clases
words = []
classes = []
documents = []
ignore_words = ['?', '!']
for intent in intents['intents']:
    for pattern in intent['patterns']:
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append((w, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

print("¡Bienvenido al Chatbot!")
print("Escribe 'salir' para finalizar.")

while True:
    user_input = input("Tú: ")
    if user_input.lower() == 'salir':
        break
    
    intents_list = predict_class(user_input)
    response = get_response(intents_list, intents)
    print(response)
