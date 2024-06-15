import json
import time
from translate import Translator
from tqdm import tqdm

# Inicializa el traductor
translator = Translator(to_lang="es")

# Rutas del archivo JSON original y del archivo traducido
input_file_path = 'D:\\Alzhivida\\nuevo\\archive\\intents.json'
output_file_path = 'D:\\Alzhivida\\nuevo\\archive\\intents_translated.json'

# Función para traducir el texto con manejo de errores
def translate_text(text):
    try:
        translated = translator.translate(text)
        return translated
    except Exception as e:
        print(f"Error translating text: {text}. Error: {e}")
        return text

# Leer el archivo JSON original
with open(input_file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

# Verificar si la estructura es la esperada
if "intents" in data and isinstance(data["intents"], list):
    translated_intents = []

    # Calcula el total de elementos a traducir para la barra de progreso
    total_items = sum(len(intent["patterns"]) + len(intent["responses"]) for intent in data["intents"])

    # Barra de progreso
    with tqdm(total=total_items, desc="Translating", unit="item") as pbar:
        # Iterar sobre cada intención en la lista
        for intent in data["intents"]:
            if isinstance(intent, dict):
                # Traducir patrones
                translated_patterns = []
                for pattern in intent["patterns"]:
                    translated_pattern = translate_text(pattern)
                    translated_patterns.append(translated_pattern)
                    pbar.update(1)  # Actualiza la barra de progreso
                    time.sleep(0.1)  # Pausa para evitar límites de la API

                # Traducir respuestas
                translated_responses = []
                for response in intent["responses"]:
                    translated_response = translate_text(response)
                    translated_responses.append(translated_response)
                    pbar.update(1)  # Actualiza la barra de progreso
                    time.sleep(0.1)  # Pausa para evitar límites de la API

                # Crear nuevo diccionario traducido
                translated_intent = {
                    "tag": intent["tag"],
                    "patterns": translated_patterns,
                    "responses": translated_responses
                }
                translated_intents.append(translated_intent)
            else:
                print(f"Unexpected item type in 'intents': {type(intent)}. Skipping...")

    # Crear nuevo diccionario con 'intents' traducido
    translated_data = {"intents": translated_intents}

    # Guardar el JSON traducido a un archivo nuevo
    with open(output_file_path, 'w', encoding='utf-8') as file:
        json.dump(translated_data, file, ensure_ascii=False, indent=4)

    print(f"Translation completed and saved to '{output_file_path}'")
else:
    print(f"Unexpected data structure: {type(data)}. Expected a dict with 'intents' key containing a list.")
