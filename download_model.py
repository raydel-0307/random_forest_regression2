import json
import requests
import os
from dotenv import load_dotenv

def download_model(minio_url: str, model_minio_path: str, model_local_path: str, model_name: str, project_number: str):
    config_str= "config.json"
    with open(f"{project_number}/{config_str}", 'r', encoding='utf-8') as file:
        config = json.load(file)

    model_minio_path_with_model_name = f"{model_minio_path}/{model_name}"
    model_local_path_with_model_name = f"{model_local_path}/{model_name}"

    try:
        # Descargar el modelo
        response = requests.post(
            f"{minio_url}download_from_models/",  # Asegúrate de que esta URL esté completa
            data={'model_name': model_minio_path_with_model_name},
        )
        
        # Verificar el estatus de la respuesta
        if response.status_code != 200:
            raise ValueError(f"Error al descargar el modelo: {response.status_code} - {response.text}")
        
        request_content = response.content
        
        # Guardar el modelo en la carpeta models
        os.makedirs(model_local_path, exist_ok=True)
        with open(model_local_path_with_model_name, 'wb') as model_file:
            model_file.write(request_content)
        
        print(f"Modelo descargado en la ruta: {model_local_path_with_model_name}")
    except Exception as e:
        raise ValueError(f"No se pudo descargar el modelo: {e}")
    
    print("Modelo creado")
    print("Creando encoder")
    
    # Descargar columnas
    encoder_minio_path_with_column_name = f"random_forest_regression/encoder/{config['encoder_name']}"
    encoder_local_path_with_column_name = f"{project_number}/models/{config['encoder_name']}"
    
    try:
        # Descargar columnas
        response = requests.post(
            f"{minio_url}download_from_models/",  # Asegúrate de que esta URL esté completa
            data={'model_name': encoder_minio_path_with_column_name},
        )
        
        # Verificar el estatus de la respuesta
        if response.status_code != 200:
            raise ValueError(f"Error al descargar el encoder: {response.status_code} - {response.text}")
        
        request_content = response.content
        
        # Crear el directorio de destino si no existe
        os.makedirs(os.path.dirname(encoder_local_path_with_column_name), exist_ok=True)
        
        # Guardar las columnas en la carpeta columns
        with open(encoder_local_path_with_column_name, 'wb') as encoder_file:
            encoder_file.write(request_content)
        
        print(f"encoder descargado en la ruta: {encoder_local_path_with_column_name}")
    except Exception as e:
        raise ValueError(f"No se pudo descargar el encoder: {e}")

if __name__ == "__main__":
    config_str = "config.json"
    with open(config_str, 'r', encoding='utf-8') as file:
        config = json.load(file)
    
    # Ruta del proyecto
    ruta = config["proyect"]
    
    with open(f"{ruta}/{config_str}", 'r', encoding='utf-8') as file:
        config = json.load(file)
    
    load_dotenv()
    
    # Obtener la URL desde el archivo .env
    minio_url = os.getenv('MINIO_URL')
    if not minio_url:
        raise ValueError("La variable de entorno MINIO_URL no está definida")
    
    model_minio_path = "random_forest_regression/models"
    model_local_path = f"{ruta}/models"
    model_name =  f'{config["name_model"]}'

    

    download_model(minio_url, model_minio_path, model_local_path, model_name, ruta)
