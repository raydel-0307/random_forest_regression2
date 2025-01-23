import json
import iamodels_main
import requests
import pickle
from dotenv import load_dotenv
import os

def fuctions_execute(config: dict, data: dict, project_number: str):
    # No es necesario hacer json.loads(config) ni json.loads(data), porque ya son diccionarios
    prompt_string = json.dumps(config["for_prediction"], ensure_ascii=False, indent=4)
    
    # Usar los valores del archivo JSON
    with open(f'{project_number}/models_saved/{config["name_model"]}', "rb") as f:
        model = pickle.load(f)
    with open(f'{project_number}/models_saved/{config["column_model"]}', "rb") as f:
        column = pickle.load(f)

    # Llamar al modelo y mostrar los resultados
    result = iamodels_main.random_forest_regression(model, column=column, prompt_string=prompt_string)
    print("Resultado: ")
    print(result)
    return result

def main():
    config_json_path = "config.json"

    # Leer el archivo de configuración
    with open(config_json_path, 'r', encoding='utf-8') as file:
        config_main = json.load(file)

    # project_number del proyecto
    project_number = config_main["proyect"]

    # leer config
    with open(f"{project_number}/{config_json_path}", 'r', encoding='utf-8') as file:
        config = json.load(file)

    # leer datos 
    json_file = config["json_file"]
    with open(f'{project_number}/{json_file}', 'r', encoding='utf-8') as file:
        data = json.load(file)

    load_dotenv()

    # Obtener la URL desde el archivo .env
    minio_url = os.getenv('MINIO_URL')
    if not minio_url:
        raise ValueError("La variable de entorno MINIO_URL no está definida")
    
    model_path = f'{project_number}/models_saved/{config["name_model"]}'
    
    # Verificar si el modelo existe en la ruta especificada
    if not os.path.exists(model_path):
        print("Descargando el modelo...")

        try:
            from download_model import download_model

            model_minio_path = "random_forest_regression/models"
            model_local_path = f"{project_number}/models_saved"
            model_name = f'{config["name_model"]}'

            column_minio_path = "random_forest_regression/columns"
            column_local_path = f"{project_number}/models_saved"
            column_name = f'{config["column_model"]}'

            # Ajuste en la llamada a download_model
            download_model(minio_url, model_minio_path, model_local_path, model_name, column_minio_path, column_local_path, column_name)
        except Exception as e:
            raise ValueError(f"No se pudo descargar el modelo: {e}")

    # Aquí, pasamos config y data ya como diccionarios
    result = fuctions_execute(config, data, project_number)

if __name__ == "__main__":
    main()