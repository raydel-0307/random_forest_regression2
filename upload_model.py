import os
import requests
import json
from dotenv import load_dotenv


def upload_model(minio_url: str, model_minio_path: str, model_local_path: str, model_name: str, column_name: str, project_number: str):
    # Construir rutas completas
    model_local_path_with_model_name = os.path.join(project_number, model_local_path, model_name)
    column_local_path_with_column_name = os.path.join(project_number, model_local_path, column_name)
    model_minio_path_with_model_name = f"{model_minio_path}/{model_name}"
    column_minio_path_with_column_name = f"{model_minio_path}/columns/{column_name}"

    # Leer el modelo
    try:
        with open(model_local_path_with_model_name, 'rb') as f:
            model = f.read()
    except FileNotFoundError:
        raise ValueError(f"El archivo del modelo no se encontró: {model_local_path_with_model_name}")

    # Subir el modelo al servidor MinIO
    try:
        response = requests.post(
            url=f"{minio_url}upload_to_models/",
            files={'file': ("model_rfr", model, 'application/zip')},
            data={'object_name': model_minio_path_with_model_name},
        )
        print("Modelo subido:", response.json())
    except Exception as e:
        raise ValueError(f"No se pudo subir el modelo: {e}")

    # Leer las columnas
    try:
        with open(column_local_path_with_column_name, 'rb') as f:
            column = f.read()
    except FileNotFoundError:
        raise ValueError(f"El archivo de las columnas no se encontró: {column_local_path_with_column_name}")

    # Subir las columnas al servidor MinIO
    try:
        response = requests.post(
            url=f"{minio_url}/api/minio/upload_to_models/",
            files={'file': ("column_rfr", column, 'application/zip')},
            data={'object_name': column_minio_path_with_column_name},
        )
        print("Columnas subidas:", response.json())
    except Exception as e:
        raise ValueError(f"No se pudo subir las columnas: {e}")


if __name__ == "__main__":
    # Cargar variables de entorno desde el archivo .env
    load_dotenv()

    # Leer la configuración desde config.json
    config_str = "config.json"
    with open(config_str, 'r', encoding='utf-8') as file:
        config = json.load(file)

    # Obtener variables desde el archivo .env y la configuración
    minio_url = os.getenv('MINIO_URL')
    if not minio_url:
        raise ValueError("La variable de entorno MINIO_URL no está definida")

    model_minio_path = "random_forest_regression/models"
    model_local_path = "models"
    model_name = config["name_model"]
    column_name = config.get("column_model", "default_column_name")
    project_number = config["proyect"]

    # Subir modelo y columnas
    upload_model(minio_url, model_minio_path, model_local_path, model_name, column_name, project_number)
