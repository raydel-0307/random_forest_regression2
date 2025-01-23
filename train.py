import json
import pandas as pd
import os
import iamodels_train
from dotenv import load_dotenv

def fuctions_execute(config_str: str):
    # Leer el archivo de configuración
    with open(config_str, 'r', encoding='utf-8') as file:
        config = json.load(file)

    # Ruta del proyecto
    ruta = config["proyect"]
    with open(f"{ruta}/{config_str}", 'r', encoding='utf-8') as file:
        config = json.load(file)

    # Usar los valores del archivo JSON
    with open(f'{ruta}/{config["json_file"]}', 'r', encoding='utf-8') as file:
        json_file = json.load(file)

    df_datos = pd.DataFrame(json_file)
    
    # Llamar al modelo y mostrar los resultados
    iamodels_train.train_rf(df_datos, config, ruta)
    
def main():
    config_str = "config.json"
    with open(config_str, 'r', encoding='utf-8') as file:
        config = json.load(file)

    # Ruta del proyecto
    ruta = config["proyect"]
    with open(f"{ruta}/{config_str}", 'r', encoding='utf-8') as file:
        config = json.load(file)

    model_path = f'models_saved/{config["name_model"]}'  # Ajuste: cambiar a models_saved
    column_path = f'{ruta}/models_saved/{config["column_model"]}'  # Ajuste: cambiar a models_saved
    
    # Crear la ruta si no existe
    if not os.path.exists("models_saved"):
        os.makedirs("models_saved")

    print("Modelo en proceso de generación")

    fuctions_execute(config_str)
    
    print("Modelo generado")
    
    # Subir modelo generado a MinIO
    from upload_model import upload_model
    
    # Cargar las variables de entorno desde el archivo .env
    load_dotenv()

    # Obtener la URL desde el archivo .env
    minio_url = os.getenv('MINIO_URL')
    if not minio_url:
        raise ValueError("La variable de entorno MINIO_URL no está definida")

    model_minio_path = "random_forest_regression/models"
    model_local_path = "models_saved"  # Ajuste: cambiar a models_saved
    model_name = f'{config["name_model"]}'
    column_minio_path = "random_forest_regression/columns"
    column_local_path = "models"  # Dejar como "models" ya que se refiere al archivo de columnas
    column_name = f'{config["column_model"]}'

    # Ajuste en upload_model: pasa los 6 parámetros correctos
    upload_model(
        minio_url,
        model_minio_path,
        model_local_path,
        model_name,
        column_name,  # solo pasa column_name, no ruta ni minio_path de columnas
        ruta  # la ruta del proyecto
    )

    print("Modelo y columnas subidos a MinIO")
    
    # Borrar modelo del local
    if os.path.exists(model_path):
        os.remove(model_path)
    if os.path.exists(column_path):
        os.remove(column_path)

    print("Modelo y columnas borrados del local")    

if __name__ == "__main__":
    main()
