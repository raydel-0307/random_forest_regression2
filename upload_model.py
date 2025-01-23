import json
import os
from dotenv import load_dotenv
import requests

def upload_model(model_name,model_path,timeout=30):
    load_dotenv()

    minio_url = os.getenv('MINIO_URL')
    if not minio_url:
        raise ValueError("La variable de entorno MINIO_URL no está definida")

    if not model_path:
        print("No se ingresó ninguna ruta de archivo.")
        exit()
    with open(model_path, 'rb') as file:
        file_data = file.read()
    url = f'http://{minio_url}/api/minio/upload_to_models/'
    try:
        files = {'file': (model_path, file_data, 'application/octet-stream')}
        response = requests.post(url,files=files,data={'object_name': f"{model_name}/models/model.pkl"},timeout=timeout)
        if response.status_code == 200:
            print("Archivo subido exitosamente")
            print(response.json())
            os.unlink(model_path)
        else:
            print(f"Error: {response.status_code}")
            print(response.json())

    except Exception as e:
        print(f"Fallo en la solicitud: {str(e)}")
    except Exception as e:
        print(f"Error: {str(e)}")

def fuctions_execute(config_json_path: str):
    # Leer el archivo de configuración
    with open(config_json_path, 'r', encoding='utf-8') as file:
        config = json.load(file)

    #Ruta del proyecto
    ruta = config["proyect"]
    with open(f"{ruta}/{config_json_path}", 'r', encoding='utf-8') as file:
        config = json.load(file)
        
    # Usar los valores del archivo JSON
    upload_data = [
        {
            "name":config["name_model"],
            "path":f"{ruta}/models_saved/{config['name_model']}"
        },
        {
            "name":config["column_model"],
            "path":f"{ruta}/models_saved/{config['column_model']}"
        },
    ]

    for i in upload_data:
        for name,path in i.items():
            print("Subiendo Modelo")
            upload_model(name,path,timeout=300000)

def main():

    config_json_path = "config.json"

    result = fuctions_execute(config_json_path)

if __name__ == "__main__":
    main()
