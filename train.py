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
      

if __name__ == "__main__":
    main()
