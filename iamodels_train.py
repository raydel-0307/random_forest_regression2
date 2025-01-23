import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from joblib import dump, load
import json
import os
from pandas import DataFrame, get_dummies
from metrics import make_metrics, get_time
import time


def train_rf(df_datos, config, dir_path):
    init_time = time.perf_counter()
    
    # Verificar que las columnas especificadas en 'features' existen en los datos
    features_existentes = [col for col in config['features'] if col in df_datos.columns]
    if not features_existentes:
        raise ValueError("Las columnas especificadas en 'features' no existen en el DataFrame.")
    if config['target'] not in df_datos.columns:
        raise ValueError("La columna objetivo especificada no existe en el DataFrame.")

    # Preparar los datos
    df_procesado = pd.get_dummies(df_datos[features_existentes], drop_first=True)
    X = df_procesado
    y = df_datos[config['target']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    print("Entrenando al Modelo")
    
    # Configuración de la búsqueda de hiperparámetros
    hyperparameter_grid = config['hyperparameter_grid']
    clf = RandomForestRegressor(random_state=config['model_params']['random_state'])
    grid_search = GridSearchCV(estimator=clf, param_grid=hyperparameter_grid, cv=3, n_jobs=-1, verbose=0, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    # Definir las rutas de guardado del modelo y las columnas
    current_script_dir = os.path.dirname(__file__)
    models_dir = os.path.join(current_script_dir, dir_path, 'models_saved')
    
    # Asegurarse de que la carpeta models_saved exista
    os.makedirs(models_dir, exist_ok=True)
    
    model_path = os.path.join(models_dir, config['name_model'])
    columns_path = os.path.join(models_dir, config['column_model'])

    # Guardar el modelo entrenado
    dump(grid_search.best_estimator_, model_path)

    # Guardar las columnas utilizadas en el modelo
    columnas_modelo = list(X.columns)  # Obtener las columnas del modelo procesado
    dump(columnas_modelo, columns_path)

    # Métricas de evaluación
    best_clf = grid_search.best_estimator_
    metrics = make_metrics(best_clf, X_test, y_test)
    print("Metrics", metrics)

    # Imprimir el tiempo total de ejecución
    get_time(init_time)
