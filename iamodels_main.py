import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from joblib import dump, load
import json
import os
from pandas import DataFrame, get_dummies
from metrics import make_metrics, get_time
import time


def random_forest_regression(model_path: str, column_path: str, prompt_string: str):
    prompt: dict = json.loads(prompt_string)
    model = load(model_path)
    columns = load(column_path)
    prompt_dataframe = get_dummies(DataFrame([prompt])).reindex(columns=columns, fill_value=0)
    predict = model.predict(prompt_dataframe)
    prompt['output'] = str(predict[0])
    prompt_string_result = json.dumps(prompt, ensure_ascii=False, indent=4)
    return prompt_string_result