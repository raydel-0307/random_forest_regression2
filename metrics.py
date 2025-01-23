from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time

def get_time(init_time):
    timer = time.perf_counter() - init_time
    print("Tiempo de ejecución:",timer,"seg")
    
def make_metrics(model, X_test, y_test):

    y_pred = model.predict(X_test)
    metrics = {
        "mean_squared_error": mean_squared_error(y_test, y_pred),
        "mean_absolute_error": mean_absolute_error(y_test, y_pred),
        "r2_score": r2_score(y_test, y_pred)
	}
    return metrics