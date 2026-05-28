import mlflow
import optuna

from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Classic ML")

X, y = fetch_california_housing(return_X_y=True)
X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=8)

def objective(trial):
    with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}") as child_run:
        params = {
            "max_depth": trial.suggest_int("rf_max_depth", 2, 32),
            "n_estimators": trial.suggest_int("rf_n_estimators", 50, 300, step=10),
            "max_features": trial.suggest_float("rf_max_features", 0.2, 1.0)
        }

        mlflow.log_params(params)

        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)

        error = mean_squared_error(y_val, model.predict(X_val))
        mlflow.log_metrics({"error": error})

        mlflow.sklearn.log_model(model, name="model")
        trial.set_user_attr("run_id", child_run.info.run_id)

        return error
    
with mlflow.start_run(run_name="study"):
    n_trials = 30
    mlflow.log_param("n_trials", n_trials)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    mlflow.log_params(study.best_trial.params)
    mlflow.log_metrics({"best_error": study.best_value})

    if best_run_id := study.best_trial.user_attrs.get("run_id"):
        mlflow.log_param("best_child_run_id", best_run_id)