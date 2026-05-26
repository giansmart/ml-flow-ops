import mlflow
import pandas as pan

model_uri = "models:/LogReg/1"

mlflow.set_tracking_uri("http://127.0.0.1:5000")
model = mlflow.pyfunc.load_model(model_uri)

data = pan.DataFrame(
    [
        [5.1, 3.5, 1.4, 0.2],
        [6.2, 3.4, 5.4, 2.3]
    ],
    columns=[
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)"
    ]
)

predictions = model.predict(data)

print(predictions)