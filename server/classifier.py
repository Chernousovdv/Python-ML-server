import pickle
import os
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor


class Classifier:
    def __init__(self, model_name, model_type, **params):
        self.model_name = model_name
        model_registry = {
            "LinearRegression": LinearRegression,
            "RandomForest": RandomForestRegressor,
            "CatBoost": CatBoostRegressor,
        }
        model_cls = model_registry.get(model_type)
        if model_cls is None:
            raise ValueError(f"Unsupported model type: {model_type}")
        self.model = model_cls(**params)

    def fit(self, x, y):
        self.model.fit(x, y)

    def predict(self, x):
        return self.model.predict(x)

    def save(self, models_path):
        path = os.path.join(models_path, self.model_name + ".pkl")
        os.makedirs(models_path, exist_ok=True)  # Recreate it if needed
        print(f"folder created at {models_path}")

        # check if model already exists
        if os.path.exists(path):
            print("File already exists, renaming it")
            i = 1
            while os.path.exists(  # Look for a free name
                os.path.join(models_path, f"{self.model_name}_{i}.pkl")
            ):
                i += 1

            with open(
                os.path.join(models_path, self.model_name + "_" + str(i) + ".pkl"), "wb"
            ) as f:
                pickle.dump(self, f)
        else:
            with open(path, "wb") as f:
                pickle.dump(self, f)

    @staticmethod
    def load(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
