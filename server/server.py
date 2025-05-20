from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
import time
import os
import uvicorn
from typing import Any
from pydantic import BaseModel
import pickle
import gc
from classifier import Classifier
from datetime import datetime
from multiprocessing import Process, Lock
import threading


app = FastAPI()
loaded_models = {}  # dict of the models that are currently loaded
n_proc = 1
lock = Lock()
procs = []


# load config variables from .env file
load_dotenv()
models_path = os.getenv("models_path")
PORT = os.getenv("port")

try:
    n_jobs = int(os.getenv("n_jobs"))
    assert n_jobs >= 1
except (TypeError, ValueError, AssertionError):
    raise ValueError("Environment variable 'n_jobs' must be an integer >= 1")

try:
    n_models = int(os.getenv("n_models"))
    assert n_models >= 1
except (TypeError, ValueError, AssertionError):
    raise ValueError("Environment variable 'n_models' must be an integer >= 1")


class FitArgs(BaseModel):
    x: Any
    y: Any
    model_name: str
    model_type: str
    params: dict


class PredArgs(BaseModel):
    x: Any
    model_name: str


class LoadArgs(BaseModel):
    model_name: str


def func(number):
    result = number**2
    print(f"func was called by process id {os.getpid()} with number {number}")


def fit_and_save(args: FitArgs):
    classifier = Classifier(args.model_name, args.model_type, **args.params)
    classifier.fit(args.x, args.y)
    classifier.save(models_path)
    print(f"Process {os.getpid()} finished fitting {args.model_name}")


# web pages
@app.post("/fit")
def fit(args: FitArgs):
    global n_proc
    global procs
    try:
        with lock:
            if n_proc >= n_jobs:
                print(
                    f"Cannot create another process, n_jobs = {n_jobs} are already running"
                )
                raise Exception(
                    f"Cannot create another process, n_jobs = {n_jobs} are already running"
                )
            n_proc += 1
        proc = Process(target=fit_and_save, args=(args,))
        proc.start()
        proc.join()
        with lock:
            n_proc -= 1

            print(f"joined {len(procs)}")
        return {"message": f"{args.model_name} fitted successfully"}

    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error))


@app.post("/predict")
def predict(args: PredArgs):
    try:
        with lock:  # lock for all loaded models
            if args.model_name in loaded_models:
                return {
                    "prediction": loaded_models[args.model_name]
                    .predict(args.x)
                    .tolist()
                }
            else:
                raise Exception(f"Model {args.model_name} is not loaded")

    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error))


@app.post("/load")
def load(args: LoadArgs):
    try:
        with lock:
            if len(loaded_models) >= n_models and args.model_name not in loaded_models:
                raise Exception(
                    f"Cannot load another model. n_models = {n_models} limit is reached"
                )
            loaded_models[args.model_name] = Classifier.load(
                os.path.join(models_path, args.model_name + ".pkl")
            )

        return {"message": f"Model '{args.model_name}' loaded successfully"}

    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error))


@app.post("/unload")
def unload(args: LoadArgs):
    try:
        with lock:
            if args.model_name in loaded_models:
                model = loaded_models.pop(args.model_name)
                del model
                gc.collect()
                return {"message": f"Model '{args.model_name}' has been unloaded"}
            else:
                return {
                    "message": f"Model '{args.model_name}' has already been unloaded"
                }

    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error))


@app.post("/remove")
def remove(args: LoadArgs):
    try:
        with lock:
            file_path = os.path.join(models_path, args.model_name + ".pkl")
            os.remove(file_path)
            return {"message": f"Model file '{args.model_name}' removed successfully"}

    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error))


@app.post("/remove_all")
def remove_all():
    try:
        with lock:
            for filename in os.listdir(models_path):
                file_path = os.path.join(models_path, filename)
                if filename.endswith(".pkl"):
                    os.remove(file_path)
        return {"message": f"folder {models_path} is cleared successfully"}
    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error))


@app.post("/list_saved_models")
def list_saved_models():
    try:
        models = []
        for filename in os.listdir(models_path):
            models.append(filename)

        return {"message": models}

    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error))


@app.post("/list_loaded_models")
def list_loaded_models():
    try:
        models = []
        for model in loaded_models.keys():
            models.append(model)

        return {"message": models}

    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error))


@app.post("/processes")
def processes():
    try:
        with lock:
            return {"n_proc": n_proc}
    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error))


if __name__ == "__main__":
    print("Running")
    start_time = datetime.now()
    uvicorn.run("server:app", host="0.0.0.0", port=PORT, reload=False, workers=1)
