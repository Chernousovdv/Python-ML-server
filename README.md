# Python ML server

This repository contains a FastAPI-based web service for managing and serving machine learning models. The API allows you to train, predict, load, unload, and manage models with concurrent processing support.

## Structure

It consists of 2 folders:

    - Server
    - Client

Server contains all the code, however it is easier to run from docker image.

Client contains a jupyter notebook demonstrating the features of the server

## Installation

Use the oficial  [dockerhub](https://hub.docker.com/_/registry) registry to setup a server.

```bash
docker pull danilach/python_ml_server:0.1.0
```

```bash
docker run -p 5123:5123 danilach/python_ml_server:0.1.0
```

Now the server is running.

Or you can build your own image from the dockerfile
## Usage


    - `fit(X, y, config)` - обучить модель и сохранить на диск по указанным именем
    - `predict(y, config)` - предсказать с помощью обученной и загруженной модели по её имени
    - `load(config)` - загрузить обученную модель по её имени в режим инференса
    - `unload(config)` - выгрузить загруженную модель по её имени
    - `remove(config)` - удалить обученную модель с диска по её имени
    - `remove_all()` - удалить все обученные модели с диска
