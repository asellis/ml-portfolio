# Overview

The IMDB dataset is trained in the IMDB.ipynb Jupyter notebook.  The model is then saved to a "model" folder which is used by a deployment application found within the "deployment" folder.

You can run the app locally by executing the following command within the "deployment" folder: `uvicorn main:app --reload`.
You can access a web UI using the hosted address with docs route, e.g. `http://127.0.0.1:8000/docs`.