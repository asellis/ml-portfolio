from fastapi import FastAPI, HTTPException
import os
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field

from transformers import pipeline

# Pydantic BaseModel lets us use json payload
class SentimentTextInput(BaseModel):
    text: str = Field(..., example="I loved this movie!")

class SentimentApp:
    """
    A class to manage interactions with the API for sentiment analysis.
    """
    # Directory where the model is stored.
    MODEL_DIR = os.path.abspath("../model")

    def __init__(self):
        self.app = app = FastAPI(
                lifespan=self.lifespan,
                title="Sentiment API"
            )
        self._add_routes()
        
    @asynccontextmanager
    async def lifespan(self, app: FastAPI):
        print("Starting")
        print(f"Looking for model in: {os.path.abspath(self.MODEL_DIR)}")

        if not os.path.exists(self.MODEL_DIR):
            raise Exception(f"Directory not found: {self.MODEL_DIR}")

        # Output files to make sure model is found
        print("Files found:")
        for f in os.listdir(self.MODEL_DIR):
            print(f"\t- {f}")

        try:
            print("Loading model + tokenizer")
            self.classifier = pipeline(
                "text-classification",
                model=self.MODEL_DIR,
                tokenizer=self.MODEL_DIR,
                device=-1, # CPU only
                return_all_scores=False,
            )
            print("Model loaded successfully!")

        except Exception as e:
            print("LIFESPAN FAILED:")
            print(e)
            raise e

        yield

        print("Shutting down")

    def _add_routes(self):
        @self.app.post("/predict", response_model=dict)
        async def predict(payload: SentimentTextInput):
            if not payload.text.strip():
                raise HTTPException(status_code=400, detail="Empty text")

            # pipeline returns List[Dict] (batches).  Take first one.
            result = self.classifier(payload.text)[0]

            return {
                "sentiment": result["label"], # POSITIVE / NEGATIVE
                "confidence": round(result["score"], 4),
            }
        
# Initiate the app
sentiment_app = SentimentApp()
app = sentiment_app.app