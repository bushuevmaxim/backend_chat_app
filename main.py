from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from transformers import pipeline

app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


model = pipeline("text-generation",
                 model="ai-forever/rugpt3small_based_on_gpt2")


@app.post("/generate")
def generate(message_from_user: str):
    res = model(message_from_user, max_length=32)
    message = res[0]['generated_text']
    return message
