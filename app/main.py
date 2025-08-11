from fastapi import FastAPI

app = FastAPI(title="Deribit Option Flows API")


@app.get("/health")
def health():
    return {"status": "ok"}


