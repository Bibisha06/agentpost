from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.agent.graph import generate_shitpost


app = FastAPI(title="AI Shitpost Generator")


class GenerateRequest(BaseModel):
    topic: str


@app.get("/health")
def health():
    return {
        "status": "ok"
    }


@app.post("/generate")
def generate(request: GenerateRequest):

    topic = request.topic.strip()

    if not topic:
        raise HTTPException(
            status_code=400,
            detail="Topic cannot be empty"
        )

    try:
        result = generate_shitpost(topic)

        return {
            "success": True,
            "topic": result["topic"],
            "post": result["draft"],
            "critique": result["critique"],
            "attempts": result["attempts"],
        }

    except Exception as e:

        print("Generation error:", e)

        raise HTTPException(
            status_code=500,
            detail="Failed to generate post"
        )