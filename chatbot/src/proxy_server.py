from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
import httpx
import yaml
import os

app = FastAPI()

# Load config
config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

MODEL_SERVER_BASE_URL = config["model_server_base_url"]
API_KEY = config["api_key"]

@app.api_route("/proxy/{path:path}", methods=["GET", "POST"])
async def proxy(path: str, request: Request):
    url = f"{MODEL_SERVER_BASE_URL}/{path}"
    headers = dict(request.headers)
    headers["Authorization"] = f"Bearer {API_KEY}"

    async with httpx.AsyncClient() as client:
        if request.method == "GET":
            params = dict(request.query_params)
            resp = await client.get(url, headers=headers, params=params)
        elif request.method == "POST":
            body = await request.body()
            resp = await client.post(url, headers=headers, content=body)
        else:
            return JSONResponse({"error": "Method not allowed"}, status_code=405)

    return Response(content=resp.content, status_code=resp.status_code, media_type=resp.headers.get("content-type", "application/json"))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
