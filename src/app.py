"""FastAPI backend for the speech-enhancement dashboard.

Replaces the old Gradio UI. The model is loaded once at startup and reused for
every request. The HTTP layer is thin: all inference, plotting and dataset
metadata live in engine.py.

Routes
    GET  /                 dashboard page
    GET  /static/*         css + js
    GET  /api/health       model / device status
    GET  /api/overview     scores, training config, charts, best/worst files
    GET  /api/dataset      dataset stats and sample clean/noisy pairs
    GET  /api/demo         clips found in test_noisy/
    GET  /api/audio        stream an audio file (whitelisted sets only)
    GET  /api/spec         PNG spectrogram for a dataset clip
    POST /api/enhance      raw audio body -> cleaned audio + plots (data URIs)
    POST /api/live         same as enhance, semantic alias for the mic tab
"""

import argparse
import asyncio
import os
import sys

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

HERE = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(HERE, "web")

sys.path.insert(0, HERE)
from engine import Engine  # noqa: E402

app = FastAPI(title="Speech Enhancement", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=os.path.join(WEB, "static")), name="static")


def engine(request: Request) -> Engine:
    return request.app.state.engine


@app.get("/")
def index():
    return FileResponse(os.path.join(WEB, "templates", "index.html"))


@app.get("/api/health")
def health(request: Request):
    return JSONResponse(engine(request).model_info())


@app.get("/api/overview")
def overview(request: Request):
    return JSONResponse(engine(request).overview())


@app.get("/api/dataset")
def dataset(request: Request):
    return JSONResponse(engine(request).dataset())


@app.get("/api/demo")
def demo(request: Request):
    return JSONResponse({"files": engine(request).demo_files()})


@app.get("/api/audio")
def audio(request: Request, set: str, name: str):
    try:
        path = engine(request).resolve_audio(set, name)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    media = "audio/wav" if name.lower().endswith(".wav") else "application/octet-stream"
    return FileResponse(path, media_type=media, filename=name)


@app.get("/api/spec")
def spec(request: Request, set: str, name: str):
    try:
        png = engine(request).spectrogram_for(set, name)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return Response(content=png, media_type="image/png",
                    headers={"Cache-Control": "public, max-age=3600"})


async def _enhance_body(request: Request):
    data = await request.body()
    if not data:
        raise HTTPException(status_code=400, detail="empty request body")
    loop = asyncio.get_running_loop()
    try:
        return await loop.run_in_executor(None, engine(request).enhance_bytes, data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:  # keep the dashboard alive on unexpected input
        raise HTTPException(status_code=500, detail=f"enhancement failed: {exc}")


@app.post("/api/enhance")
async def enhance(request: Request):
    return JSONResponse(await _enhance_body(request))


@app.post("/api/live")
async def live(request: Request):
    return JSONResponse(await _enhance_body(request))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=os.path.join(HERE, "..", "out", "checkpoints", "best.pt"))
    p.add_argument("--data", default=os.path.join(HERE, "..", "data"))
    p.add_argument("--out", default=os.path.join(HERE, "..", "out"))
    p.add_argument("--test_dir", default=os.path.join(HERE, "..", "test_noisy"))
    p.add_argument("--chunk_seconds", type=float, default=4.0)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=7860)
    args = p.parse_args()

    print(f"[app] loading checkpoint {args.checkpoint}")
    app.state.engine = Engine(
        checkpoint=args.checkpoint,
        data_dir=args.data,
        out_dir=args.out,
        test_dir=args.test_dir,
        chunk_seconds=args.chunk_seconds,
    )
    info = app.state.engine.model_info()
    print(f"[app] model on {info['device']} ({info['gpu']}), "
          f"{info['params_m']:.2f} M params")
    print(f"[app] open http://localhost:{args.port}")

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
