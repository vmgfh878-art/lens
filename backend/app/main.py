from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import prices, predict

app = FastAPI(title="Lens API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 배포 시 프론트엔드 도메인으로 교체
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(prices.router, prefix="/prices", tags=["prices"])
app.include_router(predict.router, prefix="/predict", tags=["predict"])


@app.get("/")
def health_check():
    return {"status": "ok", "message": "Lens API is running"}
