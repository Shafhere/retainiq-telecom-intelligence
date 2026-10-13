from fastapi import FastAPI
from app.database import Base, engine
from app.routers import predictions, customers, analytics, datasets

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="RetainIQ — Telecom Customer Intelligence",
    version="1.0.0",
    description="Churn prediction with SHAP explainability and bilingual retention offers.",
)


@app.get("/")
def root():
    return {"status": "ok", "service": "RetainIQ", "version": "1.0.0"}


app.include_router(datasets.router)
app.include_router(predictions.router)
app.include_router(customers.router)
app.include_router(analytics.router)