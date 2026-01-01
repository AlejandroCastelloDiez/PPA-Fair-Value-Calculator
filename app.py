from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from io import StringIO
import csv


app = FastAPI(title="Capture Price API")

# Allow calls from your static site
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "API is running", "health": "/health", "docs": "/docs"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/capture-price")
async def capture_price(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "File must be .csv")

    content = (await file.read()).decode("utf-8", errors="replace")
    df = pd.read_csv(StringIO(content))

    required = {"month", "hour", "production_mw"}
    if not required.issubset(df.columns):
        raise HTTPException(
            400,
            f"CSV must contain columns {sorted(required)}"
        )

    total_prod = df["production_mw"].sum()
    if total_prod <= 0:
        raise HTTPException(400, "Total production must be > 0")

    # placeholder until prices are wired
    return {
        "rows_received": len(df),
        "total_production": float(total_prod),
        "capture_price_eur_mwh": None
    }

@app.post("/double")
async def double(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Upload a .csv file")

    content = (await file.read()).decode("utf-8", errors="replace")
    reader = csv.reader(StringIO(content))

    # Take the first numeric cell we find
    for row in reader:
        for cell in row:
            cell = cell.strip()
            if cell == "" or cell.lower() == "number":
                continue
            try:
                x = float(cell)
                return {"input": x, "result": x * 2}
            except ValueError:
                raise HTTPException(400, f"Expected a number in the first cell, got '{cell}'")

    raise HTTPException(400, "No number found in the CSV")
