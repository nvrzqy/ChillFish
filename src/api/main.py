import tempfile
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.config import PROJECT_ROOT
from src.tabular.dataset import load_available_table
from src.tabular.predict_lot import predict_feature_row, predict_lot


FRONTEND_DIR = PROJECT_ROOT / "frontend"

app = FastAPI(title="ChillFish AI MVP", version="0.1.0")
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/lots")
def list_lots(limit: int = 600):
    df = load_available_table()
    rows = df[["lot_id", "handling_scenario", "recommended_action", "condition_status"]].head(limit)
    return {
        "count": int(len(rows)),
        "lots": rows.to_dict(orient="records"),
    }


@app.get("/api/predict/{lot_id}")
def predict(lot_id: str):
    try:
        return predict_lot(lot_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def _demerits_from_proxy(score: float) -> dict:
    per_attribute = max(0.0, min(3.0, score / 5.0))
    return {
        "eye_demerit_0_3": per_attribute,
        "gill_demerit_0_3": per_attribute,
        "odor_demerit_0_3": per_attribute,
        "texture_demerit_0_3": per_attribute,
        "mucus_demerit_0_3": per_attribute,
        "belly_burst_flag": 1 if score >= 15 else 0,
    }


def _manual_row(
    lot_id: str,
    lot_mass_kg: float,
    handling_scenario: str,
    target_market_node_id: str,
    mean_temp_c: float,
    max_temp_c: float,
    time_above_4c_h: float,
    time_above_10c_h: float,
    time_above_15c_h: float,
    remaining_quality_window_h: float,
    visual_proxy_score_0_16: float,
) -> pd.DataFrame:
    visual_proxy_score_0_16 = max(0.0, min(16.0, visual_proxy_score_0_16))
    equivalent_ice_age_h = max(0.0, 168.0 - remaining_quality_window_h)
    row = {
        "lot_id": lot_id or "MANUAL-LOT",
        "species_common": "lemuru",
        "scientific_name": "Sardinella lemuru",
        "lot_mass_kg": lot_mass_kg,
        "product_form": "whole_fresh",
        "handling_scenario": handling_scenario,
        "prehistory_equivalent_ice_h": 0.0,
        "baseline_shelf_life_h": 168.0,
        "target_market_node_id": target_market_node_id,
        "traceability_status": "MANUAL_INPUT",
        "origin_node_id": "N001",
        "min_temp_c": min(mean_temp_c, max_temp_c),
        "mean_temp_c": mean_temp_c,
        "max_temp_c": max_temp_c,
        "time_above_4c_h": time_above_4c_h,
        "time_above_10c_h": time_above_10c_h,
        "time_above_15c_h": time_above_15c_h,
        "equivalent_ice_age_h": equivalent_ice_age_h,
        "remaining_quality_window_h": remaining_quality_window_h,
        "logger_missing_intervals": 0,
        "total_proxy_0_16": visual_proxy_score_0_16,
        "condition_status": "MANUAL_UNKNOWN",
        "risk_score_0_100": 0.0,
        "recommended_action": "MANUAL_UNKNOWN",
    }
    row.update(_demerits_from_proxy(visual_proxy_score_0_16))
    return pd.DataFrame([row])


async def _try_visual_inference(photo: UploadFile | None) -> dict | None:
    if photo is None or not photo.filename:
        return None

    suffix = Path(photo.filename).suffix or ".jpg"
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
            handle.write(await photo.read())
            temp_path = Path(handle.name)

        try:
            from src.visual.predict_visual import predict_visual

            result = predict_visual(temp_path)
            result["filename"] = photo.filename
            return result
        except Exception as exc:
            return {
                "filename": photo.filename,
                "status": "PHOTO_RECEIVED",
                "note": f"Visual AI skipped in this runtime: {exc}",
            }
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except Exception:
                pass


@app.post("/api/predict/manual")
async def predict_manual(
    lot_id: str = Form("MANUAL-LOT"),
    lot_mass_kg: float = Form(...),
    handling_scenario: str = Form("controlled_ice"),
    target_market_node_id: str = Form("N004"),
    mean_temp_c: float = Form(...),
    max_temp_c: float = Form(...),
    time_above_4c_h: float = Form(0.0),
    time_above_10c_h: float = Form(0.0),
    time_above_15c_h: float = Form(0.0),
    remaining_quality_window_h: float = Form(...),
    visual_proxy_score_0_16: float = Form(5.0),
    photo: UploadFile | None = File(None),
):
    row = _manual_row(
        lot_id,
        lot_mass_kg,
        handling_scenario,
        target_market_node_id,
        mean_temp_c,
        max_temp_c,
        time_above_4c_h,
        time_above_10c_h,
        time_above_15c_h,
        remaining_quality_window_h,
        visual_proxy_score_0_16,
    )
    result = predict_feature_row(row)
    result["input_mode"] = "manual"
    result["visual_upload"] = await _try_visual_inference(photo)
    result["claim_note"] = (
        "Manual input prediction uses the trained XGBoost MVP model. "
        "Decision-support only; validate thresholds and labels before operational use."
    )
    return result
