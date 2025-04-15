from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import os
import json
import httpx
from dotenv import load_dotenv
from typing import Dict, Any, Optional

# Load environment variables with explicit path
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# For debugging
api_key = os.getenv("OPENROUTER_API_KEY")
print(f"API Key loaded: {'Present' if api_key else 'Missing'}")

app = FastAPI(title="Health Analysis API")

# Get API key and model from environment variables
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "deepseek/deepseek-chat-v3-0324:free")

class HealthData(BaseModel):
    oxygen_saturation: float = Field(..., description="Oxygen Saturation Level (%)", ge=0, le=100)
    pulse_rate: int = Field(..., description="Pulse Rate (bpm)", ge=0, le=250)
    blood_pressure: str = Field(..., description="Blood Pressure (mm Hg)")
    body_temperature: float = Field(..., description="Body Temperature (°F)", ge=80, le=120)
    blood_sugar: int = Field(..., description="Blood Sugar (mg/dL)", ge=0, le=500)
    rest_urine: int = Field(..., description="Rest Urine (ml)", ge=0, le=5000)
    water_intake: float = Field(..., description="Water Intake (liters)", ge=0, le=10)

class AnalysisResponse(BaseModel):
    analysis: str
    recommendations: str
    alert_level: str
    abnormal_readings: Dict[str, Any]

@app.get("/")
def read_root():
    return {"message": "Welcome to the Health Analysis API"}

@app.post("/analyze/", response_model=AnalysisResponse)
async def analyze_health_data(health_data: HealthData):
    if not OPENROUTER_API_KEY:
        raise HTTPException(status_code=500, detail="OpenRouter API key not configured")
    
    # Normal ranges for health metrics
    normal_ranges = {
        "oxygen_saturation": (80, 100),
        "pulse_rate": (50, 100),
        "blood_pressure": "80-120",  # This is simplified and will need custom parsing
        "body_temperature": (95, 105),
        "blood_sugar": (70, 140),
        "rest_urine": (800, 2000),
        "water_intake": (2.7, 3.7)
    }
    
    # Check for abnormal readings
    abnormal_readings = {}
    
    # Check oxygen saturation
    if health_data.oxygen_saturation < normal_ranges["oxygen_saturation"][0] or health_data.oxygen_saturation > normal_ranges["oxygen_saturation"][1]:
        abnormal_readings["oxygen_saturation"] = {
            "value": health_data.oxygen_saturation,
            "normal_range": normal_ranges["oxygen_saturation"]
        }
    
    # Check pulse rate
    if health_data.pulse_rate < normal_ranges["pulse_rate"][0] or health_data.pulse_rate > normal_ranges["pulse_rate"][1]:
        abnormal_readings["pulse_rate"] = {
            "value": health_data.pulse_rate,
            "normal_range": normal_ranges["pulse_rate"]
        }
    
    # Check blood pressure (simplified)
    # In a production app, you'd parse the systolic/diastolic values separately
    # This is just a demonstration
    bp_parts = health_data.blood_pressure.split("/")
    if len(bp_parts) == 2:
        try:
            systolic = int(bp_parts[0])
            diastolic = int(bp_parts[1])
            bp_normal_parts = normal_ranges["blood_pressure"].split("-")
            systolic_normal = int(bp_normal_parts[0])
            diastolic_normal = int(bp_normal_parts[1])
            
            if systolic < systolic_normal or systolic > diastolic_normal or diastolic < systolic_normal or diastolic > diastolic_normal:
                abnormal_readings["blood_pressure"] = {
                    "value": health_data.blood_pressure,
                    "normal_range": normal_ranges["blood_pressure"]
                }
        except:
            abnormal_readings["blood_pressure"] = {
                "value": health_data.blood_pressure,
                "normal_range": normal_ranges["blood_pressure"],
                "note": "Invalid format. Should be systolic/diastolic (e.g., 120/80)"
            }
    
    # Check body temperature
    if health_data.body_temperature < normal_ranges["body_temperature"][0] or health_data.body_temperature > normal_ranges["body_temperature"][1]:
        abnormal_readings["body_temperature"] = {
            "value": health_data.body_temperature,
            "normal_range": normal_ranges["body_temperature"]
        }
    
    # Check blood sugar
    if health_data.blood_sugar < normal_ranges["blood_sugar"][0] or health_data.blood_sugar > normal_ranges["blood_sugar"][1]:
        abnormal_readings["blood_sugar"] = {
            "value": health_data.blood_sugar,
            "normal_range": normal_ranges["blood_sugar"]
        }
    
    # Check rest urine
    if health_data.rest_urine < normal_ranges["rest_urine"][0] or health_data.rest_urine > normal_ranges["rest_urine"][1]:
        abnormal_readings["rest_urine"] = {
            "value": health_data.rest_urine,
            "normal_range": normal_ranges["rest_urine"]
        }
    
    # Check water intake
    if health_data.water_intake < normal_ranges["water_intake"][0] or health_data.water_intake > normal_ranges["water_intake"][1]:
        abnormal_readings["water_intake"] = {
            "value": health_data.water_intake,
            "normal_range": normal_ranges["water_intake"]
        }
    
    # Determine alert level
    alert_level = "Normal"
    if len(abnormal_readings) > 0:
        alert_level = "Warning"
    if "oxygen_saturation" in abnormal_readings or "pulse_rate" in abnormal_readings or "blood_pressure" in abnormal_readings:
        alert_level = "Critical"
    
    # Generate analysis using OpenRouter API
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            prompt = f"""
                Please analyze the following health data and provide detailed medical insights and recommendations. 
                Be professional but easy to understand. Focus especially on any abnormal readings.
                
                Health Data:
                - Oxygen Saturation: {health_data.oxygen_saturation}% (Normal range: 80-100%)
                - Pulse Rate: {health_data.pulse_rate} bpm (Normal range: 50-100 bpm)
                - Blood Pressure: {health_data.blood_pressure} mm Hg (Normal range: 80-120)
                - Body Temperature: {health_data.body_temperature}°F (Normal range: 95-105°F)
                - Blood Sugar: {health_data.blood_sugar} mg/dL (Normal range: 70-140 mg/dL)
                - Rest Urine: {health_data.rest_urine} ml (Normal range: 800-2000 ml)
                - Water Intake: {health_data.water_intake} liters (Normal range: 2.7-3.7 liters)
                
                Abnormal Readings: {json.dumps(abnormal_readings, indent=2)}
                Alert Level: {alert_level}
                
                Provide a comprehensive analysis of this data and clear recommendations for the patient.
            """
            
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": MODEL_NAME,
                    "messages": [{"role": "user", "content": prompt}]
                }
            )
            
            result = response.json()
            
            if "choices" not in result or len(result["choices"]) == 0:
                raise HTTPException(status_code=500, detail="Failed to get analysis from model")
                
            analysis_text = result["choices"][0]["message"]["content"]
            
            # Split the text into analysis and recommendations
            # This is a simple split - in production you might want a more sophisticated approach
            analysis_parts = analysis_text.split("Recommendations:")
            
            analysis = analysis_parts[0].strip()
            recommendations = analysis_parts[1].strip() if len(analysis_parts) > 1 else "No specific recommendations provided."
            
            return AnalysisResponse(
                analysis=analysis,
                recommendations=recommendations,
                alert_level=alert_level,
                abnormal_readings=abnormal_readings
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error performing analysis: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)