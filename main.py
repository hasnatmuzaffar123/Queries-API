import os
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
import google.generativeai as genai
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables")

genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-2.5-flash")  


class SymptomRequest(BaseModel):
    symptom_name: str
    description: str
    severity: str
    duration: str


ALLOWED_KEYWORDS = [
    "child", "baby", "infant", "toddler", "kid", "newborn",
    "fever", "cough", "vomiting", "diarrhea", "constipation",
    "rash", "vaccination", "vaccine", "immunization",
    "weight", "height", "growth", "development",
    "milestone", "sleep", "nutrition", "feeding", "breastfeed",
    "teething", "diaper", "allergy","cold", "flu"
]

EMERGENCY_KEYWORDS = [
    "seizure", "seizures", "unconscious", "unconsciousness",
    "not breathing", "choking", "asphyxia",
    "blue lips", "cyanosis", "severe breathing",
    "very high fever", "high fever over", "104", "40c",
    "severe bleeding", "unresponsive", "poisoning", "overdose",
    "severe allergic", "anaphylaxis", "head injury", "fall",
    "unable to wake", "extreme lethargy", "rigid neck"
]

SYSTEM_PROMPT = """You are a pediatric health assistant for parents caring for children aged 0-5 years.

CRITICAL RULES:
1. ONLY answer questions about child health (ages 0-5 years)
2. NEVER diagnose, prescribe medications, or provide definitive medical advice
3. Provide general educational information only
4. Keep responses SHORT and TO THE POINT (2-3 sentences max, use bullet points if needed)
5. ALWAYS include this disclaimer at the end of every response:
   "⚠️ Disclaimer: This information is for educational purposes only. Please consult a pediatrician for proper medical advice."

RESPONSE GUIDELINES:
- Be direct and concise - avoid long paragraphs
- Use bullet points for multiple points
- Give practical, actionable advice not medication
- Suggest when to contact a doctor
- Keep language simple and non-technical

For unrelated questions, respond with:
"I'm specifically designed to help with child health questions. This question appears unrelated to child health."""


def is_child_health_query(text: str) -> bool:
    text_lower = text.lower()
    return any(word in text_lower for word in ALLOWED_KEYWORDS)


def is_emergency(text: str) -> bool:
    text_lower = text.lower()
    return any(word in text_lower for word in EMERGENCY_KEYWORDS)


def sanitize_response(response_text: str) -> str:
    disclaimer = "\n\n⚠️ Disclaimer: This information is for educational purposes only and is NOT a medical diagnosis. Please consult a pediatrician for proper medical advice."
    
    if "Disclaimer" not in response_text:
        return response_text + disclaimer
    return response_text


@app.get("/")
def root():
    return {"message": "ParentPal API is running", "docs": "/docs"}


@app.post("/ask-symptom")
def ask_symptom(request: SymptomRequest):
    
    if not request.symptom_name or len(request.symptom_name.strip()) < 2:
        raise HTTPException(
            status_code=400,
            detail="Please provide a symptom name"
        )
    
    if not request.description or len(request.description.strip()) < 5:
        raise HTTPException(
            status_code=400,
            detail="Please provide a detailed description"
        )
    
    combined_query = f"{request.symptom_name} {request.description} {request.severity} {request.duration}".lower()
    
    if not is_child_health_query(combined_query):
        return {
            "response": "I'm specifically designed to help with child health questions. This question appears unrelated to child health.",
            "emergency": False,
            "symptom": request.symptom_name
        }
    
    if is_emergency(combined_query):
        return {
            "response": "🚨 EMERGENCY ALERT: Based on your description, this may be a medical emergency. Call emergency services or visit the nearest hospital immediately.",
            "emergency": True,
            "symptom": request.symptom_name
        }
    
    try:
        structured_query = f"""
        Symptom: {request.symptom_name}
        Description: {request.description}
        Severity: {request.severity}
        Duration: {request.duration}
        
        Please provide guidance for this child health concern.
        """
        
        response = model.generate_content(SYSTEM_PROMPT + "\n\nParent's Symptom Report: " + structured_query)
        
        if not response.text:
            raise ValueError("Empty response from API")
        
        sanitized_response = sanitize_response(response.text)
        
        return {
            "response": sanitized_response,
            "emergency": False,
            "symptom": request.symptom_name,
            "severity": request.severity,
            "duration": request.duration
        }
    
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"ERROR: {error_trace}")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred: {str(e)}"
        )


@app.get("/health")
def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)