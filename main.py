from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, List
from rag_engine import EventRAGEngine
from dotenv import load_dotenv
import os
import shutil
import csv
import json
import re
import uvicorn

load_dotenv()

app = FastAPI(
    title="AI Event Assistant",
    description="One intelligent chat endpoint for everything: Q&A, Navigation, Event Info, and Participant Insights",
    version="7.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG engine
rag_engine = EventRAGEngine()

# ==================== STORAGE ====================

PARTICIPANTS_FILE = "uploads/participants.json"
os.makedirs("uploads", exist_ok=True)

def load_participants():
    if os.path.exists(PARTICIPANTS_FILE):
        with open(PARTICIPANTS_FILE, "r") as f:
            return json.load(f)
    return []

def save_participants(data):
    with open(PARTICIPANTS_FILE, "w") as f:
        json.dump(data, f, indent=2)

# ==================== MODELS ====================

class ChatMessage(BaseModel):
    message: str
    session_id: Optional[str] = "default"

# ==================== ROOT ====================

@app.get("/")
def read_root():
    return {
        "service": "AI Event Assistant",
        "version": "7.0.0",
        "features": [
            "Chat with event data (PDFs)",
            "Venue navigation",
            "Participant insights",
            "Upload floor plans and event documents"
        ],
        "endpoints": {
            "POST /chat": "Main chat endpoint",
            "POST /upload-document": "Upload PDFs (Agenda, Schedule)",
            "POST /upload-floor-plan": "Upload floor plans",
            "POST /upload-participants": "Upload participants JSON or CSV",
            "GET /participants": "View all participants"
        }
    }

# ==================== CHAT ENDPOINT ====================

@app.post("/chat")
def chat(request: ChatMessage):
    message = request.message.strip()
    session_id = request.session_id

    try:
        # Detect participant-related query
        if any(keyword in message.lower() for keyword in ["participant", "attendee", "developer", "designer", "data", "founder", "team", "ai", "ml"]):
            participants = load_participants()
            if not participants:
                return {"response": "No participant data uploaded yet."}

            matches = []
            for p in participants:
                combined = f"{p.get('name', '')} {p.get('role', '')} {p.get('skills', '')} {p.get('organization', '')}".lower()
                if any(word in combined for word in message.lower().split()):
                    matches.append(p)

            if matches:
                response = "👥 Relevant participants:\n"
                for m in matches[:5]:
                    response += f"• {m.get('name', 'Unknown')} — {m.get('role', 'N/A')} ({m.get('organization', 'Independent')})\n"
                return {"response": response, "type": "participants"}
            else:
                return {"response": "No matching participants found.", "type": "participants"}

        # Detect location setting
        if "i am at" in message.lower() or "i'm at" in message.lower():
            match = re.search(r"(?:i am at|i'm at|currently at)\s+(?:the\s+)?(.+)", message.lower())
            if match:
                location = match.group(1).strip()
                result = rag_engine.set_user_location(session_id, location)
                return {"response": f"📍 Location set to {location}. You can now ask for directions.", "type": "location_set"}

        # Fallback to RAG Q&A (event info, navigation, etc.)
        result = rag_engine.ask(message, session_id=session_id)
        return {
            "message": message,
            "response": result.get("answer", "I’m not sure yet, please try again."),
            "type": result.get("type", "general"),
            "session_id": session_id
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==================== ADMIN UPLOADS ====================
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files allowed")

    path = f"uploads/{file.filename}"
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    result = rag_engine.process_pdf(path, metadata={"filename": file.filename})
    return {"message": f"{file.filename} processed", "chunks": result.get("chunks_uploaded")}

@app.post("/upload-floor-plan")
async def upload_floor_plan(file: UploadFile = File(...), floor_name: str = Form("Ground Floor")):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".png", ".jpg", ".jpeg"]:
        raise HTTPException(status_code=400, detail="Only images allowed")

    path = f"floor_plans/{file.filename}"
    os.makedirs("floor_plans", exist_ok=True)
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    result = rag_engine.process_floor_plan_image(path, floor_name, "Main Building")
    return result

@app.post("/upload-participants")
async def upload_participants(file: UploadFile = File(...)):
    """
    Upload participants JSON or CSV
    """
    ext = os.path.splitext(file.filename)[1].lower()
    participants = []

    try:
        if ext == ".json":
            data = json.load(file.file)
            participants.extend(data)
        elif ext == ".csv":
            reader = csv.DictReader(file.file.read().decode("utf-8").splitlines())
            participants.extend(list(reader))
        else:
            raise HTTPException(status_code=400, detail="Only JSON or CSV files allowed")

        save_participants(participants)
        return {"message": f"✅ Uploaded {len(participants)} participants"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/participants")
def get_participants():
    data = load_participants()
    return {"count": len(data), "participants": data}

@app.get("/health")
def health_check():
    try:
        stats = rag_engine.get_statistics()
        return {
            "status": "healthy",
            "documents_indexed": stats.get('total_vectors', 0),
            "participants_loaded": len(load_participants()),
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

# Add these endpoints to your main.py

@app.post("/upload-participants")
async def upload_participants(file: UploadFile = File(...)):
    """
    Upload participants CSV file
    
    Required CSV format:
    participant_id,name,email,role,organization,session_assignments,dietary_preferences
    P001,John Doe,john@example.com,Speaker,TechCorp,"Session A;Session B",Vegetarian
    P002,Jane Smith,jane@example.com,Attendee,StartupXYZ,Session A,None
    
    CSV Columns:
    - participant_id (required): Unique ID (P001, P002, etc.)
    - name (required): Full name
    - email (required): Email address
    - role (optional): Speaker, Attendee, Organizer, Sponsor, VIP
    - organization (optional): Company/Organization name
    - session_assignments (optional): Semicolon-separated session IDs
    - dietary_preferences (optional): Vegetarian, Vegan, Halal, etc.
    - phone (optional): Phone number
    """
    
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files allowed")
    
    try:
        # Read file content
        content = await file.read()
        
        print(f"📊 Processing participants: {file.filename}")
        
        # Process CSV
        result = rag_engine.process_participants_csv(content)
        
        if result["status"] == "error":
            raise HTTPException(status_code=400, detail=result["message"])
        
        return {
            "status": "success",
            "filename": file.filename,
            "participants_added": result["participants_added"],
            "total_participants": result["total_participants"],
            "message": result["message"]
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
    
    finally:
        file.file.close()

@app.post("/upload-sessions")
async def upload_sessions(file: UploadFile = File(...)):
    """
    Upload sessions/schedule CSV file
    
    Required CSV format:
    session_id,title,speaker,location,start_time,end_time,capacity,description
    S001,AI Workshop,Dr. Sarah Chen,Room 101,2024-12-15 09:00,2024-12-15 11:00,50,Introduction to AI
    S002,Keynote Speech,John Smith,Main Hall,2024-12-15 14:00,2024-12-15 15:00,500,Future of Technology
    
    CSV Columns:
    - session_id (required): Unique ID (S001, S002, etc.)
    - title (required): Session title
    - location (required): Room/Location name
    - speaker (optional): Speaker name
    - start_time (optional): Format: YYYY-MM-DD HH:MM
    - end_time (optional): Format: YYYY-MM-DD HH:MM
    - capacity (optional): Maximum attendees
    - description (optional): Session description
    """
    
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files allowed")
    
    try:
        content = await file.read()
        
        print(f"📅 Processing sessions: {file.filename}")
        
        result = rag_engine.process_sessions_csv(content)
        
        if result["status"] == "error":
            raise HTTPException(status_code=400, detail=result["message"])
        
        return {
            "status": "success",
            "filename": file.filename,
            "sessions_added": result["sessions_added"],
            "total_sessions": result["total_sessions"],
            "message": result["message"]
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
    
    finally:
        file.file.close()

@app.get("/participants")
def list_participants():
    """List all registered participants"""
    
    participants = rag_engine.participant_manager.participants
    
    if not participants:
        return {
            "total": 0,
            "message": "No participants uploaded yet. Use POST /upload-participants",
            "participants": []
        }
    
    participant_list = list(participants.values())
    
    # Group by role
    grouped = {}
    for p in participant_list:
        role = p['role']
        if role not in grouped:
            grouped[role] = []
        grouped[role].append(p)
    
    return {
        "total": len(participant_list),
        "by_role": {role: len(group) for role, group in grouped.items()},
        "participants": participant_list[:50],  # First 50
        "message": f"Total {len(participant_list)} participants"
    }

@app.get("/sessions")
def list_sessions():
    """List all event sessions"""
    
    sessions = rag_engine.participant_manager.sessions
    
    if not sessions:
        return {
            "total": 0,
            "message": "No sessions uploaded yet. Use POST /upload-sessions",
            "sessions": []
        }
    
    session_list = list(sessions.values())
    
    return {
        "total": len(session_list),
        "sessions": session_list,
        "message": f"Total {len(session_list)} sessions"
    }

@app.get("/participant/{participant_id}")
def get_participant_details(participant_id: str):
    """Get detailed participant info and schedule"""
    
    if participant_id not in rag_engine.participant_manager.participants:
        raise HTTPException(status_code=404, detail="Participant not found")
    
    participant = rag_engine.participant_manager.participants[participant_id]
    schedule = rag_engine.participant_manager.get_participant_schedule(participant_id)
    
    return {
        "participant": participant,
        "schedule": schedule,
        "sessions": rag_engine.participant_manager.assignments.get(participant_id, [])
    }

@app.get("/session/{session_id}/attendees")
def get_session_attendees(session_id: str):
    """Get all attendees for a specific session"""
    
    if session_id not in rag_engine.participant_manager.sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    attendees_info = rag_engine.participant_manager.get_session_attendees(session_id)
    
    return {
        "session_id": session_id,
        "attendees_info": attendees_info
    }

class Question(BaseModel):
    question: str
    session_id: Optional[str] = "default"


@app.post("/participant-query")
def participant_query(request: Question):
    """
    Ask questions about participants and sessions
    
    Examples:
    - "What is the schedule for John Doe?"
    - "Who is attending the AI Workshop?"
    - "What session is in Room 101?"
    - "Show me all speakers"
    """
    
    try:
        result = rag_engine.ask_with_participants(
            request.question,
            request.session_id
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==================== RUN ====================



if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)

    
