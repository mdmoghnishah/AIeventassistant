import os
from typing import List, Dict, Optional
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
from PyPDF2 import PdfReader
import uuid
from sentence_transformers import SentenceTransformer
import re
from datetime import datetime
import base64
import json
from collections import deque
import csv
import pandas as pd
from typing import List, Dict, Optional
import io

class EventRAGEngine:
    """Complete RAG Engine with Document Q&A + Vision Navigation + Pathfinding"""
    
    def __init__(self):
        print("🚀 Initializing Event Assistant...")
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index_name = os.getenv("PINECONE_INDEX_NAME")
        
        if self.index_name not in [idx["name"] for idx in self.pc.list_indexes()]:
            print(f"   Creating new Pinecone index: {self.index_name}")
            self.pc.create_index(
                name=self.index_name,
                dimension=1024,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
        
        self.index = self.pc.Index(self.index_name)
        
        # Initialize OpenAI
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Initialize embedding model
        print("   Loading embedding model...")
        self.embedding_model = SentenceTransformer('BAAI/bge-small-en-v1.5')
        
        # Conversation history
        self.conversation_history = {}
        
        # Vision Navigation data
        self.floor_plans = {}
        self.venue_map = {}  # Location details
        self.adjacency = {}  # Graph for pathfinding
        self.user_locations = {}
        
        print("✅ Event Assistant ready!")
        print()
    
    # ==================== PDF PROCESSING ====================
    
    def embed_text(self, text: str) -> List[float]:
        return self.embedding_model.encode(text).tolist()
    
    def smart_chunk_text(self, text: str, max_chunk_size: int = 500) -> List[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= max_chunk_size:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + " "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def process_pdf(self, pdf_path: str, metadata: Optional[Dict] = None):
        reader = PdfReader(pdf_path)
        chunks = []
        
        doc_title = os.path.basename(pdf_path)
        print(f"📄 Processing: {doc_title}")
        print(f"📊 Pages: {len(reader.pages)}")
        
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            page_chunks = self.smart_chunk_text(text)
            
            for chunk_text in page_chunks:
                if len(chunk_text) > 50:
                    chunk_metadata = {
                        "text": chunk_text,
                        "page": page_num + 1,
                        "source": doc_title,
                        "doc_type": "pdf"
                    }
                    
                    if metadata:
                        chunk_metadata.update(metadata)
                    
                    chunks.append({
                        "id": str(uuid.uuid4()),
                        "values": self.embed_text(chunk_text),
                        "metadata": chunk_metadata
                    })
        
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            self.index.upsert(vectors=batch)
        
        print(f"✅ Uploaded {len(chunks)} chunks")
        
        return {
            "chunks_uploaded": len(chunks),
            "pages_processed": len(reader.pages),
            "document_title": doc_title,
            "summary": f"Processed {len(reader.pages)} pages"
        }
    
    # ==================== VISION NAVIGATION + GRAPH BUILDING ====================
    
    def encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    
    def process_floor_plan_image(self, image_path: str, floor_name: str, building: str) -> Dict:
        """Process floor plan and build adjacency graph"""
        
        print(f"🖼️  Processing: {image_path}")
        print(f"📍 {floor_name}, {building}")
        
        try:
            base64_image = self.encode_image(image_path)
            
            prompt = """Analyze this floor plan. Extract ALL rooms AND their connections.

Return ONLY valid JSON:
{
  "locations": [
    {
      "name": "Kitchen",
      "type": "kitchen",
      "relative_position": "northwest corner",
      "nearby": ["Living Room"]
    },
    {
      "name": "Living Room",
      "type": "living_room",
      "relative_position": "center",
      "nearby": ["Kitchen", "Bathroom", "Bedroom 1"]
    }
  ]
}

IMPORTANT: The "nearby" field must list directly connected rooms (adjacency)."""

            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": "high"}}
                    ]
                }],
                max_tokens=2000,
                temperature=0.3
            )
            
            content = response.choices[0].message.content.strip()
            parsed_data = self._parse_json_from_response(content)
            
            if not parsed_data or not parsed_data.get("locations"):
                print("⚠️ No locations found, using fallback")
                # Fallback: Create simple layout
                parsed_data = self._create_fallback_layout()
            
            # Store floor plan
            floor_key = f"{building}_{floor_name}"
            self.floor_plans[floor_key] = {
                "image_path": image_path,
                "floor": floor_name,
                "building": building,
                "parsed_data": parsed_data
            }
            
            # Build venue map and adjacency graph
            locations_found = 0
            for loc in parsed_data.get("locations", []):
                loc_name = loc["name"]
                loc_key = loc_name.lower()
                
                # Store location details
                self.venue_map[loc_key] = {
                    "name": loc_name,
                    "type": loc.get("type", "room"),
                    "floor": floor_name,
                    "building": building,
                    "relative_position": loc.get("relative_position", ""),
                    "nearby": loc.get("nearby", []),
                    "floor_key": floor_key
                }
                
                # Build adjacency graph (case-insensitive)
                nearby_list = [n.lower() for n in loc.get("nearby", [])]
                self.adjacency[loc_key] = nearby_list
                
                locations_found += 1
            
            print(f"✅ Found {locations_found} locations")
            print(f"🗺️  Built adjacency graph:")
            for loc, neighbors in list(self.adjacency.items())[:5]:
                print(f"   {loc.title()} → {neighbors}")
            
            return {
                "status": "success",
                "floor": floor_name,
                "building": building,
                "locations_found": locations_found,
                "locations": [self.venue_map[k]["name"] for k in self.venue_map.keys() if self.venue_map[k]["floor_key"] == floor_key]
            }
        
        except Exception as e:
            print(f"❌ Error: {e}")
            return {"status": "error", "message": str(e)}
    
    def _create_fallback_layout(self) -> Dict:
        """Create a simple fallback layout if vision fails"""
        return {
            "locations": [
                {"name": "Kitchen", "type": "kitchen", "relative_position": "northwest", "nearby": ["Living Room"]},
                {"name": "Living Room", "type": "living_room", "relative_position": "center", "nearby": ["Kitchen", "Bathroom", "Bedroom 1"]},
                {"name": "Bathroom", "type": "bathroom", "relative_position": "northeast", "nearby": ["Living Room"]},
                {"name": "Bedroom 1", "type": "bedroom", "relative_position": "south", "nearby": ["Living Room", "Bedroom 2"]},
                {"name": "Bedroom 2", "type": "bedroom", "relative_position": "southeast", "nearby": ["Bedroom 1", "Terrace"]},
                {"name": "Terrace", "type": "terrace", "relative_position": "east", "nearby": ["Bedroom 2"]}
            ]
        }
    
    def _parse_json_from_response(self, content: str) -> Optional[Dict]:
        content = re.sub(r'```json\s*', '', content)
        content = re.sub(r'```\s*', '', content)
        
        try:
            return json.loads(content)
        except:
            pass
        
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except:
                pass
        
        return None
    
    # ==================== PATHFINDING (BFS) ====================
    
    def find_path(self, start: str, end: str) -> Optional[List[str]]:
        """Find shortest path using Breadth-First Search"""
        start_key = start.lower()
        end_key = end.lower()
        
        if start_key not in self.adjacency or end_key not in self.adjacency:
            return None
        
        visited = {start_key}
        queue = deque([(start_key, [start_key])])
        
        while queue:
            node, path = queue.popleft()
            
            if node == end_key:
                return path
            
            for neighbor in self.adjacency.get(node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        return None
    
    def get_navigation_instructions(self, start: str, end: str) -> str:
        """Generate human-readable navigation with pathfinding"""
        path = self.find_path(start, end)
        
        if not path:
            return f"Sorry, I couldn't find a route from {start.title()} to {end.title()}."
        
        if len(path) == 1:
            return f"You're already at {end.title()}!"
        
        # Build readable directions
        steps = []
        steps.append(f"🚶 Navigation from {start.title()} to {end.title()}:\n")
        steps.append(f"📍 Route: {' → '.join([p.title() for p in path])}")
        steps.append(f"\n📋 Step-by-step:")
        
        for i in range(len(path) - 1):
            current = path[i]
            next_loc = path[i + 1]
            steps.append(f"   {i+1}. From {current.title()}, go to {next_loc.title()}")
        
        steps.append(f"\n✅ You will arrive at {end.title()}")
        steps.append(f"🚶 Total steps: {len(path) - 1}")
        
        return "\n".join(steps)
    
    # ==================== INTELLIGENT Q&A ====================
    
    def ask(self, question: str, filters: Optional[Dict] = None, session_id: str = "default") -> Dict:
        """
        Smart Q&A that handles:
        1. Navigation between locations (using pathfinding)
        2. Document Q&A (using RAG)
        3. Mixed queries
        """
        
        message = question.lower()
        
        # Detect "I am at..." to set location
        if "i am at" in message or "i'm at" in message:
            loc_match = re.search(r"(?:i am at|i'm at)\s+(.+?)(?:\.|$|\?)", message)
            if loc_match:
                location = loc_match.group(1).strip()
                result = self.set_user_location(session_id, location)
                return {
                    "question": question,
                    "answer": result,
                    "type": "location_set"
                }
        
        # Detect navigation intent with FROM-TO pattern
        nav_patterns = [
            r"(?:from|at)\s+(.+?)\s+(?:to|reach|get to|go to)\s+(.+?)(?:\?|$|\.|,)",
            r"(?:help|navigate|directions?|way|route)\s+(?:from|at)?\s*(.+?)\s+(?:to|reach)\s+(.+?)(?:\?|$|\.|,)",
            r"(?:how do i get|take me)\s+(?:from\s+)?(.+?)\s+to\s+(.+?)(?:\?|$|\.|,)",
            r"(?:to|reach|get to)\s+(.+?)\s+from\s+(.+?)(?:\?|$|\.|,)",
        ]
        
        from_loc = None
        to_loc = None
        
        for pattern in nav_patterns:
            match = re.search(pattern, message)
            if match:
                # Try both group orders
                if "from" in pattern and pattern.index("from") < pattern.index("to"):
                    from_loc = match.group(1).strip()
                    to_loc = match.group(2).strip()
                else:
                    to_loc = match.group(1).strip()
                    from_loc = match.group(2).strip()
                break
        
        # If no "from" specified, use session location
        if to_loc and not from_loc:
            from_loc = self.user_locations.get(session_id)
        
        # If we have navigation request
        if from_loc and to_loc and len(self.adjacency) > 0:
            # Find actual location names (case-insensitive)
            from_match = self._find_location(from_loc)
            to_match = self._find_location(to_loc)
            
            if not from_match:
                available = ', '.join([self.venue_map[k]["name"] for k in list(self.venue_map.keys())[:5]])
                return {
                    "question": question,
                    "answer": f"❌ Couldn't find '{from_loc}'. Available locations: {available}...",
                    "type": "error"
                }
            
            if not to_match:
                available = ', '.join([self.venue_map[k]["name"] for k in list(self.venue_map.keys())[:5]])
                return {
                    "question": question,
                    "answer": f"❌ Couldn't find '{to_loc}'. Available locations: {available}...",
                    "type": "error"
                }
            
            # Get navigation instructions with pathfinding
            nav_instructions = self.get_navigation_instructions(from_match, to_match)
            
            return {
                "question": question,
                "answer": nav_instructions,
                "from": self.venue_map[from_match]["name"],
                "to": self.venue_map[to_match]["name"],
                "type": "navigation"
            }
        
        # Check for simple "where is X" queries
        where_match = re.search(r"where is\s+(?:the\s+)?(.+?)(?:\?|$)", message)
        if where_match and len(self.venue_map) > 0:
            location = where_match.group(1).strip()
            loc_match = self._find_location(location)
            
            if loc_match:
                loc_data = self.venue_map[loc_match]
                answer = f"📍 {loc_data['name']} Location:\n\n"
                answer += f"• Floor: {loc_data['floor']}\n"
                answer += f"• Building: {loc_data['building']}\n"
                answer += f"• Position: {loc_data['relative_position']}\n"
                
                if loc_data.get('nearby'):
                    answer += f"• Adjacent to: {', '.join([n.title() for n in loc_data['nearby']])}\n"
                
                current_loc = self.user_locations.get(session_id)
                if current_loc and current_loc.lower() != loc_match:
                    answer += f"\n💡 Tip: I can navigate you there! Just say 'Take me from {current_loc.title()} to {loc_data['name']}'"
                
                return {
                    "question": question,
                    "answer": answer,
                    "type": "location_info"
                }
        
        # Fallback to document RAG
        context = self.retrieve_context(question, filters=filters)
        
        if not context:
            suggestion = "I don't have any information in the uploaded documents yet.\n\n"
            
            if len(self.venue_map) > 0:
                locs = ', '.join([self.venue_map[k]["name"] for k in list(self.venue_map.keys())[:5]])
                suggestion += f"However, I can help with navigation! Available locations: {locs}...\n\n"
                suggestion += "Try asking:\n• 'Take me from Kitchen to Bathroom'\n• 'Where is the Living Room?'\n• 'I am at Kitchen' (to set your location)"
            else:
                suggestion += "Please upload:\n• PDF documents (POST /upload)\n• Floor plan images (POST /upload-floor-plan)"
            
            return {
                "question": question,
                "answer": suggestion,
                "type": "help"
            }
        
        answer = self.generate_response(question, context)
        
        return {
            "question": question,
            "answer": answer,
            "context_preview": context[:200] + "...",
            "type": "document_qa"
        }
    
    def retrieve_context(self, query: str, top_k: int = 5, filters: Optional[Dict] = None) -> str:
        query_embedding = self.embed_text(query)
        
        search_params = {
            "vector": query_embedding,
            "top_k": top_k,
            "include_metadata": True
        }
        
        if filters:
            search_params["filter"] = filters
        
        results = self.index.query(**search_params)
        
        if not results.get('matches'):
            return ""
        
        context_parts = []
        for match in results['matches']:
            text = match['metadata'].get('text', '')
            page = match['metadata'].get('page', '?')
            source = match['metadata'].get('source', 'Unknown')
            context_parts.append(f"[{source} p.{page}] {text}")
        
        return "\n\n".join(context_parts)
    
    def generate_response(self, query: str, context: str) -> str:
        prompt = f"""Answer using the context provided.

Context:
{context}

Question: {query}

Answer clearly and concisely:"""

        response = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful event assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        return response.choices[0].message.content
    
    def _find_location(self, query: str) -> Optional[str]:
        """Find location with fuzzy matching"""
        query_lower = query.lower().strip()
        
        # Exact match
        if query_lower in self.venue_map:
            return query_lower
        
        # Partial match
        for loc_key in self.venue_map.keys():
            if query_lower in loc_key or loc_key in query_lower:
                return loc_key
        
        return None
    
    # ==================== USER LOCATION ====================
    
    def set_user_location(self, session_id: str, location: str) -> str:
        matched_location = self._find_location(location)
        
        if matched_location:
            self.user_locations[session_id] = matched_location
            loc_name = self.venue_map[matched_location]["name"]
            return f"✅ Your location set to: {loc_name}"
        
        available = ', '.join([self.venue_map[k]["name"] for k in list(self.venue_map.keys())[:5]])
        return f"❌ '{location}' not found. Available: {available}..."
    
    def get_all_locations(self) -> List[str]:
        return [self.venue_map[k]["name"] for k in self.venue_map.keys()]
    
    def get_floor_plans_summary(self) -> Dict:
        return {
            "total_floors": len(self.floor_plans),
            "total_locations": len(self.venue_map),
            "locations": self.get_all_locations()
        }
    
    # ==================== STATS ====================
    
    def get_statistics(self) -> Dict:
        try:
            stats = self.index.describe_index_stats()
            return {
                "total_vectors": stats.get('total_vector_count', 0),
                "dimension": stats.get('dimension', 0),
                "active_sessions": len(self.conversation_history)
            }
        except:
            return {"total_vectors": 0, "dimension": 1024, "active_sessions": 0}
    
    def clear_session(self, session_id: str) -> bool:
        if session_id in self.conversation_history:
            del self.conversation_history[session_id]
            return True
        return False
    
    def export_conversation(self, session_id: str) -> Optional[List[Dict]]:
        return self.conversation_history.get(session_id)
    
    def get_directions(self, from_loc: str, to_loc: str, session_id: str = "default") -> str:
        """Direct directions API (for /directions endpoint)"""
        from_match = self._find_location(from_loc)
        to_match = self._find_location(to_loc)
        
        if not from_match or not to_match:
            return "❌ One or both locations not found"
        
        return self.get_navigation_instructions(from_match, to_match)
    



class ParticipantManager:
    """Manage event participants and their assignments"""
    
    def __init__(self):
        self.participants = {}  # participant_id -> participant data
        self.sessions = {}      # session_id -> session data
        self.assignments = {}   # participant_id -> [session_ids]
    
    def process_participant_csv(self, file_content: bytes) -> Dict:
        """
        Process participant CSV file
        
        Expected CSV format:
        participant_id,name,email,role,organization,session_assignments,dietary_preferences
        P001,John Doe,john@example.com,Speaker,TechCorp,"Session A;Session B",Vegetarian
        P002,Jane Smith,jane@example.com,Attendee,StartupXYZ,Session A,None
        """
        
        try:
            # Read CSV
            df = pd.read_csv(io.BytesIO(file_content))
            
            # Required columns
            required_cols = ['participant_id', 'name', 'email']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                return {
                    "status": "error",
                    "message": f"Missing required columns: {', '.join(missing_cols)}"
                }
            
            participants_added = 0
            
            for _, row in df.iterrows():
                participant_id = str(row['participant_id'])
                
                # Store participant data
                self.participants[participant_id] = {
                    "id": participant_id,
                    "name": row['name'],
                    "email": row['email'],
                    "role": row.get('role', 'Attendee'),
                    "organization": row.get('organization', ''),
                    "dietary_preferences": row.get('dietary_preferences', ''),
                    "phone": row.get('phone', ''),
                    "badge_printed": False,
                    "checked_in": False
                }
                
                # Process session assignments
                if 'session_assignments' in row and pd.notna(row['session_assignments']):
                    sessions = [s.strip() for s in str(row['session_assignments']).split(';')]
                    self.assignments[participant_id] = sessions
                else:
                    self.assignments[participant_id] = []
                
                participants_added += 1
            
            return {
                "status": "success",
                "participants_added": participants_added,
                "total_participants": len(self.participants),
                "message": f"✅ Successfully added {participants_added} participants"
            }
        
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error processing CSV: {str(e)}"
            }
    
    def process_session_csv(self, file_content: bytes) -> Dict:
        """
        Process session/event schedule CSV
        
        Expected format:
        session_id,title,speaker,location,start_time,end_time,capacity,description
        S001,AI Workshop,Dr. Sarah Chen,Room 101,2024-12-15 09:00,2024-12-15 11:00,50,Intro to AI
        S002,Keynote,John Smith,Main Hall,2024-12-15 14:00,2024-12-15 15:00,500,Future of Tech
        """
        
        try:
            df = pd.read_csv(io.BytesIO(file_content))
            
            required_cols = ['session_id', 'title', 'location']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                return {
                    "status": "error",
                    "message": f"Missing required columns: {', '.join(missing_cols)}"
                }
            
            sessions_added = 0
            
            for _, row in df.iterrows():
                session_id = str(row['session_id'])
                
                self.sessions[session_id] = {
                    "id": session_id,
                    "title": row['title'],
                    "speaker": row.get('speaker', ''),
                    "location": row['location'],
                    "start_time": row.get('start_time', ''),
                    "end_time": row.get('end_time', ''),
                    "capacity": int(row.get('capacity', 0)) if pd.notna(row.get('capacity')) else 0,
                    "description": row.get('description', '')
                }
                
                sessions_added += 1
            
            return {
                "status": "success",
                "sessions_added": sessions_added,
                "total_sessions": len(self.sessions),
                "message": f"✅ Successfully added {sessions_added} sessions"
            }
        
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error processing CSV: {str(e)}"
            }
    
    def find_participant(self, query: str) -> Optional[Dict]:
        """Find participant by name, email, or ID"""
        query_lower = query.lower()
        
        for p_id, participant in self.participants.items():
            if (query_lower in participant['name'].lower() or 
                query_lower in participant['email'].lower() or
                query_lower == p_id.lower()):
                return participant
        
        return None
    
    def get_participant_schedule(self, participant_id: str) -> str:
        """Get a participant's full schedule"""
        
        if participant_id not in self.participants:
            return "❌ Participant not found"
        
        participant = self.participants[participant_id]
        sessions = self.assignments.get(participant_id, [])
        
        response = f"📋 Schedule for {participant['name']} ({participant['role']})\n"
        response += f"📧 {participant['email']}\n\n"
        
        if not sessions:
            response += "No sessions assigned yet."
            return response
        
        response += f"📅 Assigned Sessions ({len(sessions)}):\n\n"
        
        for session_id in sessions:
            if session_id in self.sessions:
                session = self.sessions[session_id]
                response += f"• {session['title']}\n"
                response += f"  📍 Location: {session['location']}\n"
                response += f"  🕐 Time: {session['start_time']}\n"
                if session.get('speaker'):
                    response += f"  🎤 Speaker: {session['speaker']}\n"
                response += "\n"
        
        return response
    
    def get_session_attendees(self, session_id: str) -> str:
        """Get all attendees for a session"""
        
        if session_id not in self.sessions:
            return "❌ Session not found"
        
        session = self.sessions[session_id]
        
        response = f"📊 Attendees for: {session['title']}\n"
        response += f"📍 Location: {session['location']}\n"
        response += f"🕐 Time: {session['start_time']}\n\n"
        
        attendees = [
            self.participants[p_id]
            for p_id, assigned in self.assignments.items()
            if session_id in assigned and p_id in self.participants
        ]
        
        if not attendees:
            response += "No attendees registered yet."
            return response
        
        response += f"👥 Registered Attendees ({len(attendees)}):\n\n"
        
        for attendee in attendees:
            response += f"• {attendee['name']} ({attendee['role']})\n"
            response += f"  {attendee['organization']}\n"
            response += f"  📧 {attendee['email']}\n\n"
        
        if session.get('capacity'):
            response += f"📈 Capacity: {len(attendees)}/{session['capacity']}"
        
        return response
    
    def search_by_session_location(self, location: str) -> str:
        """Find which session is at a location"""
        
        location_lower = location.lower()
        matching_sessions = []
        
        for session in self.sessions.values():
            if location_lower in session['location'].lower():
                matching_sessions.append(session)
        
        if not matching_sessions:
            return f"❌ No sessions found at '{location}'"
        
        response = f"📍 Sessions at {location}:\n\n"
        
        for session in matching_sessions:
            response += f"• {session['title']}\n"
            response += f"  🕐 {session['start_time']}\n"
            if session.get('speaker'):
                response += f"  🎤 {session['speaker']}\n"
            response += "\n"
        
        return response


# ==================== UPDATE YOUR EventRAGEngine ====================

# Add this to your EventRAGEngine class __init__:
# self.participant_manager = ParticipantManager()

# Add these methods to EventRAGEngine:

def process_participants_csv(self, file_content: bytes) -> Dict:
    """Process participant CSV upload"""
    return self.participant_manager.process_participant_csv(file_content)

def process_sessions_csv(self, file_content: bytes) -> Dict:
    """Process sessions CSV upload"""
    return self.participant_manager.process_session_csv(file_content)

def ask_with_participants(self, question: str, session_id: str = "default") -> Dict:
    """Enhanced ask() that includes participant/session queries"""
    
    message = question.lower()
    
    # Check for participant queries
    if any(kw in message for kw in ['schedule for', 'sessions for', 'what is attending']):
        # Extract name/email
        name_match = re.search(r'(?:schedule for|sessions for)\s+(.+?)(?:\?|$)', message)
        if name_match:
            name = name_match.group(1).strip()
            participant = self.participant_manager.find_participant(name)
            
            if participant:
                schedule = self.participant_manager.get_participant_schedule(participant['id'])
                return {
                    "question": question,
                    "answer": schedule,
                    "type": "participant_schedule"
                }
    
    # Check for session attendee queries
    if any(kw in message for kw in ['who is attending', 'attendees for', 'participants in']):
        session_match = re.search(r'(?:attending|for|in)\s+(.+?)(?:\?|$)', message)
        if session_match:
            session_name = session_match.group(1).strip()
            
            for session_id, session in self.participant_manager.sessions.items():
                if session_name.lower() in session['title'].lower():
                    attendees = self.participant_manager.get_session_attendees(session_id)
                    return {
                        "question": question,
                        "answer": attendees,
                        "type": "session_attendees"
                    }
    
    # Check for location-based session queries
    if 'what session' in message or 'which session' in message:
        # Extract location
        loc_match = re.search(r'(?:at|in)\s+(.+?)(?:\?|$)', message)
        if loc_match:
            location = loc_match.group(1).strip()
            sessions_info = self.participant_manager.search_by_session_location(location)
            return {
                "question": question,
                "answer": sessions_info,
                "type": "location_sessions"
            }
    
    # Fallback to regular ask() method
    return self.ask(question, session_id=session_id)

