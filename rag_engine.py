import os
import re
import json
import base64
import uuid
from typing import List, Dict, Optional
from collections import deque
from datetime import datetime

from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer

class EventRAGEngine:
    """Optimized RAG Engine with Document Q&A + Vision Navigation + Pathfinding"""
    
    def __init__(self):
        print("🚀 Initializing Event Assistant...")
        
        # Pinecone setup
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index_name = os.getenv("PINECONE_INDEX_NAME")
        
        if self.index_name not in [idx["name"] for idx in self.pc.list_indexes()]:
            self.pc.create_index(
                name=self.index_name,
                dimension=384,  # Smaller model = faster
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
        
        self.index = self.pc.Index(self.index_name)
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Use smaller, faster embedding model
        print("   Loading embedding model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # 384 dim, fast!
        
        # Core data structures
        self.conversation_history = {}
        self.floor_plans = {}
        self.venue_map = {}
        self.adjacency = {}
        self.user_locations = {}
        
        print("✅ Ready!\n")
    
    # ==================== OPTIMIZED EMBEDDING ====================
    
    def embed_text(self, text: str) -> List[float]:
        """Fast embedding with caching potential"""
        return self.embedding_model.encode(text, show_progress_bar=False).tolist()
    
    # ==================== OPTIMIZED PDF PROCESSING ====================
    
    def process_pdf(self, pdf_path: str, metadata: Optional[Dict] = None) -> Dict:
        """Streamlined PDF processing"""
        reader = PdfReader(pdf_path)
        doc_title = os.path.basename(pdf_path)
        
        print(f"📄 {doc_title} ({len(reader.pages)} pages)")
        
        chunks = []
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            # Simple sentence-based chunking
            sentences = re.split(r'(?<=[.!?])\s+', text)
            
            current_chunk = ""
            for sentence in sentences:
                if len(current_chunk) + len(sentence) < 500:
                    current_chunk += sentence + " "
                else:
                    if len(current_chunk) > 50:
                        chunks.append({
                            "id": str(uuid.uuid4()),
                            "values": self.embed_text(current_chunk.strip()),
                            "metadata": {
                                "text": current_chunk.strip(),
                                "page": page_num + 1,
                                "source": doc_title
                            }
                        })
                    current_chunk = sentence + " "
        
        # Batch upload
        for i in range(0, len(chunks), 100):
            self.index.upsert(vectors=chunks[i:i+100])
        
        print(f"✅ {len(chunks)} chunks indexed\n")
        
        return {
            "chunks_uploaded": len(chunks),
            "pages_processed": len(reader.pages),
            "document_title": doc_title
        }
    
    # ==================== OPTIMIZED VISION NAVIGATION ====================
    
    def process_floor_plan_image(self, image_path: str, floor_name: str, building: str) -> Dict:
        """Optimized floor plan processing with fallback"""
        
        print(f"🖼️  {os.path.basename(image_path)}")
        
        try:
            with open(image_path, "rb") as f:
                base64_image = base64.b64encode(f.read()).decode()
            
            # Concise prompt for faster response
            prompt = """Extract rooms and connections from this floor plan. Return ONLY JSON:
{
  "locations": [
    {"name": "Kitchen", "type": "kitchen", "position": "northwest", "nearby": ["Living Room"]},
    {"name": "Living Room", "type": "living_room", "position": "center", "nearby": ["Kitchen", "Bathroom"]}
  ]
}"""
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",  # Faster, cheaper model
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "low"  # Faster processing
                        }}
                    ]
                }],
                max_tokens=1000,
                temperature=0.3
            )
            
            content = response.choices[0].message.content
            parsed = self._parse_json(content)
            
            if not parsed or not parsed.get("locations"):
                parsed = self._fallback_layout()
            
        except Exception as e:
            print(f"⚠️  Vision failed, using fallback: {e}")
            parsed = self._fallback_layout()
        
        # Build maps
        floor_key = f"{building}_{floor_name}"
        self.floor_plans[floor_key] = {"floor": floor_name, "building": building}
        
        for loc in parsed.get("locations", []):
            key = loc["name"].lower()
            self.venue_map[key] = {
                "name": loc["name"],
                "type": loc.get("type", "room"),
                "floor": floor_name,
                "building": building,
                "position": loc.get("position", ""),
                "floor_key": floor_key
            }
            self.adjacency[key] = [n.lower() for n in loc.get("nearby", [])]
        
        print(f"✅ {len(self.venue_map)} locations\n")
        
        return {
            "status": "success",
            "locations_found": len(parsed.get("locations", [])),
            "locations": [self.venue_map[k]["name"] for k in self.venue_map.keys()]
        }
    
    def _parse_json(self, content: str) -> Optional[Dict]:
        """Fast JSON parsing with cleanup"""
        content = re.sub(r'```json?', '', content).replace('```', '').strip()
        try:
            return json.loads(content)
        except:
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except:
                    pass
        return None
    
    def _fallback_layout(self) -> Dict:
        """Simple fallback layout"""
        return {
            "locations": [
                {"name": "Kitchen", "nearby": ["Living Room"]},
                {"name": "Living Room", "nearby": ["Kitchen", "Bathroom", "Bedroom"]},
                {"name": "Bathroom", "nearby": ["Living Room"]},
                {"name": "Bedroom", "nearby": ["Living Room", "Terrace"]},
                {"name": "Terrace", "nearby": ["Bedroom"]}
            ]
        }
    
    # ==================== OPTIMIZED PATHFINDING ====================
    
    def find_path(self, start: str, end: str) -> Optional[List[str]]:
        """BFS pathfinding - O(V+E) complexity"""
        start, end = start.lower(), end.lower()
        
        if start not in self.adjacency or end not in self.adjacency:
            return None
        
        visited = {start}
        queue = deque([(start, [start])])
        
        while queue:
            node, path = queue.popleft()
            if node == end:
                return path
            
            for neighbor in self.adjacency.get(node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        return None
    
    def get_directions(self, start: str, end: str) -> str:
        """Generate readable directions"""
        path = self.find_path(start, end)
        
        if not path:
            return f"❌ No route found"
        
        if len(path) == 1:
            return f"✅ You're already at {end.title()}!"
        
        steps = [f"🚶 {start.title()} → {end.title()}\n"]
        steps.append(f"📍 Route: {' → '.join([p.title() for p in path])}\n")
        
        for i in range(len(path) - 1):
            steps.append(f"{i+1}. Go to {path[i+1].title()}")
        
        steps.append(f"\n✅ Arrive at {end.title()} ({len(path)-1} steps)")
        
        return "\n".join(steps)
    
    # ==================== INTELLIGENT Q&A ====================
    
    def ask(self, question: str, filters: Optional[Dict] = None, session_id: str = "default") -> Dict:
        """Unified intelligent query handler"""
        
        msg = question.lower()
        
        # 1. Location setting
        if "i am at" in msg or "i'm at" in msg:
            match = re.search(r"(?:i am at|i'm at)\s+(.+?)(?:\.|$|\?)", msg)
            if match:
                loc = match.group(1).strip()
                result = self.set_user_location(session_id, loc)
                return {"question": question, "answer": result, "type": "location_set"}
        
        # 2. Navigation request
        nav_result = self._handle_navigation(question, session_id)
        if nav_result:
            return nav_result
        
        # 3. Location info
        where_match = re.search(r"where is\s+(?:the\s+)?(.+?)(?:\?|$)", msg)
        if where_match and self.venue_map:
            loc = self._find_location(where_match.group(1).strip())
            if loc:
                data = self.venue_map[loc]
                answer = f"📍 {data['name']}\n• {data['floor']}, {data['building']}\n• {data['position']}"
                return {"question": question, "answer": answer, "type": "location_info"}
        
        # 4. Document Q&A
        context = self._retrieve_context(question, filters)
        
        if not context:
            help_msg = "No documents uploaded yet.\n\n"
            if self.venue_map:
                locs = ', '.join(list(self.venue_map.values())[:3])
                help_msg += f"Try: 'I am at Kitchen' or 'Take me to {list(self.venue_map.keys())[0].title()}'"
            return {"question": question, "answer": help_msg, "type": "help"}
        
        answer = self._generate_answer(question, context)
        
        return {
            "question": question,
            "answer": answer,
            "type": "document_qa"
        }
    
    def _handle_navigation(self, question: str, session_id: str) -> Optional[Dict]:
        """Handle navigation queries"""
        msg = question.lower()
        
        # Pattern matching for from-to navigation
        patterns = [
            r"(?:from|at)\s+(.+?)\s+(?:to|reach|get to)\s+(.+?)(?:\?|$)",
            r"(?:take me|navigate)\s+(?:from\s+)?(.+?)\s+to\s+(.+?)(?:\?|$)",
        ]
        
        from_loc, to_loc = None, None
        
        for pattern in patterns:
            match = re.search(pattern, msg)
            if match:
                from_loc, to_loc = match.group(1).strip(), match.group(2).strip()
                break
        
        # Simple "go to X" uses session location
        if not from_loc and to_loc is None:
            go_match = re.search(r"(?:go to|get to|take me to)\s+(.+?)(?:\?|$)", msg)
            if go_match:
                to_loc = go_match.group(1).strip()
                from_loc = self.user_locations.get(session_id)
        
        if from_loc and to_loc:
            from_match = self._find_location(from_loc)
            to_match = self._find_location(to_loc)
            
            if not from_match or not to_match:
                return {"question": question, "answer": "❌ Location not found", "type": "error"}
            
            directions = self.get_directions(from_match, to_match)
            
            return {
                "question": question,
                "answer": directions,
                "from": self.venue_map[from_match]["name"],
                "to": self.venue_map[to_match]["name"],
                "type": "navigation"
            }
        
        return None
    
    def _retrieve_context(self, query: str, filters: Optional[Dict] = None) -> str:
        """Fast context retrieval"""
        query_embedding = self.embed_text(query)
        
        results = self.index.query(
            vector=query_embedding,
            top_k=3,  # Reduced for speed
            include_metadata=True,
            filter=filters
        )
        
        if not results.get('matches'):
            return ""
        
        return "\n\n".join([
            f"[{m['metadata'].get('source', '?')} p.{m['metadata'].get('page', '?')}] {m['metadata'].get('text', '')}"
            for m in results['matches']
        ])
    
    def _generate_answer(self, query: str, context: str) -> str:
        """Generate concise answer"""
        response = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a concise event assistant."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer briefly:"}
            ],
            temperature=0.7,
            max_tokens=300
        )
        
        return response.choices[0].message.content
    
    def _find_location(self, query: str) -> Optional[str]:
        """Fuzzy location matching"""
        query = query.lower().strip()
        
        if query in self.venue_map:
            return query
        
        for key in self.venue_map.keys():
            if query in key or key in query:
                return key
        
        return None
    
    # ==================== UTILITIES ====================
    
    def set_user_location(self, session_id: str, location: str) -> str:
        """Set user location"""
        loc = self._find_location(location)
        
        if loc:
            self.user_locations[session_id] = loc
            return f"✅ Location: {self.venue_map[loc]['name']}"
        
        available = ', '.join([v["name"] for v in list(self.venue_map.values())[:3]])
        return f"❌ Not found. Try: {available}..."
    
    def get_all_locations(self) -> List[str]:
        """Get all location names"""
        return [v["name"] for v in self.venue_map.values()]
    
    def get_floor_plans_summary(self) -> Dict:
        """Get floor plan summary"""
        return {
            "total_floors": len(self.floor_plans),
            "total_locations": len(self.venue_map),
            "locations": self.get_all_locations()
        }
    
    def get_statistics(self) -> Dict:
        """Get system stats"""
        try:
            stats = self.index.describe_index_stats()
            return {
                "total_vectors": stats.get('total_vector_count', 0),
                "dimension": stats.get('dimension', 0),
                "active_sessions": len(self.user_locations)
            }
        except:
            return {"total_vectors": 0, "dimension": 384, "active_sessions": 0}
    
    def clear_session(self, session_id: str) -> bool:
        """Clear session data"""
        if session_id in self.conversation_history:
            del self.conversation_history[session_id]
        if session_id in self.user_locations:
            del self.user_locations[session_id]
        return True