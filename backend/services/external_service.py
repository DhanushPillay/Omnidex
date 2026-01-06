import os
import requests
import json
import base64
from groq import Groq
from duckduckgo_search import DDGS

class ExternalService:
    def __init__(self, groq_api_key=None):
        self.groq_client = None
        self.api_key = groq_api_key or os.environ.get('GROQ_API_KEY')
        
        if self.api_key:
            try:
                self.groq_client = Groq(api_key=self.api_key)
                print("✅ ExternalService: Groq enabled")
            except Exception as e:
                print(f"⚠️ ExternalService error: {e}")
        else:
            print("⚠️ ExternalService: No GROQ_API_KEY found")

    def make_conversational(self, data, user_question, context_str="", topic="general", user_profile=None):
        """Standardized interface for generating responses with Persona & Adaptive Length"""
        if not self.groq_client:
            return str(data)  # Fallback

        # Simplify user context
        user_context = ""
        if user_profile:
            if user_profile.get('name'): user_context += f"User: {user_profile['name']}. "
            if user_profile.get('fav_pokemon'): user_context += f"Fav Pokemon: {user_profile['fav_pokemon']}."

        # Adaptive Instructions based on topic (LENGTH CONTROL)
        # Default: Short and punchy
        persona = "You are Omnidex, a friendly Pokemon expert and enthusiast. You talk like a helpful friend, not a teacher. Never say 'dear student' or address the user formally."
        length_instr = "Keep it concise (1-2 sentences). Be direct."
        
        if topic in ['lore', 'story', 'history', 'origin']:
            length_instr = "Be detailed and immersive. Tell a short story (3-4 sentences)."
            persona += " Share stories like an excited friend, not a lecturer."
        elif topic == 'battle': 
            length_instr = "Focus on strategy. Be analytical but brief."
        elif topic == 'evolution':
            length_instr = "Sound excited about growth! Keep it brief."

        prompt = f"""You are Omnidex, a friendly Pokemon expert.

IMPORTANT RULES:
1. You ONLY talk about Pokemon. Nothing else.
2. Use the DATA provided below - do not make up random information.
3. Talk casually like a friend, not a teacher.
4. {length_instr}
5. Use 1-2 relevant emoji.

USER CONTEXT: {user_context}
TOPIC: {topic.upper()}

USER QUESTION: "{user_question}"

POKEMON DATA TO USE:
{data}

Respond ONLY about the Pokemon topic. If the data is empty or irrelevant, say you don't have that information about this Pokemon."""
        try:
            response = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are Omnidex, a friendly Pokemon expert. Talk casually like a friend. Never use formal language like 'dear student'."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"❌ Groq Gen error: {e}")
            return str(data)

    def search_web(self, query, max_results=5):
        """DuckDuckGo Search"""
        print(f"🔍 Searching: {query}")
        try:
            with DDGS() as ddgs:
                # Basic text search
                results = list(ddgs.text(query, max_results=max_results))
                # Filter spam
                results = [r for r in results if "pokemmo" not in r['title'].lower()]
                return results
        except Exception as e:
            print(f"❌ Search error: {e}")
            return []

    def analyze_image(self, image_path):
        """Analyze image with Groq Vision (llama-3.2-90b-vision-preview)"""
        if not self.groq_client:
            return {"error": "Vision AI not enabled"}

        try:
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')

            prompt = """Identify the Pokemon in this image. 
Return strictly JSON: {"name": "Name", "description": "Brief desc", "is_shiny": bool}"""

            response = self.groq_client.chat.completions.create(
                model="llama-3.2-90b-vision-preview",
                messages=[
                    {
                        "role": "user", 
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        ]
                    }
                ],
                max_tokens=300
            )
            text = response.choices[0].message.content.strip()
            # Clean markdown
            if text.startswith('```'): text = text.replace('```json','').replace('```','')
            
            return json.loads(text)
        except Exception as e:
            print(f"❌ Vision error: {e}")
            return {"error": str(e)}

    def fetch_from_pokeapi(self, name):
        """Fetch raw data from PokeAPI"""
        try:
            clean_name = name.lower().replace('.', '').replace("'", '').replace(' ', '-')
            url = f"https://pokeapi.co/api/v2/pokemon/{clean_name}"
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                # Process basic data
                return {
                    'name': data['name'].title(),
                    'types': [t['type']['name'].title() for t in data['types']],
                    'stats': {s['stat']['name']: s['base_stat'] for s in data['stats']},
                    'id': data['id']
                }
        except:
            pass
        return None
    
    def get_sprite_url(self, name):
        """Get sprite URL (simple)"""
        if not name: return None
        # Quick heuristic or API fetch
        data = self.fetch_from_pokeapi(name)
        if data:
            return f"https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/other/official-artwork/{data['id']}.png"
        return None
