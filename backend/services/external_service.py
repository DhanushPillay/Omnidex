import os
import requests
import json
import base64
from openai import OpenAI
from duckduckgo_search import DDGS

class ExternalService:
    def __init__(self, openai_api_key=None):
        self.openai_client = None
        self.api_key = openai_api_key or os.environ.get('OPENAI_API_KEY')
        
        if self.api_key:
            try:
                self.openai_client = OpenAI(api_key=self.api_key)
                print("✅ ExternalService: OpenAI enabled")
            except Exception as e:
                print(f"⚠️ ExternalService error: {e}")
        else:
            print("⚠️ ExternalService: No OPENAI_API_KEY found")

    def make_conversational(self, data, user_question, context_str="", topic="general", user_profile=None):
        """Standardized interface for generating responses"""
        if not self.openai_client:
            return str(data) # Fallback

        # Simplify user context
        user_context = ""
        if user_profile:
            if user_profile.get('name'): user_context += f"User: {user_profile['name']}. "
            if user_profile.get('fav_pokemon'): user_context += f"Fav Pokemon: {user_profile['fav_pokemon']}."

        # Instructions based on topic
        instructions = "Be friendly and enthusiastic."
        if topic == 'battle': instructions = "Focus on strategy and type matchups."
        elif topic == 'lore': instructions = "Be a storyteller. Tell myths and legends in detail."
        elif topic == 'stats': instructions = "Be analytical about the stats."
        elif topic == 'evolution': instructions = "Focus on growth and potential."

        prompt = f"""You are Omnidex, an AI Pokemon Professor.
CONTEXT: {user_context}
TOPIC: {topic.upper()}
PREVIOUS CHAT:
{context_str}

USER ASKED: "{user_question}"

DATA/INFO TO USE:
{data}

INSTRUCTIONS:
{instructions}
- Use 1-2 relevant emoji.
- Keep it natural, NO raw JSON/Markdown tables unless necessary.
- If it's a story/lore request, be detailed.
"""
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are Omnidex, a Pokemon expert."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"❌ OpenAI Gen error: {e}")
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
        """Analyze image with GPT-4o"""
        if not self.openai_client:
            return {"error": "Vision AI not enabled"}

        try:
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')

            prompt = """Identify the Pokemon in this image. 
Return strictly JSON: {"name": "Name", "description": "Brief desc", "is_shiny": bool}"""

            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
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
