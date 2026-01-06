import os
import requests
import json
import base64
from groq import Groq
from ddgs import DDGS

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

    def generate_response(self, question, pokemon_data=None, web_results=None, context=None, user_profile=None):
        """
        UNIFIED RESPONSE GENERATOR - Groq gets ALL data sources:
        - pokemon_data: Dict with stats, types, weaknesses from database
        - web_results: List of web search results for lore/stories
        - context: Conversation context (last_pokemon, history)
        - user_profile: User preferences
        """
        if not self.groq_client:
            return "I'm having trouble connecting. Please try again!"

        # Build user context
        user_context = ""
        if user_profile:
            if user_profile.get('name'): 
                user_context += f"The user's name is {user_profile['name']}. "
            if user_profile.get('fav_pokemon'): 
                user_context += f"Their favorite Pokemon is {user_profile['fav_pokemon']}. "

        # Build Pokemon database info
        db_info = "No Pokemon data available."
        if pokemon_data:
            db_info = f"""
POKEMON DATABASE INFO:
- Name: {pokemon_data.get('name', 'Unknown')}
- Type: {pokemon_data.get('type1', '?')}{' / ' + pokemon_data['type2'] if pokemon_data.get('type2') else ''}
- Stats: HP {pokemon_data.get('hp', '?')}, Attack {pokemon_data.get('attack', '?')}, Defense {pokemon_data.get('defense', '?')}, Speed {pokemon_data.get('speed', '?')}
- Generation: {pokemon_data.get('generation', '?')}
- Legendary: {'Yes' if pokemon_data.get('legendary') else 'No'}
"""
            if pokemon_data.get('weak_to'):
                db_info += f"- Weak to: {', '.join(pokemon_data['weak_to'])}\n"
            if pokemon_data.get('strong_against'):
                db_info += f"- Strong against: {', '.join(pokemon_data['strong_against'])}\n"
            if pokemon_data.get('evolution'):
                db_info += f"- Evolution: {pokemon_data['evolution']}\n"
            if pokemon_data.get('similar_pokemon'):
                db_info += f"- Similar Pokemon: {', '.join(pokemon_data['similar_pokemon'])}\n"
            if pokemon_data.get('newly_learned'):
                db_info += "- (This Pokemon was just learned from PokeAPI!)\n"

        # Build web search results
        web_info = ""
        if web_results and len(web_results) > 0:
            web_info = "\nINTERNET SEARCH RESULTS (Lore/Stories):\n"
            for i, result in enumerate(web_results[:3], 1):
                title = result.get('title', 'Unknown')
                body = result.get('body', '')
                web_info += f"{i}. {title}: {body}\n"

        # Build conversation context
        context_info = ""
        if context:
            if context.get('last_pokemon'):
                context_info += f"Previously discussed Pokemon: {context['last_pokemon']}\n"

        # Determine response style based on question type
        story_keywords = ['story', 'lore', 'legend', 'history', 'origin', 'myth', 'tale']
        is_story = any(kw in question.lower() for kw in story_keywords)
        
        if is_story:
            style_instruction = "Tell an engaging story (3-4 sentences). Be immersive and exciting!"
        else:
            style_instruction = "Be concise (1-2 sentences). Get to the point quickly."

        # Build the MEGA prompt
        prompt = f"""You are Omnidex, a friendly and knowledgeable Pokemon expert. You're chatting with a friend about Pokemon.

RULES:
1. ONLY discuss Pokemon topics. Ignore anything non-Pokemon.
2. USE the data provided below - don't invent facts.
3. Talk casually and friendly. Never say "dear student" or be formal.
4. {style_instruction}
5. Use 1-2 relevant emoji (🔥⚡💧🌿 etc).
6. If you don't have info, admit it honestly.

{user_context}
{context_info}

=== ALL AVAILABLE DATA ===
{db_info}
{web_info}
===========================

USER'S QUESTION: "{question}"

Now respond naturally as Omnidex, using the data above:"""

        try:
            response = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are Omnidex, a friendly Pokemon expert. Be casual, helpful, and stick to Pokemon topics only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"❌ Groq error: {e}")
            # Fallback to simple response
            if pokemon_data:
                return f"{pokemon_data.get('name', 'This Pokemon')} is a {pokemon_data.get('type1', 'mystery')} type! ⚡"
            return "I'm having trouble right now. Try asking again!"

    # Keep old method for backwards compatibility
    def make_conversational(self, data, user_question, context_str="", topic="general", user_profile=None):
        """Legacy method - redirects to generate_response"""
        return self.generate_response(
            question=user_question,
            pokemon_data={'raw_info': str(data)} if data else None,
            web_results=None,
            context=None,
            user_profile=user_profile
        )

    def search_web(self, query, max_results=5):
        """DuckDuckGo Search with Pokemon-focused filtering"""
        # Make query more specific to Pokemon
        search_query = f"{query} site:bulbapedia.bulbagarden.net OR site:pokemon.fandom.com"
        print(f"🔍 Searching: {search_query}")
        try:
            ddgs = DDGS()
            results = list(ddgs.text(search_query, max_results=max_results))
            
            # Filter for Pokemon-related results only
            pokemon_keywords = ['pokemon', 'pokémon', 'bulbapedia', 'pokedex', 'trainer', 'legendary', 'evolution']
            filtered = []
            for r in results:
                title_lower = r.get('title', '').lower()
                body_lower = r.get('body', '').lower()
                # Check if result is Pokemon-related
                if any(kw in title_lower or kw in body_lower for kw in pokemon_keywords):
                    # Filter out spam
                    if 'pokemmo' not in title_lower and 'walkthrough' not in title_lower:
                        filtered.append(r)
            
            print(f"✅ Found {len(filtered)} Pokemon-related results")
            return filtered if filtered else results[:3]  # Fallback to top 3 if no filtered results
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
