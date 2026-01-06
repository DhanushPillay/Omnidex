import re
import os
from services.data_service import DataService
from services.intent_service import IntentService
from services.external_service import ExternalService

class PokemonChatbot:
    def __init__(self, csv_file):
        print("🤖 Initializing Omnidex Services...")
        self.data_service = DataService(csv_file)
        self.intent_service = IntentService()
        self.external_service = ExternalService()
        
        # Build Vector DB if semantic model is available
        if self.intent_service.use_semantic:
            self.data_service.build_vector_db(self.intent_service.semantic_model)
            
        # Shortcuts for compatibility
        self.df = self.data_service.df
        self.TYPE_CHART = self.data_service.type_chart

    def answer_question(self, question, context, user_profile=None):
        """Main entry point - Groq gets ALL data sources"""
        original_question = question
        question_lower = question.lower().strip()

        # 1. Profile / Greeting Checks
        if user_profile:
            name_match = re.search(r"(?:my name is|i'm|im|call me) ([A-Z][a-z]+)", original_question, re.IGNORECASE)
            if name_match:
                name = name_match.group(1)
                user_profile['name'] = name
                return f"Nice to meet you, {name}! 👋 I'll remember that."

        # Quick greetings
        greetings = ['hi', 'hello', 'hey', 'yo']
        if any(question_lower.startswith(g) for g in greetings) and len(question_lower) < 10:
            return self.external_service.generate_response(
                question=original_question,
                pokemon_data=None,
                web_results=None,
                context=context,
                user_profile=user_profile
            )

        # 2. Context Resolution - Replace pronouns with actual Pokemon name
        last_pokemon = context.get('last_pokemon')
        if last_pokemon:
            pronoun_patterns = ['its', "it's", 'it', 'this pokemon', 'this one', 'that pokemon', 'that one', 'this', 'that']
            for pattern in pronoun_patterns:
                if pattern in question_lower:
                    question_lower = question_lower.replace(pattern, last_pokemon.lower())
                    break

        # 3. Extract Pokemon name from question
        pokemon_name = self._extract_pokemon_name(question_lower)
        
        # If no name found, check if user is asking follow-up about last Pokemon
        if not pokemon_name and last_pokemon:
            pokemon_name = last_pokemon
        
        # 4. Gather ALL data for Groq
        pokemon_data = None
        web_results = None
        
        # Get Pokemon data from database
        if pokemon_name:
            context['last_pokemon'] = pokemon_name
            poke = self.data_service.get_pokemon_by_name(pokemon_name)
            
            if poke is not None:
                # Build comprehensive Pokemon data object
                pokemon_data = {
                    'name': poke['Name'],
                    'type1': poke['Type1'],
                    'type2': poke['Type2'] if poke['Type2'] else None,
                    'hp': int(poke['HP']),
                    'attack': int(poke['Attack']),
                    'defense': int(poke['Defense']),
                    'speed': int(poke['Speed']),
                    'generation': int(poke['Generation']),
                    'legendary': bool(poke['Legendary'])
                }
                
                # Add type effectiveness
                if poke['Type1'] in self.data_service.type_chart:
                    type_info = self.data_service.type_chart[poke['Type1']]
                    pokemon_data['weak_to'] = type_info.get('weak', [])
                    pokemon_data['strong_against'] = type_info.get('strong', [])
                
                # Add evolution data if available
                if poke['Name'] in self.data_service.evolution_data:
                    pokemon_data['evolution'] = self.data_service.evolution_data[poke['Name']]
                
                # Get similar Pokemon
                similar, _ = self.data_service.get_similar_pokemon(pokemon_name, n=3)
                if similar:
                    pokemon_data['similar_pokemon'] = [s['name'] for s in similar]
            else:
                # Try to learn from PokeAPI
                print(f"🧠 Unknown Pokemon '{pokemon_name}' - fetching from PokeAPI...")
                new_data = self.external_service.fetch_from_pokeapi(pokemon_name)
                if new_data:
                    self.data_service.add_pokemon(new_data)
                    pokemon_data = {
                        'name': new_data['name'],
                        'type1': new_data['types'][0],
                        'type2': new_data['types'][1] if len(new_data['types']) > 1 else None,
                        'hp': new_data['stats'].get('hp', 0),
                        'attack': new_data['stats'].get('attack', 0),
                        'defense': new_data['stats'].get('defense', 0),
                        'speed': new_data['stats'].get('speed', 0),
                        'newly_learned': True
                    }
        
        # 5. Search internet for lore/story/detailed info
        story_keywords = ['story', 'lore', 'legend', 'history', 'origin', 'myth', 'backstory', 'tale']
        needs_web_search = any(kw in question_lower for kw in story_keywords)
        
        # Also search if asking about something we might not have in DB
        general_keywords = ['why', 'how did', 'when was', 'who created', 'where does']
        if any(kw in question_lower for kw in general_keywords):
            needs_web_search = True
        
        if needs_web_search and pokemon_name:
            print(f"🌐 Searching web for: {pokemon_name}")
            search_query = f"{pokemon_name} Pokemon {' '.join([kw for kw in story_keywords if kw in question_lower])}"
            web_results = self.external_service.search_web(search_query)
        
        # 6. Let Groq synthesize EVERYTHING
        response = self.external_service.generate_response(
            question=original_question,
            pokemon_data=pokemon_data,
            web_results=web_results,
            context=context,
            user_profile=user_profile
        )
        
        return response

    def analyze_image(self, image_path):
        """Delegate to ExternalService"""
        return self.external_service.analyze_image(image_path)

    # --- Helpers ---
    def _extract_pokemon_name(self, text):
        words = text.split()
        for w in words:
            clean = re.sub(r'[^\w]', '', w)
            if len(clean) > 2:
                match = self.data_service.fuzzy_find_pokemon(clean)
                if match: return match
        return None

    def _format_pokemon_info(self, row):
        return f"{row['Name']}: {row['Type1']} type. HP: {row['HP']}, Atk: {row['Attack']}, Def: {row['Defense']}."

    # Compatibility methods for app.py
    def _get_sprite_url(self, name):
        return self.external_service.get_sprite_url(name)
    
    def _fuzzy_find_pokemon(self, name):
        return self.data_service.fuzzy_find_pokemon(name)
