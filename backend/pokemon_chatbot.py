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
        """Main entry point for chat"""
        original_question = question
        question_lower = question.lower().strip()

        # 1. Profile / Greeting Checks
        if user_profile:
            name_match = re.search(r"(?:my name is|i'm|im|call me) ([A-Z][a-z]+)", original_question, re.IGNORECASE)
            if name_match:
                name = name_match.group(1)
                user_profile['name'] = name
                return f"Nice to meet you, {name}! 👋 I'll remember that."

        # Greetings
        greetings = ['hi', 'hello', 'hey', 'yo']
        if any(question_lower.startswith(g) for g in greetings) and len(question_lower) < 10:
             return self.external_service.make_conversational(
                 "Hello! I am Omnidex. Ask me about Pokemon stats, lore, or recommendations.",
                 original_question, topic="general", user_profile=user_profile)

        # 2. Context Resolution
        resolved = self._resolve_pronoun(question_lower, context)
        if resolved:
            question_lower = question_lower.replace('it', resolved).replace('this pokemon', resolved)

        # 3. Intent Classification
        intent, confidence = self.intent_service.predict(question_lower)
        
        # Override for Lore
        if any(w in question_lower for w in ['story', 'lore', 'legend', 'history', 'origin']):
            intent = "pokemon_lore"

        print(f"🤖 Intent: {intent} (Conf: {confidence:.2f})")
        context['last_intent'] = intent
        
        # 4. Dispatch Logic
        response_data = ""
        topic = "general"

        # --- HANDLERS ---
        if intent == "pokemon_info":
            name = self._extract_pokemon_name(question_lower)
            if name:
                poke = self.data_service.get_pokemon_by_name(name)
                if poke is not None:
                    context['last_pokemon'] = name
                    response_data = self._format_pokemon_info(poke)
                    topic = "stats"
            else:
                # SELF-LEARNING: Try to find unknowns on valid API
                potential_name = None
                # Basic heuristic: look for capitalized words that aren't common knowns
                words = [w for w in question.split() if w[0].isupper() and len(w) > 3]
                if words:
                     potential_name = words[-1] # Take the last capitalized word as candidate
                
                if potential_name:
                    print(f"🧠 Unknown Pokemon '{potential_name}' detected. Learning...")
                    new_data = self.external_service.fetch_from_pokeapi(potential_name)
                    if new_data:
                        self.data_service.add_pokemon(new_data)
                        context['last_pokemon'] = new_data['name']
                        response_data = f"I didn't know about {new_data['name']}, but I just updated my Pokedex! ✨ It is a {new_data['types'][0]} type."
                        topic = "stats"
                    else:
                        response_data = "Which Pokemon are you asking about?"
                else:
                    response_data = "Which Pokemon are you asking about?"

        elif intent == "pokemon_lore":
            name = self._extract_pokemon_name(question_lower) or context.get('last_pokemon')
            if name:
                context['last_pokemon'] = name
                # Web Search for Lore
                results = self.external_service.search_web(f"{name} Pokemon lore history")
                response_data = "\n".join([f"- {r['body']}" for r in results])
                topic = "lore"
            else:
                response_data = "I can tell you stories! Which Pokemon?"

        elif intent == "recommend":
            name = self._extract_pokemon_name(question_lower)
            if name:
                similar, matched = self.data_service.get_similar_pokemon(name)
                if similar:
                    response_data = f"Pokemon similar to {matched}: " + ", ".join([s['name'] for s in similar])
                    topic = "recommendation"

        elif intent == "weakness":
            name = self._extract_pokemon_name(question_lower)
            if name:
                # Logic for weakness (can be moved to DataService, but doing here for brevity)
                poke = self.data_service.get_pokemon_by_name(name)
                if poke is not None:
                    t1 = poke['Type1']
                    weakmap = self.data_service.type_chart.get(t1, {}).get('weak', [])
                    response_data = f"{name} ({t1}) is weak to: {', '.join(weakmap)}"
                    topic = "battle"

        elif intent and intent.startswith("context_"):
            last_poke = context.get('last_pokemon')
            if last_poke:
                if "stat" in intent:
                    poke = self.data_service.get_pokemon_by_name(last_poke)
                    response_data = f"{last_poke} Stats: HP {poke['HP']}, Atk {poke['Attack']}, Def {poke['Defense']}."
                    topic = "stats"
                elif "evolution" in intent:
                    # Retrieve from evo json
                    chain = self.data_service.evolution_data.get(last_poke)
                    response_data = f"Evolution info for {last_poke}: {chain}"
                    topic = "evolution"
            else:
                response_data = "I'm not sure which Pokemon we are talking about."

        else:
            # Fallback / General Knowledge
            # SELF-LEARNING CHECK (Last Resort)
            potential_name = None
            
            # 1. Check for capitalized words (legacy)
            words = [w for w in question.split() if w[0].isupper() and len(w) > 3]
            if words:
                 potential_name = words[-1]
            
            # 2. If short query, assume it might be a name (even lowercase)
            # e.g., "lechonk", "who is lechonk"
            if not potential_name:
                clean_q = question.strip().lower()
                # Remove common prefixes
                for prefix in ["who is ", "what is ", "tell me about ", "show me "]:
                    if clean_q.startswith(prefix):
                        clean_q = clean_q.replace(prefix, "")
                
                # If what remains is a single word
                if " " not in clean_q.strip() and len(clean_q) > 3:
                     potential_name = clean_q.strip().title()

            if potential_name:
                print(f"🧠 Unknown Pokemon '{potential_name}' detected. Learning...")
                new_data = self.external_service.fetch_from_pokeapi(potential_name)
                if new_data:
                    self.data_service.add_pokemon(new_data)
                    context['last_pokemon'] = new_data['name']
                    # Recursive call now that we have it? Or just return success
                    response_data = f"I didn't know about {new_data['name']}, but I just updated my Pokedex! ✨ It is a {new_data['types'][0]} type."
                    return self.external_service.make_conversational(response_data, original_question, "", "stats", user_profile)

            return self.external_service.make_conversational(
                "I don't have specific data on that, but I can answer generally.", 
                original_question, str(context.get('conversation_history', [])), "general", user_profile)
        
        # 5. Generate Conversational Response
        if response_data:
            return self.external_service.make_conversational(response_data, original_question, "", topic, user_profile)
        
        return "I didn't quite understand that. Try asking about a specific Pokemon!"

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

    def _resolve_pronoun(self, text, context):
        if 'it' in text or 'this' in text:
            return context.get('last_pokemon')
        return None
    
    # Compatibility methods for app.py
    def _get_sprite_url(self, name):
        return self.external_service.get_sprite_url(name)
    
    def _fuzzy_find_pokemon(self, name):
        return self.data_service.fuzzy_find_pokemon(name)
