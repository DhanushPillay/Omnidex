import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from difflib import get_close_matches
import re
import os
import json
import requests
from duckduckgo_search import DDGS
from google import genai
import faiss
from PIL import Image
import io

# Try to import sentence-transformers for better NLP
USE_SEMANTIC_NLP = False
try:
    from sentence_transformers import SentenceTransformer
    USE_SEMANTIC_NLP = True
except ImportError:
    print("⚠️ sentence-transformers not installed. Using TF-IDF fallback.")

# Configure Gemini API - uses free tier
# IMPORTANT: Set GEMINI_API_KEY environment variable for security
# GEMINI_API_KEY is handled by the client initialization below
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '')


class PokemonChatbot:
    # Type effectiveness chart - what each type is weak/strong against
    TYPE_CHART = {
        'Normal': {'weak': ['Fighting'], 'immune': ['Ghost'], 'resist': [], 'strong': []},
        'Fire': {'weak': ['Water', 'Ground', 'Rock'], 'resist': ['Fire', 'Grass', 'Ice', 'Bug', 'Steel', 'Fairy'], 'immune': [], 'strong': ['Grass', 'Ice', 'Bug', 'Steel']},
        'Water': {'weak': ['Electric', 'Grass'], 'resist': ['Fire', 'Water', 'Ice', 'Steel'], 'immune': [], 'strong': ['Fire', 'Ground', 'Rock']},
        'Electric': {'weak': ['Ground'], 'resist': ['Electric', 'Flying', 'Steel'], 'immune': [], 'strong': ['Water', 'Flying']},
        'Grass': {'weak': ['Fire', 'Ice', 'Poison', 'Flying', 'Bug'], 'resist': ['Water', 'Electric', 'Grass', 'Ground'], 'immune': [], 'strong': ['Water', 'Ground', 'Rock']},
        'Ice': {'weak': ['Fire', 'Fighting', 'Rock', 'Steel'], 'resist': ['Ice'], 'immune': [], 'strong': ['Grass', 'Ground', 'Flying', 'Dragon']},
        'Fighting': {'weak': ['Flying', 'Psychic', 'Fairy'], 'resist': ['Bug', 'Rock', 'Dark'], 'immune': [], 'strong': ['Normal', 'Ice', 'Rock', 'Dark', 'Steel']},
        'Poison': {'weak': ['Ground', 'Psychic'], 'resist': ['Grass', 'Fighting', 'Poison', 'Bug', 'Fairy'], 'immune': [], 'strong': ['Grass', 'Fairy']},
        'Ground': {'weak': ['Water', 'Grass', 'Ice'], 'resist': ['Poison', 'Rock'], 'immune': ['Electric'], 'strong': ['Fire', 'Electric', 'Poison', 'Rock', 'Steel']},
        'Flying': {'weak': ['Electric', 'Ice', 'Rock'], 'resist': ['Grass', 'Fighting', 'Bug'], 'immune': ['Ground'], 'strong': ['Grass', 'Fighting', 'Bug']},
        'Psychic': {'weak': ['Bug', 'Ghost', 'Dark'], 'resist': ['Fighting', 'Psychic'], 'immune': [], 'strong': ['Fighting', 'Poison']},
        'Bug': {'weak': ['Fire', 'Flying', 'Rock'], 'resist': ['Grass', 'Fighting', 'Ground'], 'immune': [], 'strong': ['Grass', 'Psychic', 'Dark']},
        'Rock': {'weak': ['Water', 'Grass', 'Fighting', 'Ground', 'Steel'], 'resist': ['Normal', 'Fire', 'Poison', 'Flying'], 'immune': [], 'strong': ['Fire', 'Ice', 'Flying', 'Bug']},
        'Ghost': {'weak': ['Ghost', 'Dark'], 'resist': ['Poison', 'Bug'], 'immune': ['Normal', 'Fighting'], 'strong': ['Psychic', 'Ghost']},
        'Dragon': {'weak': ['Ice', 'Dragon', 'Fairy'], 'resist': ['Fire', 'Water', 'Electric', 'Grass'], 'immune': [], 'strong': ['Dragon']},
        'Dark': {'weak': ['Fighting', 'Bug', 'Fairy'], 'resist': ['Ghost', 'Dark'], 'immune': ['Psychic'], 'strong': ['Psychic', 'Ghost']},
        'Steel': {'weak': ['Fire', 'Fighting', 'Ground'], 'resist': ['Normal', 'Grass', 'Ice', 'Flying', 'Psychic', 'Bug', 'Rock', 'Dragon', 'Steel', 'Fairy'], 'immune': ['Poison'], 'strong': ['Ice', 'Rock', 'Fairy']},
        'Fairy': {'weak': ['Poison', 'Steel'], 'resist': ['Fighting', 'Bug', 'Dark'], 'immune': ['Dragon'], 'strong': ['Fighting', 'Dragon', 'Dark']},
    }
    
    # Evolution chains for popular Pokemon
    EVOLUTION_DATA = {
        'Bulbasaur': {'evolves_to': 'Ivysaur', 'level': 16, 'method': 'level'},
        'Ivysaur': {'evolves_to': 'Venusaur', 'level': 32, 'method': 'level', 'evolves_from': 'Bulbasaur'},
        'Venusaur': {'evolves_from': 'Ivysaur'},
        'Charmander': {'evolves_to': 'Charmeleon', 'level': 16, 'method': 'level'},
        'Charmeleon': {'evolves_to': 'Charizard', 'level': 36, 'method': 'level', 'evolves_from': 'Charmander'},
        'Charizard': {'evolves_from': 'Charmeleon'},
        'Squirtle': {'evolves_to': 'Wartortle', 'level': 16, 'method': 'level'},
        'Wartortle': {'evolves_to': 'Blastoise', 'level': 36, 'method': 'level', 'evolves_from': 'Squirtle'},
        'Blastoise': {'evolves_from': 'Wartortle'},
        'Pichu': {'evolves_to': 'Pikachu', 'method': 'friendship'},
        'Pikachu': {'evolves_to': 'Raichu', 'method': 'Thunder Stone', 'evolves_from': 'Pichu'},
        'Raichu': {'evolves_from': 'Pikachu'},
        'Eevee': {'evolves_to': ['Vaporeon (Water Stone)', 'Jolteon (Thunder Stone)', 'Flareon (Fire Stone)', 'Espeon (Friendship + Day)', 'Umbreon (Friendship + Night)', 'Leafeon (Leaf Stone)', 'Glaceon (Ice Stone)', 'Sylveon (Affection + Fairy move)'], 'method': 'multiple'},
        'Magikarp': {'evolves_to': 'Gyarados', 'level': 20, 'method': 'level'},
        'Gyarados': {'evolves_from': 'Magikarp'},
        'Dratini': {'evolves_to': 'Dragonair', 'level': 30, 'method': 'level'},
        'Dragonair': {'evolves_to': 'Dragonite', 'level': 55, 'method': 'level', 'evolves_from': 'Dratini'},
        'Dragonite': {'evolves_from': 'Dragonair'},
        'Gastly': {'evolves_to': 'Haunter', 'level': 25, 'method': 'level'},
        'Haunter': {'evolves_to': 'Gengar', 'method': 'trade', 'evolves_from': 'Gastly'},
        'Gengar': {'evolves_from': 'Haunter'},
        'Abra': {'evolves_to': 'Kadabra', 'level': 16, 'method': 'level'},
        'Kadabra': {'evolves_to': 'Alakazam', 'method': 'trade', 'evolves_from': 'Abra'},
        'Alakazam': {'evolves_from': 'Kadabra'},
        'Machop': {'evolves_to': 'Machoke', 'level': 28, 'method': 'level'},
        'Machoke': {'evolves_to': 'Machamp', 'method': 'trade', 'evolves_from': 'Machop'},
        'Machamp': {'evolves_from': 'Machoke'},
        'Geodude': {'evolves_to': 'Graveler', 'level': 25, 'method': 'level'},
        'Graveler': {'evolves_to': 'Golem', 'method': 'trade', 'evolves_from': 'Geodude'},
        'Golem': {'evolves_from': 'Graveler'},
    }

    def __init__(self, csv_file):
        """Initialize the Pokemon chatbot with ML-powered features"""
        self.csv_file = csv_file  # Store path for self-learning updates
        self.df = pd.read_csv(csv_file)
        self.df['Type2'] = self.df['Type2'].fillna('')  # Replace NaN with empty string
        
        # Store Pokemon names for fuzzy matching
        self.pokemon_names = self.df['Name'].tolist()
        
        # Enhanced Conversation Memory - Multi-turn context is now passed per request
        # self.conversation_context removed to be stateless
        
        # Self-learning: cache for learned data from web
        self.learned_data = {}
        self._load_learned_data()
        
        # Initialize ML components
        self._init_intent_classifier()
        self._init_recommendation_system()
        self._init_semantic_model()  # New: Semantic understanding
        
        # Initialize Vector DB for RAG
        self.faiss_index = None
        self.pokemon_metadata = []
        self._build_vector_db()

        # Initialize Gemini model
        self.gemini_model = None
        if GEMINI_API_KEY:
            try:
                self.gemini_client = genai.Client(api_key=GEMINI_API_KEY)
                print("✅ Gemini AI enabled for conversational responses!")
            except Exception as e:
                print(f"⚠️ Gemini initialization failed: {e}")
        else:
            print("⚠️ No GEMINI_API_KEY found. Set it for AI-powered responses.")
            print("   Get a free key at: https://makersuite.google.com/app/apikey")
        
        print(f"✅ Loaded {len(self.df)} Pokemon from database!")
        print(f"✅ ML Intent Classifier trained with {len(self.intent_examples)} examples")
        print(f"✅ Type Effectiveness Chart loaded (18 types)")
        print(f"✅ Evolution Data for {len(self.EVOLUTION_DATA)} Pokemon")
        print(f"✅ KNN Recommendation System ready")
        print(f"✅ Semantic Understanding: ENABLED")
        print(f"✅ Multi-Turn Memory: ENABLED")
        print(f"✅ Self-Learning Mode: ENABLED (PokeAPI + Web Search)")
    
    def _init_intent_classifier(self):
        """Initialize TF-IDF based intent classifier with natural language examples"""
        # Training examples for each intent - covers many natural ways of asking
        self.intent_examples = [
            # ============ POKEMON INFO - Natural ways to ask about a Pokemon ============
            ("tell me about pikachu", "pokemon_info"),
            ("i want to know about charizard", "pokemon_info"),
            ("what is charizard", "pokemon_info"),
            ("who is mewtwo", "pokemon_info"),
            ("whats pikachu", "pokemon_info"),
            ("what's bulbasaur", "pokemon_info"),
            ("can you tell me about dragonite", "pokemon_info"),
            ("i wanna know about gengar", "pokemon_info"),
            ("give me info on lucario", "pokemon_info"),
            ("information about eevee", "pokemon_info"),
            ("describe snorlax", "pokemon_info"),
            ("stats for gyarados", "pokemon_info"),
            ("show me pikachu", "pokemon_info"),
            ("pikachu stats", "pokemon_info"),
            ("charizard info", "pokemon_info"),
            ("details about mewtwo", "pokemon_info"),
            ("tell me something about mew", "pokemon_info"),
            ("i want details on blastoise", "pokemon_info"),
            ("what do you know about venusaur", "pokemon_info"),
            ("give me details about alakazam", "pokemon_info"),
            ("info on machamp", "pokemon_info"),
            ("what can you tell me about arcanine", "pokemon_info"),
            ("who is dragonite", "pokemon_info"),
            ("explain gengar to me", "pokemon_info"),
            ("what about lapras", "pokemon_info"),
            ("how about magikarp", "pokemon_info"),
            ("and what about jigglypuff", "pokemon_info"),
            
            # ============ TYPE QUERIES - Natural ways to ask about types ============
            ("list all fire type pokemon", "type_query"),
            ("show me water type", "type_query"),
            ("what are the grass type pokemon", "type_query"),
            ("which pokemon are fire type", "type_query"),
            ("i want to see electric types", "type_query"),
            ("show electric pokemon", "type_query"),
            ("give me all dragon types", "type_query"),
            ("what pokemon are psychic type", "type_query"),
            ("name all ice type pokemon", "type_query"),
            ("who are the fighting types", "type_query"),
            ("list poison pokemon", "type_query"),
            ("ground type pokemon please", "type_query"),
            ("i need rock type pokemon", "type_query"),
            ("ghost pokemon list", "type_query"),
            ("dark type pokemon", "type_query"),
            ("steel types", "type_query"),
            ("fairy type pokemon", "type_query"),
            ("normal type pokemon", "type_query"),
            ("bug types", "type_query"),
            ("flying pokemon", "type_query"),
            ("all the water pokemon", "type_query"),
            ("fire pokemon", "type_query"),
            
            # ============ COUNT QUERIES ============
            ("how many legendary pokemon are there", "count_legendary"),
            ("count the legendary pokemon", "count_legendary"),
            ("number of legendary", "count_legendary"),
            ("how many fire type pokemon", "count_type"),
            ("count water type", "count_type"),
            ("how many water pokemon are there", "count_type"),
            ("number of grass types", "count_type"),
            ("how many pokemon total", "count_total"),
            ("total pokemon count", "count_total"),
            ("how many pokemon do you have", "count_total"),
            ("how many pokemon in database", "count_total"),
            
            # ============ STAT LEADERS ============
            ("who has the highest attack", "stat_leader"),
            ("strongest attack pokemon", "stat_leader"),
            ("who has the best defense", "stat_leader"),
            ("fastest pokemon", "stat_leader"),
            ("which pokemon is fastest", "stat_leader"),
            ("highest hp pokemon", "stat_leader"),
            ("most hp", "stat_leader"),
            ("best attack", "stat_leader"),
            ("highest defense", "stat_leader"),
            ("who has max speed", "stat_leader"),
            ("pokemon with highest attack", "stat_leader"),
            ("who is the strongest", "strongest_overall"),
            ("most powerful pokemon", "strongest_overall"),
            ("best overall stats", "strongest_overall"),
            ("strongest pokemon overall", "strongest_overall"),
            ("who is the best pokemon", "strongest_overall"),
            ("which pokemon is most powerful", "strongest_overall"),
            
            # ============ LEGENDARY QUERIES ============
            ("list all legendary pokemon", "list_legendary"),
            ("show me legendary pokemon", "list_legendary"),
            ("name all legendary", "list_legendary"),
            ("which pokemon are legendary", "list_legendary"),
            ("i want to see legendary pokemon", "list_legendary"),
            ("legendary ones", "list_legendary"),
            ("show legendaries", "list_legendary"),
            ("all legendary", "list_legendary"),
            ("who are the legendaries", "list_legendary"),
            
            # ============ GENERATION QUERIES ============
            ("show me generation 1 pokemon", "generation_query"),
            ("list gen 2 pokemon", "generation_query"),
            ("pokemon from generation 3", "generation_query"),
            ("gen 1 pokemon", "generation_query"),
            ("first generation pokemon", "generation_query"),
            ("generation 4 pokemon", "generation_query"),
            ("show gen 5", "generation_query"),
            ("which pokemon are from gen 1", "generation_query"),
            ("original pokemon", "generation_query"),
            
            # ============ COMPARISON QUERIES ============
            ("compare pikachu and raichu", "compare"),
            ("who is better charizard or blastoise", "compare"),
            ("pikachu vs charizard", "compare"),
            ("charizard versus blastoise", "compare"),
            ("which is stronger pikachu or raichu", "compare"),
            ("battle between mewtwo and mew", "compare"),
            ("compare stats of dragonite and salamence", "compare"),
            ("who would win pikachu or eevee", "compare"),
            ("difference between charizard and dragonite", "compare"),
            
            # ============ DUAL TYPE QUERIES ============
            ("show dual type pokemon", "dual_type"),
            ("list pokemon with two types", "dual_type"),
            ("which pokemon have dual types", "dual_type"),
            ("pokemon with multiple types", "dual_type"),
            ("two type pokemon", "dual_type"),
            
            # ============ RECOMMENDATION QUERIES ============
            ("recommend pokemon like pikachu", "recommend"),
            ("similar pokemon to charizard", "recommend"),
            ("pokemon similar to gengar", "recommend"),
            ("find pokemon like mewtwo", "recommend"),
            ("suggest pokemon like dragonite", "recommend"),
            ("who is similar to pikachu", "recommend"),
            ("pokemon that are like eevee", "recommend"),
            ("i want pokemon similar to lucario", "recommend"),
            ("anything like snorlax", "recommend"),
            ("show me pokemon like gyarados", "recommend"),
            ("alternatives to charizard", "recommend"),
            
            # ============ TYPE WEAKNESS/STRENGTH QUERIES ============
            ("what is charizard weak to", "weakness"),
            ("what are pikachu weaknesses", "weakness"),
            ("pikachu weakness", "weakness"),
            ("what is fire type weak against", "weakness"),
            ("water type weakness", "weakness"),
            ("what beats fire type", "weakness"),
            ("what is super effective against water", "weakness"),
            ("charizard weaknesses", "weakness"),
            ("what type beats dragon", "weakness"),
            ("what is dragon weak to", "weakness"),
            ("weaknesses of mewtwo", "weakness"),
            ("what can beat pikachu", "weakness"),
            ("what is strong against grass", "strength"),
            ("what type is good against water", "strength"),
            ("fire type strengths", "strength"),
            ("what does electric beat", "strength"),
            ("pikachu is strong against what", "strength"),
            ("charizard is super effective against", "strength"),
            
            # ============ EVOLUTION QUERIES ============
            ("how does pikachu evolve", "evolution"),
            ("what does charmander evolve into", "evolution"),
            ("pikachu evolution", "evolution"),
            ("eevee evolutions", "evolution"),
            ("how to evolve eevee", "evolution"),
            ("what evolves into charizard", "evolution"),
            ("evolution of bulbasaur", "evolution"),
            ("does pikachu evolve", "evolution"),
            ("magikarp evolution", "evolution"),
            ("how does gastly evolve", "evolution"),
            ("what level does charmander evolve", "evolution"),
            ("dragonite evolution chain", "evolution"),
            
            # ============ CONTEXT-AWARE QUERIES (follow-up questions) ============
            ("what about its defense", "context_stat"),
            ("and its attack", "context_stat"),
            ("how about speed", "context_stat"),
            ("what type is it", "context_type"),
            ("is it legendary", "context_legendary"),
            ("what is it weak to", "context_weakness"),
            ("how does it evolve", "context_evolution"),
            ("show me similar ones", "context_recommend"),
            ("any pokemon like it", "context_recommend"),
            # Story/lore follow-ups
            ("tell me its story", "context_story"),
            ("what is its lore", "context_story"),
            ("tell me more about it", "context_story"),
            ("its backstory", "context_story"),
            ("what is the story behind it", "context_story"),
            ("give me its history", "context_story"),
            ("how was it found", "context_story"),
            ("how did they find it", "context_story"),
            ("where does it come from", "context_story"),
            ("what is its origin", "context_story"),
            ("origin story", "context_story"),
            ("how was it created", "context_story"),
            ("who created it", "context_story"),
            ("how was it discovered", "context_story"),
            ("where did it come from", "context_story"),
            ("its origin", "context_story"),
        ]
        
        self.intent_texts = [ex[0] for ex in self.intent_examples]
        self.intent_labels = [ex[1] for ex in self.intent_examples]
        
        # Use sentence-transformers if available (much better NLP!)
        if USE_SEMANTIC_NLP:
            try:
                print("🧠 Loading Semantic NLP Model (this may take a moment)...")
                self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
                self.intent_embeddings = self.semantic_model.encode(self.intent_texts)
                self.use_semantic = True
                print("✅ Semantic NLP Model loaded! (all-MiniLM-L6-v2)")
            except Exception as e:
                print(f"⚠️ Semantic model failed, using TF-IDF: {e}")
                self.use_semantic = False
        else:
            self.use_semantic = False
        
        # TF-IDF fallback
        if not getattr(self, 'use_semantic', False):
            self.intent_vectorizer = TfidfVectorizer(
                lowercase=True,
                stop_words='english',
                ngram_range=(1, 2)
            )
            self.intent_vectors = self.intent_vectorizer.fit_transform(self.intent_texts)
    
    def _init_recommendation_system(self):
        """Initialize K-Nearest Neighbors for Pokemon recommendations with Hybrid Features"""
        # 1. Stats Features (normalized)
        stat_columns = ['HP', 'Attack', 'Defense', 'Speed']
        stat_features = self.df[stat_columns].values
        stat_mean = stat_features.mean(axis=0)
        stat_std = stat_features.std(axis=0)
        normalized_stats = (stat_features - stat_mean) / stat_std

        # 2. Type Features (One-Hot Encoding)
        # Get all unique types
        all_types = sorted(list(self.TYPE_CHART.keys()))
        type_features = []

        for _, row in self.df.iterrows():
            # Create binary vector for types [0, 1, 0, ...]
            t_vec = [0] * len(all_types)
            if row['Type1'] in all_types:
                t_vec[all_types.index(row['Type1'])] = 1
            if row['Type2'] in all_types:
                t_vec[all_types.index(row['Type2'])] = 1
            type_features.append(t_vec)

        type_features = np.array(type_features)
        
        # 3. Generation Feature (Normalized)
        gen_features = self.df['Generation'].values.reshape(-1, 1)
        gen_max = gen_features.max()
        normalized_gen = gen_features / gen_max

        # 4. Legendary Feature (Binary)
        legendary_features = self.df['Legendary'].astype(int).values.reshape(-1, 1)

        # Combine all features
        # We weight them: Stats (1.0), Types (1.5), Gen (0.5), Legendary (1.0)
        self.combined_features = np.hstack([
            normalized_stats * 1.0,
            type_features * 1.5,     # Give more weight to Type similarity
            normalized_gen * 0.5,
            legendary_features * 1.0
        ])
        
        # Create KNN model
        self.knn_model = NearestNeighbors(n_neighbors=6, metric='euclidean')
        self.knn_model.fit(self.combined_features)

        # Store for later use
        self.normalized_features = self.combined_features
    
    def _init_semantic_model(self):
        """Initialize semantic model for advanced understanding"""
        # Semantic model is already initialized in intent classifier
        # This method handles additional semantic features
        
        # Pre-compute Pokemon name embeddings for fuzzy semantic matching
        if getattr(self, 'use_semantic', False) and hasattr(self, 'semantic_model'):
            try:
                # Encode all Pokemon names for semantic matching
                self.pokemon_embeddings = self.semantic_model.encode(self.pokemon_names)
                print("✅ Pokemon name embeddings computed for semantic matching!")
            except:
                self.pokemon_embeddings = None
        else:
            self.pokemon_embeddings = None

    def _build_vector_db(self):
        """Build FAISS vector database from Pokemon data for RAG"""
        try:
            if not getattr(self, 'use_semantic', False) or not hasattr(self, 'semantic_model'):
                print("⚠️ Semantic model not available. Skipping Vector DB build.")
                return

            print("🏗️ Building Vector Database...")
            self.pokemon_metadata = []
            descriptions = []

            for _, row in self.df.iterrows():
                # Create a rich textual description
                type_str = row['Type1']
                if row['Type2']:
                    type_str += f" and {row['Type2']}"

                legendary = "Legendary" if row['Legendary'] else ""

                desc = f"{row['Name']} is a {type_str} type Pokemon from Generation {row['Generation']}. "
                desc += f"It has {row['HP']} HP, {row['Attack']} Attack, {row['Defense']} Defense, and {row['Speed']} Speed. "
                if legendary:
                    desc += f"It is a Legendary Pokemon. "

                # Add weaknesses/strengths context
                if row['Type1'] in self.TYPE_CHART:
                    weak = self.TYPE_CHART[row['Type1']]['weak']
                    strong = self.TYPE_CHART[row['Type1']]['strong']
                    desc += f"It is weak against {', '.join(weak)} and strong against {', '.join(strong)}. "

                descriptions.append(desc)
                self.pokemon_metadata.append({
                    'name': row['Name'],
                    'description': desc,
                    'type': type_str,
                    'stats': {k: row[k] for k in ['HP', 'Attack', 'Defense', 'Speed']}
                })

            # Encode all descriptions
            embeddings = self.semantic_model.encode(descriptions)

            # Initialize FAISS index
            dimension = embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatL2(dimension)
            self.faiss_index.add(np.array(embeddings).astype('float32'))

            print(f"✅ Vector Database built with {len(descriptions)} entries!")

        except Exception as e:
            print(f"❌ Failed to build Vector DB: {e}")

    def _vector_search(self, query, k=3):
        """Search the vector database for relevant Pokemon context"""
        if not self.faiss_index or not hasattr(self, 'semantic_model'):
            return []

        try:
            # Encode query
            query_vector = self.semantic_model.encode([query])

            # Search
            distances, indices = self.faiss_index.search(np.array(query_vector).astype('float32'), k)

            results = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.pokemon_metadata):
                    item = self.pokemon_metadata[idx]
                    results.append({
                        'name': item['name'],
                        'description': item['description'],
                        'score': float(distances[0][i])
                    })

            return results
        except Exception as e:
            print(f"⚠️ Vector search failed: {e}")
            return []
    
    # ============ CONVERSATION MEMORY METHODS ============
    
    def _add_to_history(self, context, role, content, pokemon=None, intent=None):
        """Add an exchange to conversation history"""
        entry = {
            'role': role,  # 'user' or 'bot'
            'content': content[:200],  # Truncate for memory
            'pokemon': pokemon,
            'intent': intent
        }
        context['conversation_history'].append(entry)
        
        # Keep only last 10 exchanges (sliding window)
        if len(context['conversation_history']) > 10:
            context['conversation_history'] = \
                context['conversation_history'][-10:]
        
        # Track mentioned Pokemon
        if pokemon and pokemon not in context['mentioned_pokemon']:
            context['mentioned_pokemon'].append(pokemon)
            # Keep last 5 mentioned Pokemon
            if len(context['mentioned_pokemon']) > 5:
                context['mentioned_pokemon'] = \
                    context['mentioned_pokemon'][-5:]
    
    def _get_context_summary(self, context):
        """Get a summary of recent conversation for AI context"""
        history = context['conversation_history']
        if not history:
            return ""
        
        summary_parts = []
        for entry in history[-5:]:  # Last 5 exchanges
            role = "User" if entry['role'] == 'user' else "Omnidex"
            content = entry['content'][:100]
            if entry['pokemon']:
                summary_parts.append(f"{role} (about {entry['pokemon']}): {content}")
            else:
                summary_parts.append(f"{role}: {content}")
        
        return "\n".join(summary_parts)
    
    def _resolve_pronoun(self, text, context):
        """Resolve pronouns like 'it', 'this one', 'that Pokemon' to actual names"""
        pronouns = ['it', 'its', 'this one', 'that one', 'this pokemon', 'that pokemon', 
                    'the first one', 'the second one', 'the first', 'the second']
        text_lower = text.lower()
        
        for pronoun in pronouns:
            if pronoun in text_lower:
                # 'first' refers to first in comparison, 'second' to second
                if 'first' in pronoun and len(context.get('compared_pokemon', [])) >= 1:
                    return context['compared_pokemon'][0]
                elif 'second' in pronoun and len(context.get('compared_pokemon', [])) >= 2:
                    return context['compared_pokemon'][1]
                # Otherwise use last mentioned Pokemon
                elif context.get('last_pokemon'):
                    return context['last_pokemon']
        
        return None
    
    def _detect_topic(self, intent, question):
        """Detect the current conversation topic"""
        topic_map = {
            'pokemon_info': 'stats',
            'weakness': 'battle',
            'strength': 'battle',
            'compare': 'battle',
            'evolution': 'evolution',
            'context_evolution': 'evolution',
            'context_story': 'lore',
            'similar': 'recommendation',
            'type_query': 'types'
        }
        return topic_map.get(intent, 'general')
    
    def _classify_intent(self, question):
        """Use Semantic Embeddings or TF-IDF to classify user intent"""
        question_lower = question.lower()
        
        # Use semantic embeddings if available (much better!)
        if getattr(self, 'use_semantic', False):
            question_embedding = self.semantic_model.encode([question_lower])
            similarities = cosine_similarity(question_embedding, self.intent_embeddings).flatten()
        else:
            # TF-IDF fallback
            question_vector = self.intent_vectorizer.transform([question_lower])
            similarities = cosine_similarity(question_vector, self.intent_vectors).flatten()
        
        # Get the best match
        best_idx = similarities.argmax()
        best_score = similarities[best_idx]
        
        # Semantic embeddings have different threshold
        threshold = 0.5 if getattr(self, 'use_semantic', False) else 0.2
        
        if best_score > threshold:
            return self.intent_labels[best_idx], best_score
        
        return None, 0.0
    
    def _fuzzy_find_pokemon(self, name, cutoff=0.75):
        """Find Pokemon name using fuzzy matching (handles typos!)"""
        name_clean = name.strip()
        
        # 1. First try EXACT match (case-insensitive)
        for pokemon_name in self.pokemon_names:
            if pokemon_name.lower() == name_clean.lower():
                return pokemon_name
        
        # 2. Try fuzzy matching with higher cutoff for accuracy
        matches = get_close_matches(name_clean.capitalize(), self.pokemon_names, n=1, cutoff=cutoff)
        if matches:
            return matches[0]
        
        # 3. Try partial match (name contains search term)
        name_lower = name_clean.lower()
        for pokemon_name in self.pokemon_names:
            if pokemon_name.lower().startswith(name_lower):
                return pokemon_name
        
        # 4. Try PokeAPI for unknown Pokemon (self-learning!)
        try:
            # Handle Mega evolutions: "Charizard X" -> "charizard-mega-x"
            clean_api_name = name_lower.replace(' ', '-').replace('.', '').replace("'", '')
            
            # Check for Mega evolution patterns
            if ' x' in name_lower or '-x' in name_lower:
                base_name = name_lower.replace(' x', '').replace('-x', '')
                clean_api_name = f"{base_name}-mega-x"
            elif ' y' in name_lower or '-y' in name_lower:
                base_name = name_lower.replace(' y', '').replace('-y', '')
                clean_api_name = f"{base_name}-mega-y"
            elif 'mega ' in name_lower:
                base_name = name_lower.replace('mega ', '')
                clean_api_name = f"{base_name}-mega"
            
            response = requests.get(f"https://pokeapi.co/api/v2/pokemon/{clean_api_name}", timeout=5)
            if response.status_code == 200:
                data = response.json()
                # Use display-friendly name
                api_name = data['name'].replace('-', ' ').title()
                # Check if we need to add it to our database
                if api_name not in self.pokemon_names:
                    print(f"🆕 Found {api_name} via PokeAPI - adding to database...")
                    self._add_pokemon_from_api(data)
                return api_name
        except:
            pass
        
        return None
    
    def _add_pokemon_from_api(self, data):
        """Add a Pokemon from PokeAPI response to our database"""
        try:
            name = data['name'].replace('-', ' ').title()
            types = [t['type']['name'].capitalize() for t in data['types']]
            stats = {s['stat']['name']: s['base_stat'] for s in data['stats']}
            
            new_pokemon = {
                'Name': name,
                'Type1': types[0] if types else 'Unknown',
                'Type2': types[1] if len(types) > 1 else '',
                'HP': stats.get('hp', 0),
                'Attack': stats.get('attack', 0),
                'Defense': stats.get('defense', 0),
                'Speed': stats.get('speed', 0),
                'Generation': 1,  # Default, can't always determine
                'Legendary': False
            }
            
            # Add to dataframe
            self.df = pd.concat([self.df, pd.DataFrame([new_pokemon])], ignore_index=True)
            self.pokemon_names = self.df['Name'].tolist()
            print(f"✅ Added {name} to database!")
        except Exception as e:
            print(f"❌ Failed to add Pokemon: {e}")
    
    def _get_sprite_url(self, pokemon_name):
        """Get Pokemon sprite URL from PokeAPI GitHub"""
        if not pokemon_name:
            return None
        
        try:
            # Handle Mega evolution patterns
            name_lower = pokemon_name.lower()
            clean_name = name_lower.replace(' ', '-').replace('.', '').replace("'", '')
            
            # Check for Mega X/Y patterns
            if ' mega x' in name_lower or 'mega-x' in clean_name:
                base = clean_name.replace(' mega x', '').replace('mega-x', '').replace('-mega-x', '')
                clean_name = f"{base}-mega-x"
            elif ' mega y' in name_lower or 'mega-y' in clean_name:
                base = clean_name.replace(' mega y', '').replace('mega-y', '').replace('-mega-y', '')
                clean_name = f"{base}-mega-y"
            elif ' x' in name_lower:
                base = name_lower.replace(' x', '')
                clean_name = f"{base}-mega-x"
            elif ' y' in name_lower:
                base = name_lower.replace(' y', '')
                clean_name = f"{base}-mega-y"
            
            response = requests.get(f"https://pokeapi.co/api/v2/pokemon/{clean_name}", timeout=5)
            if response.status_code == 200:
                data = response.json()
                pokemon_id = data['id']
                return f"https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/other/official-artwork/{pokemon_id}.png"
        except:
            pass
        
        return None
    
    def _get_similar_pokemon(self, pokemon_name, n=5):
        """Get similar Pokemon using KNN based on stats"""
        # Find the Pokemon index
        pokemon_match = self._fuzzy_find_pokemon(pokemon_name)
        if not pokemon_match:
            return None, f"Pokemon '{pokemon_name}' not found."
        
        idx = self.df[self.df['Name'] == pokemon_match].index[0]
        
        # Find nearest neighbors
        distances, indices = self.knn_model.kneighbors([self.normalized_features[idx]])
        
        # Get similar Pokemon (excluding the Pokemon itself)
        similar = []
        for i, dist in zip(indices[0][1:n+1], distances[0][1:n+1]):
            pokemon = self.df.iloc[i]
            similar.append({
                'name': pokemon['Name'],
                'type': f"{pokemon['Type1']}/{pokemon['Type2']}" if pokemon['Type2'] else pokemon['Type1'],
                'similarity': round(100 * (1 / (1 + dist)), 1)  # Convert distance to similarity %
            })
        
        return similar, pokemon_match
    
    def _make_conversational(self, data, user_question, context, user_profile=None):
        """Use Gemini to make a response sound natural and conversational"""
        if not getattr(self, 'gemini_client', None):
            return data  # Return raw data if no Gemini
        
        # Get current topic from context
        topic = context.get('current_topic', 'general')
        
        # Build history string for context
        history = context.get('conversation_history', [])
        history_str = ""
        if history:
            # Take last 3 exchanges
            recent = history[-3:] 
            for h in recent:
                role = "User" if h['role'] == 'user' else "Omnidex"
                history_str += f"{role}: {h['content']}\n"
        
        # User personalization
        user_context = ""
        if user_profile:
            if user_profile.get('name'):
                user_context += f"User's name is {user_profile['name']}. "
            if user_profile.get('fav_pokemon'):
                user_context += f"User's favorite Pokemon is {user_profile['fav_pokemon']}. "

        # Topic-specific instructions
        topic_instruction = ""
        if topic == 'battle':
            topic_instruction = "- Focus on battle strategy, type matchups, and competitive viability. Sound like a Gym Leader!"
        elif topic == 'lore':
            topic_instruction = "- Speak like a storyteller sharing myths and legends. Be mysterious and intriguing. Do NOT be brief - tell the full story."
        elif topic == 'evolution':
            topic_instruction = "- Express excitement about growth, metamorphosis, and potential. Sound like a Pokemon Professor!"
        elif topic == 'stats':
            topic_instruction = "- Be analytical but enthusiastic. Compare stats to average Pokemon."
        else:
            topic_instruction = "- Be friendly, helpful and enthusiastic!"

        try:
            prompt = f"""You are Omnidex, an expert AI Pokemon Professor.
Transform this raw data into a natural, conversational response.

USER CONTEXT: {user_context}

RECENT CONVERSATION:
{history_str}

User asked: "{user_question}"
Topic: {topic.upper()}

Data to transform:
{data}

Instructions:
{topic_instruction}
- Use relevant emoji to set the mood
- Sound natural, exclude raw JSON formatting
- If data is a list, weave it into a sentence naturally
- Reference previous context if relevant
- If user's name is known, use it occasionally.
"""

            response = self.gemini_client.models.generate_content(
                model='gemini-1.5-flash',
                contents=prompt
            )
            result = response.text
            print(f"✅ Conversational response generated successfully!")
            return result
        except Exception as e:
            print(f"❌ Gemini conversational error: {e}")
            return data  # Fallback to raw data
            
    def _answer_general_knowledge(self, question, context, user_profile=None):
        """Use Gemini to answer general Pokemon questions not in CSV"""
        if not getattr(self, 'gemini_client', None):
            return "I don't have that information in my database."
            
        try:
            # Build history string for context
            history = context.get('conversation_history', [])
            history_str = ""
            if history:
                recent = history[-3:] 
                for h in recent:
                    role = "User" if h['role'] == 'user' else "Omnidex"
                    history_str += f"{role}: {h['content']}\n"

            # User personalization
            user_context = ""
            if user_profile:
                if user_profile.get('name'):
                    user_context += f"User's name is {user_profile['name']}. "
                if user_profile.get('fav_pokemon'):
                    user_context += f"User's favorite Pokemon is {user_profile['fav_pokemon']}. "

            prompt = f"""You are Omnidex, an expert AI Pokemon Professor.
Answer the user's question about Pokemon using your general knowledge.

USER CONTEXT: {user_context}

RECENT CONVERSATION:
{history_str}

User asked: "{question}"

Instructions:
- Answer ONLY if it relates to Pokemon (anime, games, manga, lore).
- If it's not about Pokemon, politely redirect them to ask about Pokemon.
- Be friendly, enthusiastic and knowledgeable.
- Keep it to 2-3 sentences.
- Use 1-2 emoji.
- If user's name is known, use it naturally.
"""
            response = self.gemini_client.models.generate_content(
                model='gemini-2.0-flash',
                contents=prompt
            )
            return response.text
        except Exception as e:
            print(f"Gemini general knowledge error: {e}")
            return "I'm having trouble accessing my Pokemon knowledge base right now."
        
    def answer_question(self, question, context, user_profile=None):
        """Process natural language questions about Pokemon using ML"""
        original_question = question
        question_lower = question.lower().strip()

        # ============ USER PROFILE EXTRACTION ============
        # Try to extract user name or favorite pokemon
        if user_profile is not None:
            # Extract Name
            name_match = re.search(r"(?:my name is|i'm|im|call me) ([A-Z][a-z]+)", original_question, re.IGNORECASE)
            if name_match:
                name = name_match.group(1)
                user_profile['name'] = name
                return f"Nice to meet you, {name}! 👋 I'll remember that. What's your favorite Pokemon?"

            # Extract Favorite Pokemon
            fav_match = re.search(r"(?:my favorite pokemon is|i love|i like) ([A-Za-z]+)", original_question, re.IGNORECASE)
            if fav_match:
                fav_poke = fav_match.group(1)
                matched = self._fuzzy_find_pokemon(fav_poke)
                if matched:
                    user_profile['fav_pokemon'] = matched
                    return f"Awesome! {matched} is a great choice! I've noted that down as your favorite."

        # Handle greetings and general conversation with Gemini
        greetings = ['hi', 'hello', 'hey', 'heyy', 'yo', 'sup', 'what\'s up', 'howdy', 'greetings']
        if any(question_lower.startswith(g) for g in greetings) or len(question_lower) < 5:
            if getattr(self, 'gemini_client', None):
                try:
                    # Personalize greeting if name is known
                    greeting_name = f" {user_profile['name']}" if user_profile and user_profile.get('name') else ""

                    prompt = f"""You are PokéBot, a friendly and enthusiastic Pokemon expert chatbot.
The user{greeting_name} just said: "{original_question}"

Respond naturally and warmly! Introduce yourself as PokéBot and invite them to ask you anything about Pokemon - 
stats, lore, comparisons, evolutions, stories, etc. Also mention you can recommend similar Pokemon! 
Be friendly and use 1-2 Pokemon emoji. Keep it to 2-3 sentences."""
                    response = self.gemini_client.models.generate_content(
                        model='gemini-1.5-flash',
                        contents=prompt
                    )
                    return response.text
                except Exception as e:
                    print(f"Gemini greeting error: {e}")
            return "Hey there! 👋 I'm PokéBot, your Pokemon expert! Ask me anything about Pokemon - stats, stories, evolutions, comparisons, or let me recommend similar Pokemon!"
        
        # ============ PRONOUN RESOLUTION ============
        # Check if user used pronouns like "it", "its", "this one"
        resolved_pokemon = self._resolve_pronoun(question_lower, context)
        if resolved_pokemon:
            # Replace pronouns with actual Pokemon name for processing
            for pronoun in ['it', 'its', 'this one', 'that one', 'this pokemon', 'that pokemon']:
                if pronoun in question_lower:
                    question_lower = question_lower.replace(pronoun, resolved_pokemon.lower())
        
        # Add user message to history
        self._add_to_history(context, 'user', question)
        
        # Use ML to classify intent
        intent, confidence = self._classify_intent(question_lower)
        
        # OVERRIDE: Check for Story/Lore requests specifically
        # ML classifiers often confuse "story of mewtwo" with "info about mewtwo"
        story_keywords = ['story', 'lore', 'legend', 'history', 'origin', 'myth', 'tale', 'background']
        if any(word in question_lower for word in story_keywords):
            intent = "pokemon_lore"
            print(f"📖 Intent overridden to: pokemon_lore (keyword detected)")

        print(f"🤖 ML Intent: {intent} (confidence: {confidence:.2f})")
        
        # Save intent to context
        context['last_intent'] = intent
        
        # Detect and update topic
        context['current_topic'] = self._detect_topic(intent, question_lower)
        
        # Handle based on detected intent
        if intent == "type_query":
            # Extract type from question using fuzzy matching
            pokemon_types = ['Fire', 'Water', 'Grass', 'Electric', 'Psychic', 'Dragon', 
                           'Ice', 'Fighting', 'Poison', 'Ground', 'Rock', 'Ghost', 
                           'Dark', 'Steel', 'Fairy', 'Normal', 'Bug', 'Flying']
            for ptype in pokemon_types:
                if ptype.lower() in question_lower:
                    data = self._get_pokemon_by_type(ptype)
                    return self._make_conversational(data, original_question, context, user_profile)
        
        elif intent == "pokemon_lore":
            # Handle Story/Lore requests with Web Search
            pokemon_name = self._extract_pokemon_name(question_lower)
            if not pokemon_name and context.get('last_pokemon'):
                pokemon_name = context['last_pokemon']
                
            if pokemon_name:
                context['last_pokemon'] = pokemon_name
                context['current_topic'] = 'lore'
                
                # 1. First try to get web data for rich storytelling
                search_query = f"story legend lore history of {pokemon_name} pokemon"
                print(f"🔍 Searching web for lore: {search_query}")
                web_data = self._search_web(search_query)
                
                if web_data:
                    return self._make_conversational(web_data, original_question, context, user_profile)
                else:
                    # Fallback to internal knowledge if search fails
                    return self._make_conversational(f"Tell me the story of {pokemon_name}", original_question, context, user_profile)
            else:
                return "I'd love to tell you a story! Which Pokemon are you interested in?"

        elif intent == "pokemon_info":
            # Extract Pokemon name using fuzzy matching
            pokemon_name = self._extract_pokemon_name(question_lower)
            if pokemon_name:
                result = self._get_pokemon_info(pokemon_name)
                if result:
                    context['last_pokemon'] = pokemon_name
                    return self._make_conversational(result, original_question, context, user_profile)
        
        elif intent == "count_legendary":
            data = self._count_legendary()
            return self._make_conversational(data, original_question, context, user_profile)
        
        elif intent == "count_type":
            pokemon_types = ['Fire', 'Water', 'Grass', 'Electric', 'Psychic', 'Dragon', 
                           'Ice', 'Fighting', 'Poison', 'Ground', 'Rock', 'Ghost', 
                           'Dark', 'Steel', 'Fairy', 'Normal', 'Bug', 'Flying']
            for ptype in pokemon_types:
                if ptype.lower() in question_lower:
                    data = self._count_type(ptype)
                    return self._make_conversational(data, original_question, context, user_profile)
        
        elif intent == "count_total":
            return self._make_conversational(f"There are {len(self.df)} Pokemon in the database.", original_question, context, user_profile)
        
        elif intent == "stat_leader":
            if 'attack' in question_lower:
                data = self._get_stat_leader('Attack', 'highest')
            elif 'defense' in question_lower:
                data = self._get_stat_leader('Defense', 'highest')
            elif 'speed' in question_lower or 'fast' in question_lower:
                data = self._get_stat_leader('Speed', 'highest')
            elif 'hp' in question_lower or 'health' in question_lower:
                data = self._get_stat_leader('HP', 'highest')
            else:
                data = self._get_stat_leader('Attack', 'highest')
            return self._make_conversational(data, original_question, context, user_profile)
        
        elif intent == "strongest_overall":
            data = self._get_strongest_overall()
            return self._make_conversational(data, original_question, context, user_profile)
        
        elif intent == "list_legendary":
            data = self._get_legendary_pokemon()
            return self._make_conversational(data, original_question, context, user_profile)
        
        elif intent == "generation_query":
            gen_match = re.search(r'gen(?:eration)?\s*(\d+)', question_lower)
            if gen_match:
                gen = int(gen_match.group(1))
                data = self._get_pokemon_by_generation(gen)
                return self._make_conversational(data, original_question, context, user_profile)
        
        elif intent == "compare":
            # Extract two Pokemon names for comparison
            pokemon_names = self._extract_multiple_pokemon_names(question_lower)
            if len(pokemon_names) >= 2:
                data = self._compare_pokemon(pokemon_names[0], pokemon_names[1])
                # Set context to the first Pokemon for follow-up questions
                context['last_pokemon'] = pokemon_names[0]
                context['compared_pokemon'] = pokemon_names  # Store both
                return self._make_conversational(data, original_question, context, user_profile)
        
        elif intent == "dual_type":
            data = self._get_dual_type_pokemon()
            return self._make_conversational(data, original_question, context, user_profile)
        
        elif intent == "recommend":
            # Extract Pokemon name for recommendations
            pokemon_name = self._extract_pokemon_name(question_lower)
            if pokemon_name:
                similar, matched_name = self._get_similar_pokemon(pokemon_name)
                if similar:
                    data = f"Pokemon similar to {matched_name}:\n"
                    for s in similar:
                        data += f"• {s['name']} ({s['type']}) - {s['similarity']}% similar\n"
                    return self._make_conversational(data, original_question, context, user_profile)
        
        # ============ NEW: WEAKNESS QUERIES ============
        elif intent == "weakness":
            # Try to extract a Pokemon name first
            pokemon_name = self._extract_pokemon_name(question_lower)
            if pokemon_name:
                data = self._get_pokemon_weakness(pokemon_name)
                context['last_pokemon'] = pokemon_name
                return self._make_conversational(data, original_question, context, user_profile)
            # Otherwise check for type name
            for ptype in self.TYPE_CHART.keys():
                if ptype.lower() in question_lower:
                    data = self._get_type_weakness(ptype)
                    return self._make_conversational(data, original_question, context, user_profile)
        
        # ============ NEW: STRENGTH QUERIES ============
        elif intent == "strength":
            pokemon_name = self._extract_pokemon_name(question_lower)
            if pokemon_name:
                data = self._get_pokemon_strength(pokemon_name)
                context['last_pokemon'] = pokemon_name
                return self._make_conversational(data, original_question, context, user_profile)
            for ptype in self.TYPE_CHART.keys():
                if ptype.lower() in question_lower:
                    data = self._get_type_strength(ptype)
                    return self._make_conversational(data, original_question, context, user_profile)
        
        # ============ NEW: EVOLUTION QUERIES ============
        elif intent == "evolution":
            pokemon_name = self._extract_pokemon_name(question_lower)
            if pokemon_name:
                data = self._get_evolution_info(pokemon_name, context)
                context['last_pokemon'] = pokemon_name
                return self._make_conversational(data, original_question, context, user_profile)
        
        # ============ NEW: CONTEXT-AWARE QUERIES ============
        elif intent and intent.startswith("context_"):
            last_pokemon = context.get('last_pokemon')
            if last_pokemon:
                if intent == "context_stat":
                    # Check which stat they're asking about
                    if 'defense' in question_lower:
                        pokemon = self.df[self.df['Name'] == last_pokemon].iloc[0]
                        data = f"{last_pokemon}'s Defense is {pokemon['Defense']}."
                    elif 'attack' in question_lower:
                        pokemon = self.df[self.df['Name'] == last_pokemon].iloc[0]
                        data = f"{last_pokemon}'s Attack is {pokemon['Attack']}."
                    elif 'speed' in question_lower:
                        pokemon = self.df[self.df['Name'] == last_pokemon].iloc[0]
                        data = f"{last_pokemon}'s Speed is {pokemon['Speed']}."
                    elif 'hp' in question_lower:
                        pokemon = self.df[self.df['Name'] == last_pokemon].iloc[0]
                        data = f"{last_pokemon}'s HP is {pokemon['HP']}."
                    else:
                        data = self._get_pokemon_info(last_pokemon)
                    return self._make_conversational(data, original_question, context, user_profile)
                elif intent == "context_type":
                    pokemon = self.df[self.df['Name'] == last_pokemon].iloc[0]
                    type_str = pokemon['Type1']
                    if pokemon['Type2']:
                        type_str += f" and {pokemon['Type2']}"
                    data = f"{last_pokemon} is a {type_str} type Pokemon."
                    return self._make_conversational(data, original_question, context, user_profile)
                elif intent == "context_legendary":
                    pokemon = self.df[self.df['Name'] == last_pokemon].iloc[0]
                    is_legendary = "Yes, it is!" if pokemon['Legendary'] else "No, it's not."
                    data = f"Is {last_pokemon} legendary? {is_legendary}"
                    return self._make_conversational(data, original_question, context, user_profile)
                elif intent == "context_weakness":
                    data = self._get_pokemon_weakness(last_pokemon)
                    return self._make_conversational(data, original_question, context, user_profile)
                elif intent == "context_evolution":
                    data = self._get_evolution_info(last_pokemon, context)
                    return self._make_conversational(data, original_question, context, user_profile)
                elif intent == "context_recommend":
                    similar, matched_name = self._get_similar_pokemon(last_pokemon)
                    if similar:
                        data = f"Pokemon similar to {matched_name}:\n"
                        for s in similar:
                            data += f"• {s['name']} ({s['type']}) - {s['similarity']}% similar\n"
                        return self._make_conversational(data, original_question, context, user_profile)
                elif intent == "context_story":
                    # Search for the last Pokemon's story/lore
                    return self._search_web(f"{last_pokemon} Pokemon lore backstory origin")
            else:
                return "I'm not sure which Pokemon you're referring to. Could you tell me the Pokemon's name?"
        
        # Fallback: Try to extract and look up a Pokemon name
        pokemon_name = self._extract_pokemon_name(question_lower)
        if pokemon_name:
            context['last_pokemon'] = pokemon_name
            result = self._get_pokemon_info(pokemon_name)
            if result:
                return self._make_conversational(result, original_question, context, user_profile)
        
        # Final fallback: Use Gemini for general knowledge
        return self._answer_general_knowledge(original_question, context, user_profile)

    def analyze_image(self, image_path):
        """Analyze an uploaded image to identify a Pokemon using Gemini Vision"""
        if not self.gemini_model:
            return {"error": "AI Vision not enabled (missing API key)"}

        try:
            print(f"👁️ Analyzing image: {image_path}")

            # Load image
            img = Image.open(image_path)

            # Use Gemini Vision
            # vision_model = genai.GenerativeModel('gemini-flash-latest') # No longer needed with new client pattern

            prompt = """Identify the Pokemon in this image.
            Return a valid JSON object with:
            {
                "name": "Pokemon Name",
                "confidence": "High/Medium/Low",
                "description": "Brief visual description",
                "is_shiny": true/false
            }
            Do not include Markdown formatting like ```json. Just return the raw JSON string."""

            response = self.gemini_client.models.generate_content(
                model='gemini-1.5-flash',
                contents=[prompt, img]
            )
            text = response.text.strip()

            # Clean up potential markdown formatting
            if text.startswith('```json'):
                text = text.replace('```json', '').replace('```', '')

            # Parse JSON
            try:
                data = json.loads(text)
            except:
                # Fallback if not valid JSON
                data = {
                    "name": "Unknown",
                    "confidence": "Low",
                    "description": text,
                    "is_shiny": False
                }

            # If we found a name, try to match it to our database
            if data.get('name'):
                matched = self._fuzzy_find_pokemon(data['name'])
                if matched:
                    data['name'] = matched
                    data['db_match'] = True

            return data

        except Exception as e:
            print(f"❌ Image analysis failed: {e}")
            return {"error": f"Failed to analyze image: {str(e)}"}
    
    def _extract_pokemon_name(self, text):
        """Extract a Pokemon name from text using fuzzy matching"""
        # Remove common words
        stop_words = ['tell', 'me', 'about', 'what', 'is', 'who', 'info', 'information',
                      'similar', 'like', 'recommend', 'pokemon', 'find', 'suggest', 'the',
                      'a', 'an', 'stats', 'for', 'on', 'describe', 'show']
        
        words = text.lower().split()
        for word in words:
            clean_word = re.sub(r'[^\w]', '', word)
            if clean_word not in stop_words and len(clean_word) > 2:
                match = self._fuzzy_find_pokemon(clean_word)
                if match:
                    return match
        return None
    
    def _extract_multiple_pokemon_names(self, text):
        """Extract multiple Pokemon names from text for comparison"""
        found_names = []
        words = re.split(r'\s+|vs|and|or|,', text.lower())
        
        for word in words:
            clean_word = re.sub(r'[^\w]', '', word)
            if len(clean_word) > 2:
                match = self._fuzzy_find_pokemon(clean_word)
                if match and match not in found_names:
                    found_names.append(match)
        
        return found_names
    
    def _get_pokemon_by_type(self, pokemon_type):
        """Get all Pokemon of a specific type"""
        filtered = self.df[(self.df['Type1'] == pokemon_type) | (self.df['Type2'] == pokemon_type)]
        if len(filtered) == 0:
            return f"No {pokemon_type} type Pokemon found in the database."
        
        names = filtered['Name'].tolist()
        result = f"Found {len(names)} {pokemon_type} type Pokemon:\n"
        result += ", ".join(names)
        return result
    
    def _get_pokemon_info(self, pokemon_name):
        """Get detailed information about a specific Pokemon"""
        # Use fuzzy matching to find the Pokemon
        matched_name = self._fuzzy_find_pokemon(pokemon_name)
        if not matched_name:
            return None
        
        pokemon = self.df[self.df['Name'] == matched_name]
        if len(pokemon) == 0:
            return None
        
        info = pokemon.iloc[0]
        type_str = info['Type1']
        if info['Type2']:
            type_str += f"/{info['Type2']}"
        
        legendary_str = "Yes" if info['Legendary'] else "No"
        
        result = f"\n📊 {info['Name']} Information:\n"
        result += f"Type: {type_str}\n"
        result += f"HP: {info['HP']}\n"
        result += f"Attack: {info['Attack']}\n"
        result += f"Defense: {info['Defense']}\n"
        result += f"Speed: {info['Speed']}\n"
        result += f"Generation: {info['Generation']}\n"
        result += f"Legendary: {legendary_str}\n"
        return result
    
    def _count_legendary(self):
        """Count legendary Pokemon"""
        count = len(self.df[self.df['Legendary'] == True])
        names = self.df[self.df['Legendary'] == True]['Name'].tolist()
        result = f"There are {count} legendary Pokemon:\n"
        result += ", ".join(names)
        return result
    
    def _count_type(self, pokemon_type):
        """Count Pokemon of a specific type"""
        count = len(self.df[(self.df['Type1'] == pokemon_type) | (self.df['Type2'] == pokemon_type)])
        return f"There are {count} {pokemon_type} type Pokemon."
    
    def _get_stat_leader(self, stat, direction='highest'):
        """Get Pokemon with highest/lowest stat"""
        if direction == 'highest':
            pokemon = self.df.loc[self.df[stat].idxmax()]
        else:
            pokemon = self.df.loc[self.df[stat].idxmin()]
        
        return f"{pokemon['Name']} has the {direction} {stat} with {pokemon[stat]}!"
    
    def _get_strongest_overall(self):
        """Get strongest Pokemon by total stats"""
        self.df['Total'] = self.df['HP'] + self.df['Attack'] + self.df['Defense'] + self.df['Speed']
        strongest = self.df.loc[self.df['Total'].idxmax()]
        
        return f"The strongest Pokemon overall is {strongest['Name']} with total stats of {strongest['Total']}!"
    
    def _get_legendary_pokemon(self):
        """Get all legendary Pokemon"""
        legendary = self.df[self.df['Legendary'] == True]
        names = legendary['Name'].tolist()
        result = f"There are {len(names)} legendary Pokemon:\n"
        result += ", ".join(names)
        return result
    
    def _get_pokemon_by_generation(self, gen):
        """Get Pokemon from a specific generation"""
        filtered = self.df[self.df['Generation'] == gen]
        if len(filtered) == 0:
            return f"No Pokemon found from generation {gen}."
        
        names = filtered['Name'].tolist()
        result = f"Found {len(names)} Pokemon from Generation {gen}:\n"
        result += ", ".join(names)
        return result
    
    def _get_dual_type_pokemon(self):
        """Get all dual-type Pokemon"""
        dual_type = self.df[self.df['Type2'] != '']
        names = dual_type['Name'].tolist()
        result = f"There are {len(names)} dual-type Pokemon:\n"
        for _, row in dual_type.head(20).iterrows():  # Limit to prevent huge output
            result += f"{row['Name']} ({row['Type1']}/{row['Type2']}), "
        result = result.rstrip(', ')
        if len(names) > 20:
            result += f"\n... and {len(names) - 20} more!"
        return result
    
    def _compare_pokemon(self, poke1, poke2):
        """Compare two Pokemon"""
        # Use fuzzy matching for both Pokemon
        matched1 = self._fuzzy_find_pokemon(poke1)
        matched2 = self._fuzzy_find_pokemon(poke2)
        
        if not matched1:
            return f"Pokemon '{poke1}' not found. Did you mean one of these: {', '.join(self.pokemon_names[:5])}...?"
        if not matched2:
            return f"Pokemon '{poke2}' not found. Did you mean one of these: {', '.join(self.pokemon_names[:5])}...?"
        
        p1 = self.df[self.df['Name'] == matched1].iloc[0]
        p2 = self.df[self.df['Name'] == matched2].iloc[0]
        
        result = f"\n⚔️ Comparing {matched1} vs {matched2}:\n\n"
        stats = ['HP', 'Attack', 'Defense', 'Speed']
        
        p1_wins = 0
        p2_wins = 0
        
        for stat in stats:
            val1 = p1[stat]
            val2 = p2[stat]
            if val1 > val2:
                winner = matched1
                p1_wins += 1
            elif val2 > val1:
                winner = matched2
                p2_wins += 1
            else:
                winner = "Tie"
            result += f"{stat}: {matched1}={val1} vs {matched2}={val2} → Winner: {winner}\n"
        
        result += f"\n🏆 Overall: {matched1} wins {p1_wins} stats, {matched2} wins {p2_wins} stats"
        
        return result
    
    def _search_web(self, query):
        """Search the web with self-learning capabilities - Optimized for Lore/Stories"""
        try:
            # 1. First check if we've learned this before
            cached = self._check_learned_cache(query)
            if cached:
                print(f"💾 Using cached answer for: {query}")
                context = "\n".join([f"- {r.get('title', '')}: {r.get('body', '')}" for r in cached.get('results', [])])
                if getattr(self, 'gemini_client', None):
                    try:
                        prompt = f"""You are Omnidex, an expert Pokemon Storyteller.
User asked: {query}

Using this cached information:
{context}

Tell a detailed and engaging story/answer. Do NOT be brief.
If it's about lore, be dramatic and immersive."""
                        response = self.gemini_client.models.generate_content(
                            model='gemini-1.5-flash',
                            contents=prompt
                        )
                        return response.text
                    except:
                        pass
            
            # 2. Try to extract a Pokemon name and check PokeAPI
            potential_pokemon = self._extract_pokemon_name(query.lower())
            # (Keeping existing PokeAPI check logic...)
            if potential_pokemon is None:
                 # Maybe it's an unknown Pokemon - try words in the query
                words = query.lower().split()
                for word in words:
                    if len(word) > 3 and word not in ['what', 'about', 'tell', 'story', 'lore', 'pokemon']:
                        pokemon_info, is_new = self._fetch_from_pokeapi(word)
                        if pokemon_info:
                             if is_new:
                                result = f"🆕 I just learned about {pokemon_info['name']}!\n\n"
                                result += f"📊 {pokemon_info['name']} Info:\n"
                                result += f"Type: {pokemon_info['type1']}"
                                if pokemon_info['type2']:
                                    result += f"/{pokemon_info['type2']}"
                                result += f"\nHP: {pokemon_info['hp']}, Attack: {pokemon_info['attack']}, Defense: {pokemon_info['defense']}, Speed: {pokemon_info['speed']}\n"
                                result += f"Generation: {pokemon_info['generation']}\n"
                                result += f"\n✅ Added to my database! I'll remember this Pokemon from now on."
                                return self._make_conversational(result, query) if getattr(self, 'gemini_client', None) else result
                             break

            # 3. Search the web for story/lore questions - UPDATED LOGIC
            is_lore_query = any(w in query.lower() for w in ['story', 'lore', 'legend', 'history', 'origin', 'myth'])
            
            if is_lore_query:
                # Target high-quality lore sources
                search_query = f"{query} site:bulbapedia.bulbagarden.net OR site:serebii.net OR site:pokemon.fandom.com"
                max_res = 8  # Get more context for stories
            else:
                search_query = f"Pokemon {query}"
                max_res = 5

            print(f"🔍 Searching web: {search_query}")
            
            with DDGS() as ddgs:
                results = list(ddgs.text(search_query, max_results=max_res))
            
            if not results:
                # Fallback to broader search if Wiki search fails
                if is_lore_query:
                    print("⚠️ Wiki search failed, trying broad search...")
                    with DDGS() as ddgs:
                        # Fallback: search for lore specifically
                        raw_results = list(ddgs.text(f"Pokemon {query} lore", max_results=8))
                        # Manual filtering
                        results = [r for r in raw_results if "pokemmo" not in r['title'].lower() and "pokemmo" not in r['body'].lower()]
            
            if not results:
                return "I couldn't find any information about that. Try asking about specific Pokemon stats or types!"

            # Filter out any lingering PokeMMO results if coming from main search
            results = [r for r in results if "pokemmo" not in r['title'].lower() and "pokemmo" not in r['body'].lower()]
            
            # 4. Store this for future reference (self-learning!)
            self._learn_from_web(query, [{'title': r.get('title', ''), 'body': r.get('body', '')} for r in results])
            
            # Combine search results as context
            context = "\n".join([
                f"- {r.get('title', '')}: {r.get('body', '')}" 
                for r in results
            ])
            
            # If Gemini is available, generate a natural response
            if getattr(self, 'gemini_client', None):
                try:
                    if is_lore_query:
                        prompt = f"""You are Omnidex, the ultimate Pokemon Storyteller.
The user wants to know the LORE/STORY about: {query}

I have gathered these snippets from the web:
{context}

YOUR TASK:
Weave these snippets into a COMPLETE, ENGAGING, and DETAILED narrative.

RULES:
- Do NOT return links, article titles, or timestamps.
- Do NOT dump search results or bullet points.
- Ignore game mechanics (raids, IVs, builds, stats) and non-canonical guides.
- Prioritize canonical lore from games, anime, and Pokedex entries.
- Treat Pokemon as a fictional universe.
- Be dramatic and immersive, like a true storyteller.
- Synthesize the information into a single, cohesive story (at least 2 paragraphs).
- If information is missing, focus on what is known about the Pokemon's myths or origins."""
                    else:
                         prompt = f"""You are Omnidex, a friendly Pokemon expert.
User's question: {query}

Info found:
{context}

Answer conversationally and helpfully. Keep it to 3-4 sentences."""

                    response = self.gemini_client.models.generate_content(
                        model='gemini-1.5-flash',
                        contents=prompt
                    )
                    return response.text
                    
                except Exception as e:
                    print(f"Gemini error: {e}")
                    # Fall back to raw results
            
            # Fallback: return formatted search results
            response = "Here's what I found:\n\n"
            for result in results[:3]:
                body = result.get('body', '')[:150]
                response += f"• {body}...\n\n"
            return response
            
        except Exception as e:
            print(f"Search error: {e}")
            return f"I had trouble searching for that. Try asking about Pokemon stats instead!"
    
    def _get_type_weakness(self, pokemon_type):
        """Get weaknesses for a specific type"""
        if pokemon_type not in self.TYPE_CHART:
            return f"Unknown type: {pokemon_type}"
        
        type_info = self.TYPE_CHART[pokemon_type]
        weak_to = type_info.get('weak', [])
        immune_to = type_info.get('immune', [])
        
        result = f"⚡ {pokemon_type} Type Weaknesses:\n"
        if weak_to:
            result += f"Weak to: {', '.join(weak_to)}\n"
        if immune_to:
            result += f"Immune to: {', '.join(immune_to)}\n"
        
        return result
    
    def _get_type_strength(self, pokemon_type):
        """Get strengths for a specific type"""
        if pokemon_type not in self.TYPE_CHART:
            return f"Unknown type: {pokemon_type}"
        
        type_info = self.TYPE_CHART[pokemon_type]
        strong_against = type_info.get('strong', [])
        resist = type_info.get('resist', [])
        
        result = f"💪 {pokemon_type} Type Strengths:\n"
        if strong_against:
            result += f"Super effective against: {', '.join(strong_against)}\n"
        if resist:
            result += f"Resists: {', '.join(resist)}\n"
        
        return result
    
    def _get_pokemon_weakness(self, pokemon_name):
        """Get weaknesses for a specific Pokemon based on its type(s)"""
        matched_name = self._fuzzy_find_pokemon(pokemon_name)
        if not matched_name:
            return f"Pokemon '{pokemon_name}' not found."
        
        pokemon = self.df[self.df['Name'] == matched_name].iloc[0]
        type1 = pokemon['Type1']
        type2 = pokemon['Type2'] if pokemon['Type2'] else None
        
        # Collect all weaknesses from both types
        all_weaknesses = set()
        all_immunities = set()
        
        if type1 in self.TYPE_CHART:
            all_weaknesses.update(self.TYPE_CHART[type1]['weak'])
            all_immunities.update(self.TYPE_CHART[type1]['immune'])
        if type2 and type2 in self.TYPE_CHART:
            all_weaknesses.update(self.TYPE_CHART[type2]['weak'])
            all_immunities.update(self.TYPE_CHART[type2]['immune'])
        
        type_str = f"{type1}/{type2}" if type2 else type1
        result = f"⚡ {matched_name} ({type_str}) Weaknesses:\n"
        if all_weaknesses:
            result += f"Weak to: {', '.join(sorted(all_weaknesses))}\n"
        if all_immunities:
            result += f"Immune to: {', '.join(sorted(all_immunities))}\n"
        
        return result
    
    def _get_pokemon_strength(self, pokemon_name):
        """Get strengths for a specific Pokemon based on its type(s)"""
        matched_name = self._fuzzy_find_pokemon(pokemon_name)
        if not matched_name:
            return f"Pokemon '{pokemon_name}' not found."
        
        pokemon = self.df[self.df['Name'] == matched_name].iloc[0]
        type1 = pokemon['Type1']
        type2 = pokemon['Type2'] if pokemon['Type2'] else None
        
        # Collect all strengths from both types
        all_strengths = set()
        
        if type1 in self.TYPE_CHART:
            all_strengths.update(self.TYPE_CHART[type1]['strong'])
        if type2 and type2 in self.TYPE_CHART:
            all_strengths.update(self.TYPE_CHART[type2]['strong'])
        
        type_str = f"{type1}/{type2}" if type2 else type1
        result = f"💪 {matched_name} ({type_str}) Strengths:\n"
        if all_strengths:
            result += f"Super effective against: {', '.join(sorted(all_strengths))}\n"
        else:
            result += "No special type advantages.\n"
        
        return result
    
    def _get_evolution_info(self, pokemon_name, context):
        """Get evolution information for a Pokemon from local data or PokeAPI"""
        matched_name = self._fuzzy_find_pokemon(pokemon_name)
        if not matched_name:
            return f"Pokemon '{pokemon_name}' not found."
        
        # Try local data first
        if matched_name in self.EVOLUTION_DATA:
            evo_data = self.EVOLUTION_DATA[matched_name]
            result = f"🔄 {matched_name} Evolution Info:\n"
            
            if 'evolves_from' in evo_data:
                result += f"• Evolves from: {evo_data['evolves_from']}\n"
            
            if 'evolves_to' in evo_data:
                evolves_to = evo_data['evolves_to']
                method = evo_data.get('method', 'unknown')
                
                if isinstance(evolves_to, list):
                    result += f"• Can evolve into:\n"
                    for evo in evolves_to:
                        result += f"  - {evo}\n"
                else:
                    if method == 'level':
                        result += f"• Evolves to: {evolves_to} at level {evo_data.get('level', '?')}\n"
                    elif method == 'trade':
                        result += f"• Evolves to: {evolves_to} when traded\n"
                    elif method == 'friendship':
                        result += f"• Evolves to: {evolves_to} with high friendship\n"
                    else:
                        result += f"• Evolves to: {evolves_to} using {method}\n"
            else:
                result += "• This is the final evolution!\n"
            
            return result
        
        # Fetch from PokeAPI if not in local data
        try:
            clean_name = matched_name.lower().replace(' ', '-').replace('.', '').replace("'", '')
            
            # Get Pokemon species for evolution chain
            species_resp = requests.get(f"https://pokeapi.co/api/v2/pokemon-species/{clean_name}", timeout=5)
            if species_resp.status_code == 200:
                species_data = species_resp.json()
                evo_chain_url = species_data['evolution_chain']['url']
                
                # Get evolution chain
                chain_resp = requests.get(evo_chain_url, timeout=5)
                if chain_resp.status_code == 200:
                    chain_data = chain_resp.json()
                    result = f"🔄 {matched_name} Evolution Chain:\n"
                    
                    # Parse evolution chain with image URLs
                    chain = chain_data['chain']
                    evo_list = []
                    
                    def parse_chain(node, level=0):
                        name = node['species']['name'].title()
                        pokemon_id = int(node['species']['url'].split('/')[-2])
                        sprite_url = f"https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/other/official-artwork/{pokemon_id}.png"
                        evo_list.append({'name': name, 'level': level, 'sprite': sprite_url, 'id': pokemon_id})
                        for evo in node.get('evolves_to', []):
                            parse_chain(evo, level + 1)
                    
                    parse_chain(chain)
                    
                    # Store evolution sprites for frontend
                    context['evolution_chain'] = evo_list
                    
                    # Format text chain
                    for evo in evo_list:
                        prefix = "└─> " if evo['level'] > 0 else ""
                        indent = "    " * evo['level']
                        is_current = evo['name'].lower() == matched_name.lower()
                        marker = " ⭐" if is_current else ""
                        result += f"{indent}{prefix}{evo['name']}{marker}\n"
                    
                    return result
                    
        except Exception as e:
            print(f"PokeAPI evolution error: {e}")
        
        return f"I don't have evolution data for {matched_name} yet."
    
    # ============ SELF-LEARNING METHODS ============
    
    def _load_learned_data(self):
        """Load previously learned data from cache file"""
        cache_file = os.path.join(os.path.dirname(self.csv_file), 'learned_cache.json')
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    self.learned_data = json.load(f)
                print(f"✅ Loaded {len(self.learned_data)} learned entries from cache")
            except Exception as e:
                print(f"⚠️ Could not load learned cache: {e}")
                self.learned_data = {}
        else:
            self.learned_data = {}
    
    def _save_learned_data(self):
        """Save learned data to cache file"""
        cache_file = os.path.join(os.path.dirname(self.csv_file), 'learned_cache.json')
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.learned_data, f, indent=2)
        except Exception as e:
            print(f"⚠️ Could not save learned cache: {e}")
    
    def _fetch_from_pokeapi(self, pokemon_name):
        """Fetch Pokemon data from PokeAPI and optionally add to database"""
        try:
            # Clean the name for API call
            clean_name = pokemon_name.lower().replace(' ', '-').replace('.', '').replace("'", '')
            
            # First try PokeAPI
            url = f"https://pokeapi.co/api/v2/pokemon/{clean_name}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                # Extract Pokemon info
                name = data['name'].capitalize()
                types = [t['type']['name'].capitalize() for t in data['types']]
                stats = {s['stat']['name']: s['base_stat'] for s in data['stats']}
                
                pokemon_info = {
                    'name': name,
                    'type1': types[0] if types else 'Normal',
                    'type2': types[1] if len(types) > 1 else '',
                    'hp': stats.get('hp', 50),
                    'attack': stats.get('attack', 50),
                    'defense': stats.get('defense', 50),
                    'speed': stats.get('speed', 50),
                    'generation': self._guess_generation(data['id']),
                    'legendary': data['id'] in [144, 145, 146, 150, 151, 243, 244, 245, 249, 250, 251]  # Basic legendary check
                }
                
                # Check if already in database
                if name not in self.pokemon_names:
                    # Add to database!
                    self._add_pokemon_to_database(pokemon_info)
                    print(f"🆕 LEARNED: Added {name} to database from PokeAPI!")
                    return pokemon_info, True  # True = newly learned
                else:
                    return pokemon_info, False  # False = already knew it
                    
            return None, False
            
        except requests.exceptions.Timeout:
            print(f"⚠️ PokeAPI timeout for {pokemon_name}")
            return None, False
        except Exception as e:
            print(f"⚠️ PokeAPI error: {e}")
            return None, False
    
    def _guess_generation(self, pokedex_id):
        """Guess generation based on Pokedex ID"""
        if pokedex_id <= 151: return 1
        elif pokedex_id <= 251: return 2
        elif pokedex_id <= 386: return 3
        elif pokedex_id <= 493: return 4
        elif pokedex_id <= 649: return 5
        elif pokedex_id <= 721: return 6
        elif pokedex_id <= 809: return 7
        elif pokedex_id <= 905: return 8
        else: return 9
    
    def _add_pokemon_to_database(self, pokemon_info):
        """Add a new Pokemon to the CSV database"""
        try:
            new_row = pd.DataFrame([{
                'Name': pokemon_info['name'],
                'Type1': pokemon_info['type1'],
                'Type2': pokemon_info['type2'],
                'HP': pokemon_info['hp'],
                'Attack': pokemon_info['attack'],
                'Defense': pokemon_info['defense'],
                'Speed': pokemon_info['speed'],
                'Generation': pokemon_info['generation'],
                'Legendary': pokemon_info['legendary']
            }])
            
            # Add to dataframe
            self.df = pd.concat([self.df, new_row], ignore_index=True)
            self.pokemon_names.append(pokemon_info['name'])
            
            # Save to CSV
            self.df.to_csv(self.csv_file, index=False)
            
            # Retrain recommendation system with new data
            self._init_recommendation_system()
            
            return True
        except Exception as e:
            print(f"⚠️ Failed to add Pokemon to database: {e}")
            return False
    
    def _learn_from_web(self, query, search_results):
        """Store story/lore learned from web search for future reference"""
        # Create a cache key from the query
        cache_key = query.lower().strip()
        
        if cache_key not in self.learned_data:
            # Store the learned information
            self.learned_data[cache_key] = {
                'query': query,
                'results': search_results[:3],  # Store top 3 results
                'timestamp': pd.Timestamp.now().isoformat()
            }
            self._save_learned_data()
            print(f"📚 LEARNED: Stored answer for '{query}' for future reference")
    
    def _check_learned_cache(self, query):
        """Check if we've answered this question before"""
        cache_key = query.lower().strip()
        
        # Check exact match
        if cache_key in self.learned_data:
            return self.learned_data[cache_key]
        
        # Check fuzzy match
        for key in self.learned_data.keys():
            if cache_key in key or key in cache_key:
                return self.learned_data[key]
        
        return None


def main():
    """Main function to run the Pokemon chatbot"""
    print("🔥 Pokemon Chatbot Starting...\n")
    
    # Initializdef main():
    # Check if data exists in 'data' folder or current directory
    csv_path = 'data/pokemon_data.csv'
    if not os.path.exists(csv_path):
        csv_path = 'pokemon_data.csv'
    
    if os.path.exists(csv_path):
        chatbot = PokemonChatbot(csv_path)
    else:
        print("❌ Error: pokemon_data.csv not found in 'data/' or current directory.")
        return
    
    print("\n" + "="*60)
    print("Welcome to the ML-Powered Pokemon Chatbot!")
    print("="*60)
    print("\nAsk me anything about Pokemon! Examples:")
    print("- 'List all fire type Pokemon'")
    print("- 'Tell me about Charizard' (or 'charazard' - I handle typos!)")
    print("- 'How many legendary Pokemon are there?'")
    print("- 'Who has the highest attack?'")
    print("- 'Show me generation 1 Pokemon'")
    print("- 'Compare Charizard and Blastoise'")
    print("- 'Recommend Pokemon similar to Pikachu' (NEW! ML-powered)")
    print("\nType 'quit' or 'exit' to stop.\n")
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
            print("Goodbye! Thanks for using Pokemon Chatbot!")
            break
        
        if not user_input:
            continue
        
        response = chatbot.answer_question(user_input)
        print(f"\nBot: {response}\n")


if __name__ == "__main__":
    main()
