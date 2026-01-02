import pandas as pd
import numpy as np
import json
import os
try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
from sklearn.neighbors import NearestNeighbors
from difflib import get_close_matches

class DataService:
    def __init__(self, csv_file, data_dir='data'):
        self.csv_file = csv_file
        self.data_dir = data_dir
        self.df = None
        self.pokemon_names = []
        self.type_chart = {}
        self.evolution_data = {}
        
        # ML Models
        self.knn_model = None
        self.normalized_features = None
        
        # Vector DB
        self.faiss_index = None
        self.pokemon_metadata = []
        
        self._load_data()
        self._init_recommendation_system()

    def _load_data(self):
        """Load CSV and JSON data"""
        # Load CSV
        try:
            self.df = pd.read_csv(self.csv_file)
            self.df['Type2'] = self.df['Type2'].fillna('')
            self.pokemon_names = self.df['Name'].tolist()
            print(f"✅ Loaded {len(self.df)} Pokemon from database!")
        except Exception as e:
            print(f"❌ Failed to load CSV data: {e}")
            self.df = pd.DataFrame()

        # Load JSONs
        try:
            with open(os.path.join(self.data_dir, 'type_chart.json'), 'r') as f:
                self.type_chart = json.load(f)
            print(f"✅ Type Effectiveness Chart loaded (18 types)")
        except Exception as e:
            print(f"❌ Failed to load type chart: {e}")

        try:
            with open(os.path.join(self.data_dir, 'evolution.json'), 'r') as f:
                self.evolution_data = json.load(f)
            print(f"✅ Evolution Data loaded")
        except Exception as e:
            print(f"❌ Failed to load evolution data: {e}")

    def fuzzy_find_pokemon(self, name, cutoff=0.75):
        """Find Pokemon name using fuzzy matching"""
        if not name: return None
        name_clean = name.strip()
        
        # 1. Exact match
        for pokemon_name in self.pokemon_names:
            if pokemon_name.lower() == name_clean.lower():
                return pokemon_name
        
        # 2. Fuzzy match
        matches = get_close_matches(name_clean.capitalize(), self.pokemon_names, n=1, cutoff=cutoff)
        if matches:
            return matches[0]
        
        # 3. Partial match
        name_lower = name_clean.lower()
        for pokemon_name in self.pokemon_names:
            if pokemon_name.lower().startswith(name_lower):
                return pokemon_name
                
        return None

    def get_pokemon_by_name(self, name):
        """Get raw dataframe row for a Pokemon"""
        matched = self.fuzzy_find_pokemon(name)
        if matched:
            return self.df[self.df['Name'] == matched].iloc[0]
        return None

    def add_pokemon(self, pokemon_data):
        """Add new Pokemon to dataframe"""
        new_row = pd.DataFrame([pokemon_data])
        self.df = pd.concat([self.df, new_row], ignore_index=True)
        self.pokemon_names = self.df['Name'].tolist()
        # Re-save to CSV? Optionally yes, but for now in-memory is fine or we can append
        # self.df.to_csv(self.csv_file, index=False)

    def _init_recommendation_system(self):
        """Initialize KNN model"""
        if self.df.empty: return

        # 1. Stats Features
        stat_columns = ['HP', 'Attack', 'Defense', 'Speed']
        stat_features = self.df[stat_columns].values
        stat_mean = stat_features.mean(axis=0)
        stat_std = stat_features.std(axis=0)
        normalized_stats = (stat_features - stat_mean) / stat_std

        # 2. Type Features (One-Hot)
        all_types = sorted(list(self.type_chart.keys()))
        type_features = []
        for _, row in self.df.iterrows():
            t_vec = [0] * len(all_types)
            if row['Type1'] in all_types: t_vec[all_types.index(row['Type1'])] = 1
            if row['Type2'] in all_types: t_vec[all_types.index(row['Type2'])] = 1
            type_features.append(t_vec)
        type_features = np.array(type_features)
        
        # 3. Generation & Legendary
        gen_features = self.df['Generation'].values.reshape(-1, 1) / self.df['Generation'].max()
        leg_features = self.df['Legendary'].astype(int).values.reshape(-1, 1)

        # Combine
        self.combined_features = np.hstack([
            normalized_stats * 1.0,
            type_features * 1.5,
            gen_features * 0.5,
            leg_features * 1.0
        ])
        
        self.knn_model = NearestNeighbors(n_neighbors=6, metric='euclidean')
        self.knn_model.fit(self.combined_features)
        self.normalized_features = self.combined_features
        print(f"✅ KNN Recommendation System ready")

    def get_similar_pokemon(self, pokemon_name, n=5):
        """Get similar Pokemon using KNN"""
        matched = self.fuzzy_find_pokemon(pokemon_name)
        if not matched: return None, "Pokemon not found"
        
        idx = self.df[self.df['Name'] == matched].index[0]
        distances, indices = self.knn_model.kneighbors([self.normalized_features[idx]])
        
        similar = []
        for i, dist in zip(indices[0][1:n+1], distances[0][1:n+1]):
            # Boundary check
            if i < len(self.df):
                pokemon = self.df.iloc[i]
                similar.append({
                    'name': pokemon['Name'],
                    'type': f"{pokemon['Type1']}/{pokemon['Type2']}" if pokemon['Type2'] else pokemon['Type1'],
                    'similarity': round(100 * (1 / (1 + dist)), 1)
                })
        return similar, matched

    def build_vector_db(self, semantic_model):
        """Build FAISS index using provided semantic model"""
        if not semantic_model or not HAS_FAISS: return

        
        print("🏗️ Building Vector Database...")
        self.pokemon_metadata = []
        descriptions = []

        for _, row in self.df.iterrows():
            type_str = row['Type1']
            if row['Type2']: type_str += f" and {row['Type2']}"
            legendary = "Legendary" if row['Legendary'] else ""
            
            desc = f"{row['Name']} is a {type_str} type Pokemon from Generation {row['Generation']}. "
            desc += f"It has {row['HP']} HP, {row['Attack']} Attack, {row['Defense']} Defense, and {row['Speed']} Speed. "
            if legendary: desc += f"It is a Legendary Pokemon. "
            
            if row['Type1'] in self.type_chart:
                weak = self.type_chart[row['Type1']]['weak']
                strong = self.type_chart[row['Type1']]['strong']
                desc += f"It is weak against {', '.join(weak)} and strong against {', '.join(strong)}. "

            descriptions.append(desc)
            self.pokemon_metadata.append({
                'name': row['Name'],
                'description': desc,
                'type': type_str,
                'stats': {k: row[k] for k in ['HP', 'Attack', 'Defense', 'Speed']}
            })

        embeddings = semantic_model.encode(descriptions)
        dimension = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatL2(dimension)
        self.faiss_index.add(np.array(embeddings).astype('float32'))
        print(f"✅ Vector Database built with {len(descriptions)} entries!")

    def vector_search(self, query_vector, k=3):
        """Search vector DB"""
        if not self.faiss_index: return []
        
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
