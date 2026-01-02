import json
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# Try importing sentence_transformers
try:
    from sentence_transformers import SentenceTransformer
    USE_SEMANTIC = True
except ImportError:
    USE_SEMANTIC = False

class IntentService:
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        self.intent_examples = []
        self.intent_texts = []
        self.intent_labels = []
        
        self.use_semantic = False
        self.semantic_model = None
        self.intent_embeddings = None
        
        self.intent_vectorizer = None
        self.intent_vectors = None
        
        self._load_intents()
        self._init_models()

    def _load_intents(self):
        """Load intents from JSON"""
        try:
            with open(os.path.join(self.data_dir, 'intents.json'), 'r') as f:
                self.intent_examples = json.load(f)
            
            self.intent_texts = [ex[0] for ex in self.intent_examples]
            self.intent_labels = [ex[1] for ex in self.intent_examples]
            print(f"✅ Loaded {len(self.intent_examples)} intent examples")
        except Exception as e:
            print(f"❌ Failed to load intents: {e}")
            self.intent_examples = []

    def _init_models(self):
        """Initialize NLP models"""
        if USE_SEMANTIC:
            try:
                print("🧠 Loading Semantic NLP Model (all-MiniLM-L6-v2)...")
                self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
                self.intent_embeddings = self.semantic_model.encode(self.intent_texts)
                self.use_semantic = True
                print("✅ Semantic NLP Model loaded!")
            except Exception as e:
                print(f"⚠️ Semantic model failed: {e}")
                
        # Fallback to TF-IDF if semantic not available or failed
        if not self.use_semantic:
            print("⚠️ Using TF-IDF Fallback for intents")
            self.intent_vectorizer = TfidfVectorizer(
                lowercase=True, stop_words='english', ngram_range=(1, 2)
            )
            self.intent_vectors = self.intent_vectorizer.fit_transform(self.intent_texts)

    def predict(self, text):
        """Classify intent of text"""
        if not text or not self.intent_examples:
            return None, 0.0

        text_lower = text.lower()
        
        if self.use_semantic:
            # Semantic search
            embedding = self.semantic_model.encode([text_lower])
            similarities = cosine_similarity(embedding, self.intent_embeddings).flatten()
            threshold = 0.5
        else:
            # TF-IDF search
            vector = self.intent_vectorizer.transform([text_lower])
            similarities = cosine_similarity(vector, self.intent_vectors).flatten()
            threshold = 0.2
            
        best_idx = similarities.argmax()
        best_score = similarities[best_idx]
        
        if best_score > threshold:
            return self.intent_labels[best_idx], best_score
            
        return None, 0.0
