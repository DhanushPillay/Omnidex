# AI/ML Architecture in Omnidex

> Technical documentation for the AI and Machine Learning features powering the Omnidex Pokemon Chatbot

---

## Table of Contents

1. [Overview](#overview)
2. [AI/ML Technologies Used](#aiml-technologies-used)
3. [Intent Classification System](#1-intent-classification-system)
4. [Pokemon Recommendation Engine](#2-pokemon-recommendation-engine-knn)
5. [Natural Language Processing](#3-natural-language-processing)
6. [Gemini AI Integration](#4-gemini-ai-integration)
7. [Self-Learning System](#5-self-learning-system)
8. [Data Flow Architecture](#data-flow-architecture)
9. [File References](#file-references)

---

## Overview

Omnidex is an intelligent Pokemon chatbot that uses **multiple AI/ML techniques** to understand natural language questions and provide accurate, conversational responses. The system combines:

- **Traditional ML algorithms** (TF-IDF, KNN) for fast, reliable predictions
- **Transformer-based models** (Sentence-BERT) for semantic understanding
- **Generative AI** (Google Gemini) for natural conversation

```
┌─────────────────────────────────────────────────────────────────────┐
│                        OMNIDEX AI ARCHITECTURE                      │
├─────────────────────────────────────────────────────────────────────┤
│  User Question                                                      │
│       │                                                             │
│       ▼                                                             │
│  ┌─────────────────┐     ┌────────────────┐     ┌───────────────┐  │
│  │  NLP Layer      │ ──► │ Intent         │ ──► │ Data Lookup   │  │
│  │  (Semantic/     │     │ Classification │     │ (Pandas +     │  │
│  │   TF-IDF)       │     │ (ML Model)     │     │  PokeAPI)     │  │
│  └─────────────────┘     └────────────────┘     └───────────────┘  │
│                                                          │          │
│                                                          ▼          │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │                    GEMINI AI (Response Generation)            │ │
│  │          Converts data into natural, conversational text      │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                                    │                                │
│                                    ▼                                │
│                          Natural Response to User                   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## AI/ML Technologies Used

| Technology | Library | Purpose | File Location |
|------------|---------|---------|---------------|
| **TF-IDF Vectorizer** | `scikit-learn` | Text feature extraction for intent classification | `pokemon_chatbot.py:3` |
| **Cosine Similarity** | `scikit-learn` | Measure similarity between user query and training examples | `pokemon_chatbot.py:4` |
| **K-Nearest Neighbors (KNN)** | `scikit-learn` | Find similar Pokemon based on stats | `pokemon_chatbot.py:5` |
| **Sentence Transformers** | `sentence-transformers` | Semantic understanding of user queries | `pokemon_chatbot.py:17` |
| **Google Gemini AI** | `google-generativeai` | Natural language generation & conversation | `pokemon_chatbot.py:12` |
| **Fuzzy Matching** | `difflib` | Handle typos in Pokemon names | `pokemon_chatbot.py:6` |

---

## 1. Intent Classification System

### What is Intent Classification?

Intent classification determines **what the user wants** from their message. For example:
- "Tell me about Pikachu" → `pokemon_info` intent
- "Who is stronger, Charizard or Blastoise?" → `compare` intent
- "What is Electric type weak to?" → `weakness` intent

### How It Works in Omnidex

#### Step 1: Training Data
The system is trained on **200+ example phrases** covering different intents:

```python
# From pokemon_chatbot.py lines 131-331
self.intent_examples = [
    ("tell me about pikachu", "pokemon_info"),
    ("i want to know about charizard", "pokemon_info"),
    ("compare pikachu and raichu", "compare"),
    ("who has the highest attack", "stat_leader"),
    # ... 200+ more examples
]
```

**Intent Categories:**
| Intent | Example Questions | Count |
|--------|-------------------|-------|
| `pokemon_info` | "Tell me about Pikachu", "What is Mewtwo?" | 28 |
| `type_query` | "List all fire type", "Show water Pokemon" | 22 |
| `compare` | "Pikachu vs Raichu", "Who is stronger?" | 9 |
| `weakness` | "What is Pikachu weak to?" | 12 |
| `evolution` | "How does Pikachu evolve?" | 12 |
| `recommend` | "Pokemon similar to Charizard" | 11 |
| And more... | | |

#### Step 2: Text Vectorization

**Option A: Semantic Embeddings (Primary)**

Uses the `all-MiniLM-L6-v2` Sentence-BERT model to convert text into 384-dimensional vectors:

```python
# From pokemon_chatbot.py lines 337-343
self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
self.intent_embeddings = self.semantic_model.encode(self.intent_texts)
```

**Why Semantic Embeddings?**
- Understands **meaning**, not just keywords
- "What's Pikachu?" and "Tell me about Pikachu" have similar embeddings
- Handles variations in phrasing naturally

**Option B: TF-IDF (Fallback)**

If sentence-transformers isn't installed, falls back to TF-IDF:

```python
# From pokemon_chatbot.py lines 351-357
self.intent_vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words='english',
    ngram_range=(1, 2)  # Uses unigrams and bigrams
)
self.intent_vectors = self.intent_vectorizer.fit_transform(self.intent_texts)
```

**TF-IDF Explained:**
- **TF (Term Frequency)**: How often a word appears in a document
- **IDF (Inverse Document Frequency)**: How unique/important a word is across all documents
- Creates numerical vectors from text for ML algorithms

#### Step 3: Classification via Cosine Similarity

```python
# From pokemon_chatbot.py lines 466-489
def _classify_intent(self, question):
    # Encode the user's question
    question_embedding = self.semantic_model.encode([question])
    
    # Compare to all training examples
    similarities = cosine_similarity(question_embedding, self.intent_embeddings)
    
    # Find the best match
    best_idx = similarities.argmax()
    best_score = similarities[best_idx]
    
    # Only return if confidence is high enough
    if best_score > 0.5:  # 50% threshold for semantic
        return self.intent_labels[best_idx], best_score
```

**Cosine Similarity Formula:**
```
similarity = cos(θ) = (A · B) / (||A|| × ||B||)
```
- Result ranges from -1 to 1 (1 = identical, 0 = unrelated)

---

## 2. Pokemon Recommendation Engine (KNN)

### What is K-Nearest Neighbors?

KNN finds the "K" most similar items to a given input. In Omnidex, it finds Pokemon with similar stats.

### How It Works

#### Step 1: Feature Selection
Uses 4 key stats as features:
```python
# From pokemon_chatbot.py lines 361-363
stat_columns = ['HP', 'Attack', 'Defense', 'Speed']
self.stat_features = self.df[stat_columns].values
```

#### Step 2: Feature Normalization
Normalize so all stats are on the same scale:
```python
# From pokemon_chatbot.py lines 365-368
self.stat_mean = self.stat_features.mean(axis=0)
self.stat_std = self.stat_features.std(axis=0)
self.normalized_features = (self.stat_features - self.stat_mean) / self.stat_std
```

**Why Normalize?**
- Without normalization, high-value stats (like HP ~100) would dominate low-value ones (Speed ~50)
- Z-score normalization: `(value - mean) / std_dev`

#### Step 3: Build & Query KNN Model
```python
# From pokemon_chatbot.py lines 370-372
self.knn_model = NearestNeighbors(n_neighbors=6, metric='euclidean')
self.knn_model.fit(self.normalized_features)
```

```python
# From pokemon_chatbot.py lines 602-622
def _get_similar_pokemon(self, pokemon_name, n=5):
    # Find index of the Pokemon
    idx = self.df[self.df['Name'] == pokemon_name].index[0]
    
    # Query KNN model
    distances, indices = self.knn_model.kneighbors([self.normalized_features[idx]])
    
    # Convert distance to similarity percentage
    for i, dist in zip(indices[0][1:], distances[0][1:]):
        similarity = round(100 * (1 / (1 + dist)), 1)  # Higher = more similar
```

**Example Output:**
```
User: "Pokemon similar to Pikachu"
→ Raichu (92% similar), Jolteon (87%), Electabuzz (84%)...
```

---

## 3. Natural Language Processing

### Fuzzy Matching for Pokemon Names

Handles typos and variations in Pokemon names:

```python
# From pokemon_chatbot.py lines 491-540
def _fuzzy_find_pokemon(self, name, cutoff=0.75):
    # 1. Try exact match (case-insensitive)
    for pokemon_name in self.pokemon_names:
        if pokemon_name.lower() == name.lower():
            return pokemon_name
    
    # 2. Fuzzy matching (handles typos)
    matches = get_close_matches(name.capitalize(), self.pokemon_names, n=1, cutoff=0.75)
    
    # 3. Partial match (prefix)
    for pokemon_name in self.pokemon_names:
        if pokemon_name.lower().startswith(name.lower()):
            return pokemon_name
```

**Example:**
- "Pikichu" → Pikachu (fuzzy match)
- "chariz" → Charizard (prefix match)

### Pronoun Resolution

Understands context from previous messages:

```python
# From pokemon_chatbot.py lines 432-449
def _resolve_pronoun(self, text, context):
    pronouns = ['it', 'its', 'this one', 'that pokemon']
    
    if 'it' in text.lower():
        return context.get('last_pokemon')  # Returns previously discussed Pokemon
```

**Example Conversation:**
```
User: "Tell me about Pikachu"
Bot: "Pikachu is an Electric type..."
User: "What is it weak to?"  ← 'it' resolved to 'Pikachu'
Bot: "Pikachu is weak to Ground types"
```

### Semantic Pokemon Name Matching

When sentence-transformers is available, uses embeddings for Pokemon name matching:

```python
# From pokemon_chatbot.py lines 374-388
def _init_semantic_model(self):
    if self.use_semantic:
        self.pokemon_embeddings = self.semantic_model.encode(self.pokemon_names)
```

---

## 4. Gemini AI Integration

### Purpose

Google Gemini transforms raw data into natural, conversational responses.

### How It Works

```python
# From pokemon_chatbot.py lines 107-116
if GEMINI_API_KEY:
    self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
```

#### Response Generation

```python
# From pokemon_chatbot.py lines 626-683
def _make_conversational(self, data, user_question, context):
    prompt = f"""You are Omnidex, an expert AI Pokemon Professor.
Transform this raw data into a natural, conversational response.

User asked: "{user_question}"
Topic: {topic.upper()}

Data to transform:
{data}

Instructions:
- Keep it concise (2-4 sentences max)
- Use 1-2 relevant emoji
- Sound natural, exclude raw JSON formatting
"""
    response = self.gemini_model.generate_content(prompt)
    return response.text
```

**Before Gemini:**
```json
{"name": "Pikachu", "type": "Electric", "hp": 35, "attack": 55, "defense": 40}
```

**After Gemini:**
```
⚡ Pikachu is a beloved Electric-type Pokemon! With 35 HP and 55 Attack, 
this little powerhouse packs quite a punch. Its speed of 90 makes it 
one of the fastest Pokemon around!
```

#### General Knowledge Questions

For lore/story questions not in the database:

```python
# From pokemon_chatbot.py lines 685-719
def _answer_general_knowledge(self, question, context):
    prompt = f"""You are Omnidex, an expert AI Pokemon Professor.
Answer the user's question about Pokemon using your general knowledge.

User asked: "{question}"

Instructions:
- Answer ONLY if it relates to Pokemon
- Be friendly, enthusiastic and knowledgeable
"""
```

---

## 5. Self-Learning System

### Dynamic Pokemon Discovery

When a Pokemon isn't in the CSV, fetches from PokeAPI:

```python
# From pokemon_chatbot.py lines 511-540
# Try PokeAPI for unknown Pokemon
response = requests.get(f"https://pokeapi.co/api/v2/pokemon/{name}")
if response.status_code == 200:
    self._add_pokemon_from_api(data)  # Add to database dynamically!
```

### Web Search for Lore

Uses DuckDuckGo for Pokemon lore questions:

```python
# From pokemon_chatbot.py line 11
from duckduckgo_search import DDGS
```

---

## Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER INPUT                                     │
│                     "What Pokemon is similar to Pikachu?"                   │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  1. NLP PREPROCESSING                                                       │
│  ├── Lowercase & clean text                                                 │
│  ├── Resolve pronouns (if any)                                              │
│  └── Fuzzy match Pokemon names                                              │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  2. INTENT CLASSIFICATION (ML)                                              │
│  ├── Encode question using Sentence-BERT / TF-IDF                           │
│  ├── Cosine similarity with training examples                               │
│  └── Output: intent="recommend", confidence=0.89                            │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  3. DATA RETRIEVAL                                                          │
│  ├── Query Pokemon CSV database (pandas DataFrame)                          │
│  ├── If "recommend": Use KNN model to find similar Pokemon                  │
│  ├── If not found: Fetch from PokeAPI (self-learning)                       │
│  └── Output: [Raichu (92%), Jolteon (87%), ...]                             │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  4. RESPONSE GENERATION (Gemini AI)                                         │
│  ├── Transform raw data into conversational text                            │
│  ├── Add personality, emoji, context awareness                              │
│  └── Output: "Looking for Pokemon like Pikachu? ⚡ Try Raichu (92%          │
│       similar stats), Jolteon, or Electabuzz!"                              │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              RESPONSE TO USER                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## File References

| File | Purpose |
|------|---------|
| [`backend/pokemon_chatbot.py`](../backend/pokemon_chatbot.py) | Main AI/ML logic (1568 lines) |
| [`data/pokemon_data.csv`](../data/pokemon_data.csv) | Pokemon database for ML features |
| [`requirements.txt`](../requirements.txt) | Python dependencies |
| [`app.py`](../app.py) | Flask API serving the chatbot |

---

## Summary

| Component | ML Technique | Library | Purpose |
|-----------|-------------|---------|---------|
| **Intent Classification** | Semantic Embeddings + Cosine Similarity | `sentence-transformers`, `scikit-learn` | Understand what user wants |
| **Pokemon Recommendations** | K-Nearest Neighbors | `scikit-learn` | Find similar Pokemon by stats |
| **Name Matching** | Fuzzy String Matching | `difflib` | Handle typos |
| **Response Generation** | Large Language Model | `google-generativeai` | Natural conversation |
| **Self-Learning** | API Integration | `requests` | Expand knowledge dynamically |

---

*Technical documentation for the Omnidex project - Pokemon AI Chatbot*
