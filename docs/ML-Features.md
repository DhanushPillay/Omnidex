# Machine Learning Features

This document explains the ML/AI capabilities of Omnidex in detail.

---

## Overview

Omnidex uses multiple machine learning techniques:

1. **Intent Classification** - TF-IDF + Semantic Embeddings
2. **Pokemon Recommendations** - K-Nearest Neighbors
3. **Fuzzy Name Matching** - String similarity algorithms
4. **Natural Language Generation** - Google Gemini API
5. **Semantic Understanding** - Sentence Transformers

---

## 1. Intent Classification

### Purpose
Determine what the user wants to do (get info, compare, find weaknesses, etc.)

### Algorithms

#### TF-IDF (Term Frequency-Inverse Document Frequency)
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

vectorizer = TfidfVectorizer()
intent_vectors = vectorizer.fit_transform(training_examples)

# Classify new question
question_vector = vectorizer.transform([user_question])
similarities = cosine_similarity(question_vector, intent_vectors)
best_intent = intent_labels[similarities.argmax()]
```

#### Semantic Embeddings (sentence-transformers)
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(training_examples)

# Classify with better understanding
question_embedding = model.encode([user_question])
similarities = cosine_similarity(question_embedding, embeddings)
```

### Supported Intents

| Intent | Example Queries |
|--------|-----------------|
| `pokemon_info` | "Tell me about Pikachu", "What is Charizard?" |
| `type_query` | "List Fire type Pokemon", "Show me Electric types" |
| `weakness` | "What is Pikachu weak to?", "Charizard weaknesses" |
| `strength` | "What is Water strong against?" |
| `compare` | "Compare Mewtwo and Mew", "Pikachu vs Raichu" |
| `recommend` | "Pokemon similar to Dragonite" |
| `evolution` | "How does Eevee evolve?", "Charmander evolution" |
| `stat_leader` | "Fastest Pokemon", "Highest attack" |
| `count_legendary` | "How many legendary Pokemon?" |
| `count_type` | "How many Fire types?" |
| `count_total` | "How many Pokemon are there?" |
| `generation_query` | "Gen 1 Pokemon", "Generation 3 starters" |
| `dual_type` | "Dual type Pokemon" |
| `context_stat` | "What's its defense?" (follow-up) |
| `context_evolve` | "Does it evolve?" (follow-up) |

### Training Data

Each intent has 5-10 example phrases:
```python
training_data = [
    ("Tell me about Pikachu", "pokemon_info"),
    ("What can you tell me about Charizard?", "pokemon_info"),
    ("Give me information on Mewtwo", "pokemon_info"),
    ("What is Pikachu weak to?", "weakness"),
    ("Charizard weaknesses", "weakness"),
    # ... 200+ examples
]
```

---

## 2. Pokemon Recommendations (KNN)

### Purpose
Find Pokemon similar to a given one based on stats.

### Implementation
```python
from sklearn.neighbors import NearestNeighbors

# Features: HP, Attack, Defense, Sp.Atk, Sp.Def, Speed
features = df[['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']]

# Normalize features
normalized = (features - features.mean()) / features.std()

# Train KNN
knn = NearestNeighbors(n_neighbors=6, algorithm='ball_tree')
knn.fit(normalized)

# Find similar Pokemon
distances, indices = knn.kneighbors([pikachu_stats])
similar_pokemon = df.iloc[indices[0]]  # Top 5 similar
```

### Similarity Calculation
```python
# Convert distance to percentage similarity
similarity_score = max(0, 100 - (distance * 20))
```

---

## 3. Fuzzy Name Matching

### Purpose
Handle typos and partial names (e.g., "pkachu" → "Pikachu")

### Algorithm
```python
from difflib import get_close_matches

def fuzzy_find_pokemon(name):
    all_names = df['Name'].tolist()
    
    # First: exact match
    if name in all_names:
        return name
    
    # Second: case-insensitive match
    for n in all_names:
        if n.lower() == name.lower():
            return n
    
    # Third: fuzzy match with difflib
    matches = get_close_matches(name, all_names, n=1, cutoff=0.75)
    if matches:
        return matches[0]
    
    # Fourth: substring match
    for n in all_names:
        if name.lower() in n.lower():
            return n
    
    return None
```

### Special Cases
- "mega charizard" → "CharizardMega Charizard X"
- "pika" → "Pikachu"
- "mew2" → "Mewtwo"

---

## 4. Natural Language Generation (Gemini)

### Purpose
Convert raw data into conversational responses.

### Integration
```python
import google.generativeai as genai

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-pro')

def make_conversational(data, question, context):
    prompt = f"""You are Omnidex, a Pokemon expert.
    
    Transform this data into a natural response:
    Data: {data}
    User asked: {question}
    Topic: {context.get('current_topic')}
    
    Instructions:
    - Be concise (2-4 sentences)
    - Use 1-2 relevant emoji
    - Sound natural and enthusiastic
    """
    
    response = model.generate_content(prompt)
    return response.text
```

### Topic-Specific Personalities
- **Battle**: Sound like a Gym Leader
- **Lore**: Be mysterious and storytelling
- **Evolution**: Express excitement about growth
- **Stats**: Be analytical but enthusiastic

---

## 5. Semantic Understanding

### Purpose
Understand meaning beyond keywords.

### Model
- **Model**: `all-MiniLM-L6-v2` (sentence-transformers)
- **Dimensions**: 384
- **Size**: ~23MB

### Benefits
- "What's Pikachu susceptible to?" matches "weakness" intent
- "Electric rodent Pokemon" matches Pikachu
- Context-aware responses

---

## 6. Context Management

### Pronoun Resolution
```python
def resolve_pronoun(text, context):
    pronouns = ['it', 'its', 'this one', 'that one']
    last_pokemon = context.get('last_pokemon')
    
    for pronoun in pronouns:
        if pronoun in text.lower() and last_pokemon:
            return text.replace(pronoun, last_pokemon)
    
    return text
```

### Conversation History
```python
context = {
    'conversation_history': [
        {'role': 'user', 'content': 'Tell me about Pikachu'},
        {'role': 'assistant', 'content': 'Pikachu is...'},
        {'role': 'user', 'content': 'What is it weak to?'}
    ]
}
```

---

## 7. Type Chart

### Implementation
```python
TYPE_CHART = {
    'Normal': {'weak': ['Fighting'], 'immune': ['Ghost'], 'resist': [], 'strong': []},
    'Fire': {'weak': ['Water', 'Ground', 'Rock'], 'resist': ['Fire', 'Grass', 'Ice', 'Bug', 'Steel', 'Fairy'], 'strong': ['Grass', 'Ice', 'Bug', 'Steel']},
    'Water': {'weak': ['Electric', 'Grass'], 'resist': ['Fire', 'Water', 'Ice', 'Steel'], 'strong': ['Fire', 'Ground', 'Rock']},
    # ... all 18 types
}
```

### Dual-Type Calculation
For Pokemon with two types, weaknesses are combined and resistances cancel out.

---

## Performance Metrics

| Feature | Accuracy | Speed |
|---------|----------|-------|
| Intent Classification (TF-IDF) | ~85% | <10ms |
| Intent Classification (Semantic) | ~95% | ~50ms |
| Fuzzy Matching | ~98% | <5ms |
| KNN Recommendations | N/A | ~15ms |
| Gemini Response | N/A | ~1-2s |

---

## Dependencies

```txt
scikit-learn>=0.24.0      # TF-IDF, KNN, cosine similarity
sentence-transformers>=2.2.0  # Semantic embeddings
pandas>=1.3.0             # Data handling
numpy>=1.21.0             # Numerical operations
google-generativeai>=0.3.0  # Gemini API
```
