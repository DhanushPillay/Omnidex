# Omnidex - Pokemon AI Chatbot

A self-learning Pokemon chatbot powered by Machine Learning, featuring semantic NLP understanding, intelligent recommendations, and real-time web learning capabilities.

![Python](https://img.shields.io/badge/Python-3.8+-3776ab?style=flat-square)
![Flask](https://img.shields.io/badge/Flask-2.0+-000000?style=flat-square)
![ML](https://img.shields.io/badge/ML-scikit--learn-f7931e?style=flat-square)

---

## Overview

Omnidex is an intelligent Pokemon assistant that combines multiple machine learning techniques to understand natural language queries and provide accurate, conversational responses. The system learns from user interactions and can discover new Pokemon automatically via the PokeAPI.

---

## Architecture

### System Flow

```
User Query
    |
    v
+-------------------+
|  Intent           |  Semantic embeddings (sentence-transformers)
|  Classification   |  or TF-IDF with cosine similarity
+-------------------+
    |
    v
+-------------------+
|  Entity           |  Fuzzy matching with PokeAPI fallback
|  Extraction       |  for unknown Pokemon
+-------------------+
    |
    v
+-------------------+
|  Response         |  Grok AI (Puter.js) for natural
|  Generation       |  conversational output
+-------------------+
    |
    v
Response to User
```

---

## Machine Learning Components

### 1. Intent Classification

The system classifies user intent using semantic embeddings:

- **Model**: `all-MiniLM-L6-v2` (sentence-transformers)
- **Fallback**: TF-IDF vectorizer with n-grams
- **Training Data**: 165+ labeled examples across 15 intent categories
- **Method**: Cosine similarity matching

**Supported Intents:**
| Intent | Example Query |
|--------|---------------|
| `pokemon_info` | "Tell me about Pikachu" |
| `type_query` | "List fire type Pokemon" |
| `weakness` | "What is Charizard weak to?" |
| `evolution` | "How does Eevee evolve?" |
| `compare` | "Compare Charizard vs Blastoise" |
| `recommend` | "Pokemon similar to Gengar" |
| `context_*` | Follow-up questions |

### 2. Entity Recognition

Pokemon names are extracted using a multi-stage matching process:

```python
1. Exact match (case-insensitive)
2. Fuzzy matching (difflib, cutoff=0.75)
3. Prefix matching
4. PokeAPI lookup (self-learning)
```

### 3. Recommendation Engine

Similar Pokemon are suggested using K-Nearest Neighbors:

- **Features**: Normalized stat vectors (HP, Attack, Defense, Speed)
- **Algorithm**: KNN with Euclidean distance
- **Output**: Top 5 similar Pokemon with similarity scores

### 4. Self-Learning System

The chatbot automatically expands its knowledge:

```
Unknown Pokemon Query
        |
        v
Query PokeAPI (/api/v2/pokemon/{name})
        |
        v
Extract: name, types, stats, generation
        |
        v
Append to pokemon_data.csv
        |
        v
Retrain KNN model
```

**Web Search Caching:**
- Story/lore queries are searched via DuckDuckGo
- Results are cached in `learned_cache.json`
- Future identical queries return cached results

---

## Type Effectiveness System

Complete type chart for all 18 Pokemon types:

```python
TYPE_CHART = {
    'Fire': {
        'strong': ['Grass', 'Ice', 'Bug', 'Steel'],
        'weak': ['Water', 'Ground', 'Rock'],
        'resist': ['Fire', 'Grass', 'Ice', 'Bug', 'Steel', 'Fairy'],
        'immune': []
    },
    # ... 17 more types
}
```

---

## Project Structure

```
Omnidex/
|
|-- app.py                       # Flask web server (entry point)
|-- requirements.txt             # Dependencies
|-- README.md
|
|-- backend/
|   |-- pokemon_chatbot.py       # Core ML logic
|   +-- demo.py                  # Testing script
|
|-- data/
|   |-- pokemon_data.csv         # Pokemon database (650+ entries)
|   +-- learned_cache.json       # Cached web search results
|
+-- frontend/
    |-- templates/
    |   +-- index.html           # Chat interface
    +-- static/
        |-- style.css            # Styling
        +-- script.js            # Grok AI integration
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serves the chat interface |
| `/ask` | POST | Processes user queries |
| `/stats` | GET | Returns database statistics |

**Request/Response Example:**

```json
// POST /ask
// Request
{ "question": "Tell me about Pikachu" }

// Response
{
  "success": true,
  "response": "Pikachu is an Electric type Pokemon with 35 HP...",
  "image_url": "https://raw.githubusercontent.com/.../25.png",
  "pokemon_name": "Pikachu"
}
```

---

## Technologies

| Component | Technology |
|-----------|------------|
| Backend | Python 3, Flask |
| NLP | sentence-transformers, TF-IDF |
| ML | scikit-learn (KNN, cosine similarity) |
| AI | Grok via Puter.js |
| Data Source | PokeAPI |
| Web Search | DuckDuckGo |

---

## Data Sources

- **Pokemon Database**: 650+ Pokemon with stats, types, generations
- **PokeAPI**: Real-time data for unknown Pokemon
- **Type Chart**: All 18 types with effectiveness relationships
- **Evolution Data**: 30+ popular Pokemon evolution chains

---

## License

Free to use and modify.

---

**Author**: [DhanushPillay](https://github.com/DhanushPillay)
