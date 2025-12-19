# PokéBot AI 🔥

An intelligent, **self-learning** Pokemon chatbot powered by Machine Learning. Ask questions naturally and the bot learns from the internet to expand its knowledge!

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![ML](https://img.shields.io/badge/ML-scikit--learn-orange)
![Status](https://img.shields.io/badge/Status-Active-green)

## ✨ Features

### 🤖 Machine Learning
- **TF-IDF Intent Classification** - Understands 165+ natural language patterns
- **Fuzzy Name Matching** - Handles typos (e.g., "charazard" → "Charizard")
- **K-Nearest Neighbors** - Recommends similar Pokemon based on stats
- **Conversation Memory** - Remembers context for follow-up questions

### 🧠 Self-Learning (NEW!)
- **PokeAPI Integration** - Automatically fetches unknown Pokemon from the internet
- **Auto-Database Updates** - Adds newly discovered Pokemon to CSV
- **Answer Caching** - Stores lore/story answers for instant future responses

### 📊 Pokemon Knowledge
- **Type Effectiveness** - Weakness/strength calculations for all 18 types
- **Evolution Chains** - Evolution info for 30+ popular Pokemon
- **649+ Pokemon** - Full stats, types, and generation data

### 🌐 Web Interface
- Modern, dark-themed chat UI
- Responsive design (mobile-friendly)
- Real-time stats dashboard

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. (Optional) Set Gemini API Key
```bash
# For AI-powered conversational responses
set GEMINI_API_KEY=your_key_here
```
Get a free key at: https://makersuite.google.com/app/apikey

### 3. Run the Web App
```bash
python app.py
```
Then open: http://localhost:5000

### 4. Or Run CLI Mode
```bash
python pokemon_chatbot.py
```

## 💬 Example Queries

### Basic Info
- "Tell me about Pikachu"
- "I want to know about Charizard"
- "What's Mewtwo?"

### Type Queries
- "Show me all fire type Pokemon"
- "List water types"
- "Electric Pokemon"

### Stats & Comparisons
- "Who has the highest attack?"
- "Compare Charizard and Blastoise"
- "Fastest Pokemon"

### Type Effectiveness (NEW!)
- "What is Charizard weak to?"
- "Pikachu weakness"
- "What type beats Dragon?"

### Evolution (NEW!)
- "How does Pikachu evolve?"
- "Eevee evolutions"
- "What level does Charmander evolve?"

### Recommendations (ML-Powered)
- "Recommend Pokemon similar to Pikachu"
- "Who is similar to Gengar?"

### Context-Aware Follow-ups (NEW!)
- "Tell me about Pikachu"
- "What about its defense?" ← Bot remembers Pikachu!
- "Is it legendary?"

### Self-Learning (NEW!)
Ask about newer Pokemon not in the database:
- "Tell me about Sprigatito" → Bot learns from PokeAPI and adds to database!

## 📁 Project Structure

```
├── app.py                 # Flask web server
├── pokemon_chatbot.py     # Core ML chatbot logic
├── pokemon_data.csv       # Pokemon database (auto-updates!)
├── learned_cache.json     # Cached web search answers
├── requirements.txt       # Python dependencies
├── templates/
│   └── index.html        # Chat interface
└── static/
    ├── style.css         # Styling
    └── script.js         # Frontend logic
```

## 🛠️ Technical Stack

| Component | Technology |
|-----------|------------|
| Backend | Python 3, Flask |
| ML | scikit-learn (TF-IDF, KNN) |
| NLP | Fuzzy matching, Intent classification |
| AI | Google Gemini (optional) |
| Web Search | DuckDuckGo Search |
| API | PokeAPI (for self-learning) |
| Frontend | HTML5, CSS3, JavaScript |

## 📈 ML Features Explained

### Intent Classification (TF-IDF)
```python
# Trained on 165+ example phrases per intent
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
# Uses cosine similarity to match user queries
```

### Pokemon Recommendations (KNN)
```python
# Finds similar Pokemon based on stat vectors
knn_model = NearestNeighbors(n_neighbors=5)
# Normalized HP, Attack, Defense, Speed features
```

### Self-Learning Flow
```
User asks about unknown Pokemon
        ↓
Check local database (miss)
        ↓
Query PokeAPI
        ↓
Extract stats & types
        ↓
Add to pokemon_data.csv
        ↓
Retrain recommendation model
        ↓
Respond to user
```

## 🔧 Configuration

### Environment Variables
| Variable | Description | Required |
|----------|-------------|----------|
| `GEMINI_API_KEY` | Google Gemini API key for AI responses | Optional |

### Files
| File | Purpose |
|------|---------|
| `pokemon_data.csv` | Main database (auto-updated by self-learning) |
| `learned_cache.json` | Cached answers from web searches |

## 📜 License

Free to use and modify!

## 🙏 Credits

- [PokeAPI](https://pokeapi.co/) - Pokemon data source for self-learning
- [DuckDuckGo](https://duckduckgo.com/) - Web search for lore/stories
- [Google Gemini](https://ai.google.dev/) - AI-powered responses
