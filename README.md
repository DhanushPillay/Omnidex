# � Omnidex - The All-Knowing Pokémon AI

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-3776ab?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-2.0+-000000?style=for-the-badge&logo=flask&logoColor=white)
![ML](https://img.shields.io/badge/ML-scikit--learn-f7931e?style=for-the-badge&logo=scikit-learn&logoColor=white)
![AI](https://img.shields.io/badge/AI-Grok%20%2B%20Puter.js-8b5cf6?style=for-the-badge)

**An intelligent, self-learning Pokémon chatbot powered by Machine Learning and Grok AI**

[Features](#-features) • [Demo](#-demo) • [Installation](#-installation) • [Usage](#-usage) • [Tech Stack](#-tech-stack)

</div>

---

## ✨ Features

### � Advanced Machine Learning
- **Semantic NLP (sentence-transformers)** - Uses `all-MiniLM-L6-v2` for deep language understanding
- **TF-IDF Intent Classification** - 165+ training examples for accurate intent detection
- **Fuzzy Name Matching** - Handles typos & misspellings (e.g., "charazard" → "Charizard")
- **K-Nearest Neighbors (KNN)** - Recommends similar Pokémon based on stat vectors

### 🤖 Free AI-Powered Responses
- **Grok AI via Puter.js** - FREE unlimited conversational AI (no API key needed!)
- **Natural Conversations** - Responses feel like chatting with a real Pokémon expert
- **Context-Aware** - Remembers the last Pokémon discussed for follow-up questions

### 📚 Self-Learning Capabilities
- **PokeAPI Integration** - Automatically fetches unknown Pokémon from the internet
- **Auto-Database Updates** - Newly discovered Pokémon are added to the CSV database
- **Answer Caching** - Stores web search results for instant future responses

### 🎮 Rich Pokémon Knowledge
- **650+ Pokémon** - Complete stats, types, and generation data
- **Type Effectiveness** - Weakness/strength calculations for all 18 types
- **Evolution Chains** - Evolution info for 30+ popular Pokémon
- **Pokémon Images** - Official artwork sprites displayed in chat

---

## � Demo

### What You Can Ask:

| Query Type | Example |
|------------|---------|
| **Basic Info** | "Tell me about Pikachu", "What's Charizard?" |
| **Type Queries** | "Show me all fire types", "List water Pokémon" |
| **Stats** | "Who has the highest attack?", "Fastest Pokémon?" |
| **Comparisons** | "Compare Charizard and Blastoise" |
| **Weaknesses** | "What is Pikachu weak to?", "Fire type weakness" |
| **Evolution** | "How does Eevee evolve?", "Pikachu evolution" |
| **Recommendations** | "Recommend Pokémon like Gengar" |
| **Lore/Stories** | "Who is Ash Ketchum?", "Tell me about Misty" |
| **Follow-ups** | "What about its defense?" (after asking about a Pokémon) |

---

## 🚀 Installation

### Prerequisites
- Python 3.8+
- pip (Python package manager)

### Step 1: Clone the Repository
```bash
git clone https://github.com/DhanushPillay/Omnidex.git
cd Omnidex
```

### Step 2: Create Virtual Environment (Recommended)
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run the Application
```bash
python app.py
```

### Step 5: Open in Browser
Navigate to: **http://localhost:5000**

---

## � Usage

### Web Interface
Simply open the web interface and type your questions naturally:
- "I want to know about Mewtwo"
- "What type beats Dragon?"
- "Show me legendary Pokémon"

### Command Line (Optional)
```bash
python pokemon_chatbot.py
```

---

## 🛠 Tech Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| **Backend** | Python 3, Flask | Web server & API |
| **ML - NLP** | sentence-transformers | Semantic understanding |
| **ML - Classification** | scikit-learn (TF-IDF) | Intent classification |
| **ML - Recommendations** | scikit-learn (KNN) | Similar Pokémon suggestions |
| **AI** | Grok via Puter.js | Conversational responses |
| **Data API** | PokeAPI | Self-learning new Pokémon |
| **Web Search** | DuckDuckGo | Lore & story queries |
| **Frontend** | HTML5, CSS3, JavaScript | Modern chat interface |

---

## 📁 Project Structure

```
Omnidex/
│
├── app.py                       # 🚀 Entry point - Flask web server
├── requirements.txt             # 📦 Python dependencies
├── README.md                    # 📝 Documentation
│
├── backend/                     # 🐍 Backend Logic
│   ├── pokemon_chatbot.py       # Core ML chatbot (62KB)
│   └── demo.py                  # Testing script
│
├── data/                        # 📊 Data Files
│   ├── pokemon_data.csv         # Pokémon database (650+ entries)
│   └── learned_cache.json       # Cached web search answers
│
└── frontend/                    # 🎨 Web Interface
    ├── templates/
    │   └── index.html           # Main chat interface
    └── static/
        ├── style.css            # Dark theme styling
        └── script.js            # Grok AI integration
```

### File Descriptions

| File | Size | Description |
|------|------|-------------|
| `app.py` | 2.4KB | Flask server with `/ask` and `/stats` API endpoints |
| `pokemon_chatbot.py` | 62KB | Core ML: TF-IDF, KNN, sentence-transformers, PokeAPI integration |
| `pokemon_data.csv` | 26KB | 650+ Pokémon with stats, types, generations |
| `learned_cache.json` | 2.5KB | Auto-saved web search results for faster responses |
| `index.html` | 3KB | Responsive chat UI with Puter.js Grok integration |
| `style.css` | 8KB | Dark theme with animations and mobile support |
| `script.js` | 5.8KB | Frontend logic, Grok AI enhancement, image display |

---

## 🧠 How the ML Works

### 1. Intent Classification
```
User: "What is Pikachu weak to?"
      ↓
Semantic Embedding (sentence-transformers)
      ↓
Cosine Similarity with 165+ training examples
      ↓
Intent: "weakness" (98.8% confidence)
```

### 2. Self-Learning Flow
```
User asks about unknown Pokémon
      ↓
Check local database (miss)
      ↓
Query PokeAPI
      ↓
Extract stats, types, generation
      ↓
Add to pokemon_data.csv
      ↓
Retrain KNN model
      ↓
Respond with new data
```

### 3. Response Enhancement
```
Backend returns raw data
      ↓
Puter.js sends to Grok AI
      ↓
Grok makes it conversational
      ↓
Natural response displayed
```

---

## 🔧 Configuration

### Environment Variables (Optional)
| Variable | Description | Required |
|----------|-------------|----------|
| `GEMINI_API_KEY` | Google Gemini API (legacy fallback) | No |

> **Note:** Grok AI via Puter.js requires NO API key! It's completely free.

### Files
| File | Purpose |
|------|---------|
| `pokemon_data.csv` | Main database (auto-updated by self-learning) |
| `learned_cache.json` | Cached answers from web searches |

---

## 🤝 Contributing

Contributions are welcome! Feel free to:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

---

## 📜 License

This project is free to use and modify.

---

## 🙏 Credits

- **[PokeAPI](https://pokeapi.co/)** - Pokémon data source for self-learning
- **[Puter.js](https://puter.com)** - Free Grok AI access
- **[DuckDuckGo](https://duckduckgo.com/)** - Web search for lore/stories
- **[Hugging Face](https://huggingface.co/)** - sentence-transformers models

---

<div align="center">

**Built with ❤️ by [DhanushPillay](https://github.com/DhanushPillay)**

⭐ Star this repo if you found it helpful!

</div>
