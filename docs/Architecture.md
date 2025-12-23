# Architecture

This document describes the system architecture of Omnidex.

## Overview

Omnidex is a full-stack Pokemon AI chatbot with:
- **Flask backend** serving REST API
- **ML-powered intent classification**
- **Gemini AI for natural language**
- **Modern chat UI frontend**

---

## System Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER BROWSER                            │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    Frontend (HTML/JS/CSS)                   ││
│  │  • Chat UI with message bubbles                             ││
│  │  • Pokemon image display                                    ││
│  │  • VS comparison view                                       ││
│  │  • Evolution chain display                                  ││
│  └─────────────────────────────────────────────────────────────┘│
└────────────────────────────┬────────────────────────────────────┘
                             │ HTTP POST /ask
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Flask Server (app.py)                      │
│  • Route handling                                               │
│  • Session management for context                               │
│  • Pokemon image/metadata extraction                            │
│  • DuckDuckGo lore search                                       │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│               PokemonChatbot (pokemon_chatbot.py)               │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────────┐│
│  │Intent Classif│  │ KNN Recomm.  │  │ Gemini AI Integration  ││
│  │ (TF-IDF +    │  │ (sklearn)    │  │ • make_conversational  ││
│  │  Semantic)   │  │              │  │ • general_knowledge    ││
│  └──────────────┘  └──────────────┘  └────────────────────────┘│
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────────┐│
│  │Fuzzy Matching│  │ Type Chart   │  │ Evolution Data         ││
│  │ (difflib)    │  │ (18 types)   │  │ (50+ Pokemon)          ││
│  └──────────────┘  └──────────────┘  └────────────────────────┘│
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Data Sources                             │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────────┐│
│  │pokemon_data  │  │  PokeAPI     │  │ Gemini API             ││
│  │.csv (800+    │  │ (sprites)    │  │ (AI responses)         ││
│  │ Pokemon)     │  │              │  │                        ││
│  └──────────────┘  └──────────────┘  └────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

---

## Component Details

### 1. Frontend (`frontend/`)

| File | Purpose |
|------|---------|
| `templates/index.html` | Main HTML structure, chat container |
| `static/style.css` | Modern dark theme styling |
| `static/script.js` | AJAX requests, message rendering |
| `static/pokeball.png` | Logo and bot avatar |

**Key Frontend Features:**
- Responsive chat interface
- Auto-scroll on new messages
- Typing indicator animation
- Pokemon image display
- VS comparison cards
- Evolution chain visualization

---

### 2. Flask Server (`app.py`)

**Responsibilities:**
1. Serve the frontend HTML
2. Handle `/ask` POST requests
3. Maintain session-based conversation context
4. Extract Pokemon images from PokeAPI
5. Perform lore searches via DuckDuckGo
6. Return rich JSON responses

**Session Context Structure:**
```python
context = {
    'last_pokemon': None,        # Most recently discussed Pokemon
    'last_intent': None,         # e.g., 'compare', 'weakness'
    'conversation_history': [],  # List of exchanges
    'mentioned_pokemon': [],     # All Pokemon mentioned
    'compared_pokemon': [],      # Last comparison pair
    'current_topic': None,       # 'battle', 'lore', etc.
    'evolution_chain': None      # Current evolution display
}
```

---

### 3. AI Engine (`backend/pokemon_chatbot.py`)

The core of Omnidex with **1568 lines** and **46 methods**.

#### Class: PokemonChatbot

**Initialization:**
```python
chatbot = PokemonChatbot('data/pokemon_data.csv')
```

Loads:
- Pokemon CSV data into pandas DataFrame
- Initializes TF-IDF vectorizer for intents
- Loads semantic model (sentence-transformers)
- Sets up KNN for recommendations
- Configures Gemini API

**Main Entry Point:**
```python
response = chatbot.answer_question(question, context)
```

This method:
1. Resolves pronouns ("it" → actual Pokemon name)
2. Classifies intent using ML
3. Routes to appropriate handler
4. Formats response via Gemini
5. Updates conversation context

---

## Data Flow

### Request Lifecycle

```
1. User types "What is Pikachu weak to?"
   ↓
2. Frontend sends POST /ask {question: "..."}
   ↓
3. Flask extracts question from JSON
   ↓
4. Chatbot classifies intent → "weakness"
   ↓
5. Chatbot extracts Pokemon name → "Pikachu"
   ↓
6. Chatbot looks up Pikachu's type → "Electric"
   ↓
7. Chatbot checks TYPE_CHART → weak to ["Ground"]
   ↓
8. Gemini formats response naturally
   ↓
9. Flask adds image URL from PokeAPI
   ↓
10. JSON response sent to frontend
    ↓
11. Frontend renders message with image
```

---

## File Structure

```
📁 Omnidex/
├── 📄 app.py                     # Flask server (179 lines)
├── 📄 Dockerfile                 # Docker deployment
├── 📄 requirements.txt           # Dependencies
├── 📄 README.md                  # Project overview
│
├── 📁 backend/
│   ├── pokemon_chatbot.py       # AI engine (1568 lines)
│   └── __pycache__/
│
├── 📁 data/
│   └── pokemon_data.csv         # Pokemon database
│
├── 📁 frontend/
│   ├── 📁 templates/
│   │   └── index.html           # Chat UI
│   └── 📁 static/
│       ├── style.css            # Styling
│       ├── script.js            # Frontend JS
│       └── pokeball.png         # Assets
│
├── 📁 docs/                      # Documentation
│   ├── README.md
│   ├── Architecture.md
│   ├── API.md
│   ├── ML-Features.md
│   ├── Deployment.md
│   └── Contributing.md
│
└── 📁 .github/workflows/
    └── deploy-hf.yml            # CI/CD pipeline
```
