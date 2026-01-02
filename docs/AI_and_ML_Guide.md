# AI/ML Architecture in Omnidex

> Technical documentation for the AI and Machine Learning features powering the Omnidex Pokemon Chatbot.

---

## Table of Contents

1. [Overview](#overview)
2. [Service-Oriented Architecture](#service-oriented-architecture)
3. [Intent Classification (NLP)](#1-intent-classification-nlp)
4. [Data Management & Recommendations](#2-data-management--recommendations)
5. [Generative AI (OpenAI)](#3-generative-ai-openai)
6. [Self-Learning System](#4-self-learning-system)
7. [File References](#file-references)

---

## Overview

Omnidex has evolved from a monolithic script into a **modular, service-oriented AI application**. It combines deterministic rules, statistical machine learning, and Large Language Models (LLMs) to provide a rich user experience.

**Key Features:**
- **Hybrid NLP**: Uses Sentence-BERT for semantic understanding, falling back to TF-IDF.
- **Adaptive Persona**: "Professor" persona that adjusts response length based on topic (Stats = Short, Lore = Detailed).
- **Self-Learning**: Automatically fetches unknown Pokemon from PokeAPI and persists them to the local database.
- **Visual Intelligence**: GPT-4o Vision for identifying Pokemon from uploaded images.

---

## Service-Oriented Architecture

The backend logic is decoupled into specialized services, orchestrated by a central controller.

```
Omnidex Backend
├── PokemonChatbot (Controller)
│   ├── Orchestrates the flow
│   └── Manages conversation context
│
├── Services
│   ├── IntentService (NLP)
│   │   ├── Loads data/intents.json
│   │   └── Predicts user intent (e.g., "pokemon_info", "weakness")
│   │
│   ├── DataService (Database)
│   │   ├── Manages CSV/JSON data
│   │   ├── Handles KNN Recommendations
│   │   └── Vector Database (FAISS)
│   │
│   └── ExternalService (API)
│       ├── OpenAI (Chat & Vision)
│       ├── DuckDuckGo (Web Search)
│       └── PokeAPI (Data Fetching)
```

---

## 1. Intent Classification (NLP)

**Files:** `backend/services/intent_service.py`, `data/intents.json`

The system determines *what* the user wants using a supervised classification approach.

### Training Data
Intents are stored in `data/intents.json`, decoupling data from code.
- **Sample Count**: ~200 examples.
- **Categories**: `pokemon_info`, `weakness`, `compare`, `lore`, `evolution`, etc.

### Algorithm
1.  **Semantic Search (Primary)**:
    - Uses `sentence-transformers` (all-MiniLM-L6-v2) to generate 384-dimensional embeddings.
    - Calculates **Cosine Similarity** between the user's query and stored intent vectors.
2.  **TF-IDF Fallback**:
    - If semantic models fail to load, falls back to Scikit-Learn's `TfidfVectorizer`.

---

## 2. Data Management & Recommendations

**Files:** `backend/services/data_service.py`, `data/pokemon_data.csv`

### Data Sources
- **CSV**: Core database (`pokemon_data.csv`).
- **JSON**: Static rules (`type_chart.json`, `evolution.json`).
- **Vector DB**: FAISS index built at runtime for semantic search over Pokemon descriptions.

### Recommendation Engine (KNN)
Uses **K-Nearest Neighbors** to find Pokemon with similar battle properties.
- **Features**: HP, Attack, Defense, Speed (Normalized with Z-Score).
- **Metric**: Euclidean Distance.
- **Usage**: "Show me Pokemon like Pikachu" -> Returns Raichu, Pichu, Plusle.

---

## 3. Generative AI (OpenAI)

**Files:** `backend/services/external_service.py`

We migrated from Google Gemini to **OpenAI (GPT-4o-mini)** for more reliable persona control and formatting.

### Adaptive Persona System
The `make_conversational` method dynamically builds the system prompt based on the **Topic**:

| Topic | Persona Instruction | Response Style |
|-------|---------------------|----------------|
| **Stats** | "Be analytical but little enthusiastic." | Short, concise (1-2 sentences). |
| **Lore** | "You are a storyteller sharing myths." | Detailed, immersive (3-4 sentences). |
| **Battle** | "Focus on strategy and matchups." | Tactical advice. |
| **General** | "Friendly Pokemon Professor." | Warm, emoji-rich. |

### Visual Analysis
Uses **GPT-4o Vision** to analyze uploaded images.
1.  User uploads image.
2.  Image encoded to Base64.
3.  Prompt: *"Identify this Pokemon. Return JSON with name, description, shiny_status."*
4.  Result is fuzzy-matched against the local DB.

---

## 4. Self-Learning System

**Files:** `backend/pokemon_chatbot.py`, `backend/services/external_service.py`

Omnidex can "learn" about new Pokemon it doesn't know (e.g., from new generations).

### Workflow
1.  **Detection**:
    - User asks "Tell me about Lechonk".
    - `IntentService` might classify as `None` or `pokemon_info`.
    - Database lookup fails (Lechonk is not in CSV).
2.  **Hypothesis**:
    - Heuristics identify "Lechonk" as a potential name (Title case or short unknown word).
3.  **Validation**:
    - Queries `ExternalService.fetch_from_pokeapi("lechonk")`.
    - PokeAPI returns valid JSON data.
4.  **Learning**:
    - `DataService.add_pokemon()` appends the new data to `pokemon_data.csv`.
    - The Pokemon is now permanently part of the dataset.

---

## File References

| Path | Description |
|------|-------------|
| `backend/pokemon_chatbot.py` | Central Orchestrator. Handles logic flow. |
| `backend/services/intent_service.py` | Handles ML/NLP intent classification. |
| `backend/services/data_service.py` | Manages CSV I/O, FAISS, and KNN models. |
| `backend/services/external_service.py` | Wrapper for OpenAI, DuckDuckGo, PokeAPI. |
| `data/intents.json` | Training examples for NLP. |
| `data/pokemon_data.csv` | Main database (persisted storage). |
