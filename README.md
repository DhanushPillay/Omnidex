# Omnidex: The AI-Powered Pokémon Encyclopedia

<div align="center">

<img src="IMG/image.png" alt="Omnidex Logo" width="150">

![Omnidex Banner](https://img.shields.io/badge/Omnidex-AI%20Powered-red?style=for-the-badge&logo=pokemon)

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![Groq](https://img.shields.io/badge/AI-Groq%20LLaMA-F55036?logo=meta&logoColor=white)](https://groq.com)
[![Flask](https://img.shields.io/badge/Framework-Flask-000000?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

*An intelligent, conversational AI that bridges the gap between structured statistical data and rich Pokémon lore.*

[**Live Demo (Coming Soon)**](#) | [**Report Bug**](https://github.com/DhanushPillay/Omnidex/issues) | [**Request Feature**](https://github.com/DhanushPillay/Omnidex/issues)

</div>

---

## 🌟 Overview

**Omnidex** is not just another Pokédex — it's a sophisticated AI agent that combines **Machine Learning (ML)**, **Vector Search (RAG)**, and **Large Language Models (LLMs)** to provide a truly interactive Pokémon experience.

Unlike traditional wikis, Omnidex understands natural language. Ask about competitive strategies, deep lore, statistical comparisons, or even upload an image of a Pokémon to identify it. It uses **Groq's blazing-fast LLaMA models** to synthesize information from a local statistical database and real-time web searches into engaging, accurate narratives.

---

## ✨ Key Features

### 🧠 Intelligent Conversational AI
- **Natural Language Understanding** — Ask questions freely (e.g., *"Who is the strongest Fire type in Gen 1?"* or *"Tell me the tragic backstory of Cubone"*)
- **Context Awareness** — The AI remembers the conversation flow, allowing follow-up questions without repeating context
- **Persona-Based Responses** — Omnidex acts as an enthusiastic Pokémon Professor, making responses engaging and educational

### 📚 Deep Knowledge & Lore
- **Hybrid Retrieval System** — Combines a local CSV dataset (stats, types, evolutions) with real-time **DuckDuckGo Web Search** for obscure lore, anime history, and myths
- **Self-Learning Cache** — Learns from web searches, caching high-quality lore to improve future response times and accuracy
- **Fact-Checked Storytelling** — Filters out irrelevant game guides to prioritize canonical lore from the games and anime

### ⚔️ Competitive Analysis Engine
- **Type Matchup Calculator** — Instantly calculates weaknesses, resistances, and immunities
- **Stat Comparison** — Side-by-side comparison of any two Pokémon (e.g., *"Charizard vs. Blastoise"*)
- **Team Recommendations** — Uses **K-Nearest Neighbors (KNN)** to suggest similar Pokémon based on base stats and typing

### 👁️ Computer Vision
- **Image Recognition** — Upload an image of any Pokémon, and Omnidex will identify it and provide detailed information instantly

### 🎤 Voice Input (Experimental)
- **Speech Recognition** — Ask questions using your voice with built-in speech-to-text support

---

## 🛠️ Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Core AI** | **Groq (LLaMA 3)** | Ultra-fast natural language generation |
| **Backend** | **Python (Flask)** | REST API and server logic |
| **Vector DB** | **FAISS** | Fast similarity search for Pokémon recommendations |
| **Web Search** | **DuckDuckGo (ddgs)** | Real-time web retrieval for lore |
| **NLP** | **Sentence-Transformers** | Semantic understanding via embeddings |
| **Data** | **Pandas & NumPy** | High-performance data manipulation |
| **Frontend** | **HTML5, CSS3, JavaScript** | Responsive, modern chat interface |
| **Voice** | **SpeechRecognition** | Voice-to-text input support |

---

## 🚀 Quick Start Guide

### Prerequisites
- **Python 3.9+**
- **Groq API Key** (Free at [console.groq.com](https://console.groq.com))

### Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/DhanushPillay/Omnidex.git
   cd Omnidex
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv .venv
   
   # Windows
   .venv\Scripts\activate
   
   # Mac/Linux
   source .venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment**
   
   Create a `.env` file in the root directory:
   ```env
   GROQ_API_KEY=gsk_your-actual-api-key-here
   ```

### Running the Application

1. **Start the Server**
   ```bash
   python app.py
   ```

2. **Access the Interface**
   
   Open your browser and navigate to:
   ```
   http://localhost:5000
   ```

---

## 📁 Project Structure

```
Omnidex/
├── app.py                      # Main Flask Server & Routing
├── requirements.txt            # Python Dependencies
├── Dockerfile                  # Container Deployment
├── .env                        # API Keys (GitIgnored)
│
├── backend/
│   ├── pokemon_chatbot.py      # Main Orchestrator & Controller
│   ├── demo.py                 # Demo/Testing Script
│   └── services/
│       ├── __init__.py         # Service Exports
│       ├── data_service.py     # Data Loading, Vector DB, KNN
│       ├── intent_service.py   # Intent Classification & NLP
│       └── external_service.py # Groq AI, Web Search, PokeAPI
│
├── data/
│   ├── pokemon_data.csv        # Statistical Database (800+ Pokémon)
│   ├── intents.json            # NLP Training Data
│   ├── type_chart.json         # Type Effectiveness Rules
│   ├── evolution.json          # Evolution Chain Data
│   └── learned_cache.json      # Self-Learning Memory Cache
│
├── frontend/
│   ├── templates/
│   │   └── index.html          # Chat Interface
│   └── static/
│       ├── style.css           # Styling
│       └── script.js           # Frontend Logic
│
├── docs/
│   ├── AI_and_ML_Guide.md      # AI Architecture Deep Dive
│   └── Project_File_Guide.md   # Detailed File Breakdowns
│
├── IMG/
│   └── image.png               # Project Logo
│
└── uploads/                    # Uploaded Images (for Vision AI)
```

---

## 🧪 Example Queries

Try these example questions to explore Omnidex:

| Query Type | Example |
|------------|---------|
| **Stats** | *"What are Pikachu's base stats?"* |
| **Comparison** | *"Compare Charizard vs Blastoise"* |
| **Lore** | *"Tell me the origin story of Mewtwo"* |
| **Type Analysis** | *"What is Gyarados weak to?"* |
| **Recommendations** | *"Suggest Pokémon similar to Gengar"* |
| **Image Upload** | Upload any Pokémon image for instant identification |

---

## 🤝 Contributing

Contributions are what make the open-source community amazing! Any contributions are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

Distributed under the **MIT License**. See [LICENSE](LICENSE) for more information.

---

## 📚 Documentation

For detailed technical documentation, see:
- [AI and ML Guide](docs/AI_and_ML_Guide.md) — Deep dive into the AI architecture
- [Project File Guide](docs/Project_File_Guide.md) — Detailed file-by-file breakdown

---

<div align="center">

**Built with ❤️ using Python & Groq AI**

⭐ Star this repo if you find it useful!

</div>
