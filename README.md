# Omnidex: The AI-Powered Pokémon Encyclopedia

<div align="center">

![Omnidex Banner](https://img.shields.io/badge/Omnidex-AI%20Powered-red?style=for-the-badge&logo=pokemon)

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![OpenAI](https://img.shields.io/badge/AI-OpenAI%20GPT--4o-412991?logo=openai&logoColor=white)](https://openai.com)
[![Flask](https://img.shields.io/badge/Framework-Flask-000000?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

*An intelligent, conversational AI capability that bridges the gap between structured statistical data and rich Pokémon lore.*

[**Live Demo (Coming Soon)**](#) | [**Report Bug**](https://github.com/DhanushPillay/Omnidex/issues) | [**Request Feature**](https://github.com/DhanushPillay/Omnidex/issues)

</div>

---

## Overview

**Omnidex** is not just another Pokédex. It is a sophisticated AI agent that combines **Machine Learning (ML)**, **Vector Search (RAG)**, and **Large Language Models (LLMs)** to provide a truly interactive experience.

Unlike traditional wikis, Omnidex understands natural language. You can ask about competitive strategies, deep lore, statistical comparisons, or even upload an image of a Pokémon to identify it. It uses **OpenAI's GPT-4o** to synthesize information from a local statistical database and real-time web searches into engaging, accurate narratives.

---

## Key Features

### Intelligent Conversational AI
*   **Natural Language Understanding**: Ask questions freely (e.g., *"Who is the strongest Fire type in Gen 1?"* or *"Tell me the tragic backstory of Cubone"*).
*   **Context Awareness**: The AI remembers the conversation flow, allowing for follow-up questions without repeating context.
*   **Persona-Based Responses**: Omnidex acts as an enthusiastic Pokémon Professor, tailoring responses to be engaging and educational.

### Deep Knowledge & Lore
*   **Hybrid Retrieval System**: Combines a local CSV dataset (stats, types, evolutions) with real-time **DuckDuckGo Web Search** to find obscure lore, anime history, and myths.
*   **Learned Cache**: The system "learns" from web searches, caching high-quality lore to improve future response times and accuracy.
*   **Fact-Checked Storytelling**: Filters out irrelevant game guides (like PokeMMO) to prioritize canonical lore from the games and anime.

### Competitive Analysis Engine
*   **Type Matchup Calculator**: Instantly calculates weaknesses, resistances, and immunities.
*   **Stat Comparison**: Side-by-side comparison of any two Pokémon (e.g., *"Charizard vs. Blastoise"*).
*   **Team Recommendations**: Uses **K-Nearest Neighbors (KNN)** to suggest similar Pokémon based on base stats and typing.

### Computer Vision (Vision AI)
*   **Image Recognition**: Upload an image of any Pokémon, and Omnidex will use **GPT-4o Vision** to identify it and provide detailed information immediately.

---

## Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Core AI** | **OpenAI GPT-4o & GPT-4o-mini** | Natural language generation and image analysis. |
| **Backend** | **Python (Flask)** | REST API and server logic. |
| **Vector DB** | **FAISS (Facebook AI Similarity Search)** | Fast retrieval of similar Pokémon based on stats/description. |
| **Search** | **DuckDuckGo Search** | Real-time web retrieval for lore and latest info. |
| **NLP** | **Sentence-Transformers** | Semantic understanding of user queries (Embeddings). |
| **Data** | **Pandas & NumPy** | High-performance data manipulation for 800+ Pokémon. |
| **Frontend** | **HTML5, CSS3, JavaScript** | Responsive, modern chat interface. |

---

## Quick Start Guide

### Prerequisites
*   Python 3.9+
*   An **OpenAI API Key** (Required for AI features)

### Installation

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/DhanushPillay/Omnidex.git
    cd Omnidex
    ```

2.  **Create Virtual Environment**
    ```bash
    python -m venv .venv
    # Windows
    .venv\Scripts\activate
    # Mac/Linux
    source .venv/bin/activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Environment**
    Create a `.env` file in the root directory:
    ```env
    OPENAI_API_KEY=sk-proj-your-actual-api-key-here
    ```

### Running the Application

1.  **Start the Server**
    ```bash
    python app.py
    ```

2.  **Access the Interface**
    Open your browser and navigate to:
    ```
    http://localhost:5000
    ```

---

## Project Structure

```bash
Omnidex/
├── app.py                  # Main Flask Server & Routing
├── requirements.txt        # Python Dependencies
├── .env                    # API Keys (GitIgnored)
├── backend/
│   ├── services/           # Microservices (Intent, Data, External)
│   └── pokemon_chatbot.py  # Orchestrator & Controller
├── data/
│   ├── pokemon_data.csv    # Statistical Database
│   ├── intents.json        # NLP Training Data
│   └── type_chart.json     # Game Rules
└── frontend/
    ├── templates/
    │   └── index.html      # Chat Interface (HTML)
    └── static/
        ├── style.css       # Styling
        └── script.js       # Frontend Logic
```

---

## Contributing

Contributions are what make the open-source community an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

---

## License

Distributed under the **MIT License**. See [LICENSE](LICENSE) for more information.

---

<div align="center">

**Built using Python & OpenAI**

</div>

