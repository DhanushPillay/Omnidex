---
title: Omnidex
emoji: 🔴
colorFrom: red
colorTo: yellow
sdk: docker
pinned: false
license: mit
---

<div align="center">

# 🔴 Omnidex

### The Ultimate AI-Powered Pokemon Assistant

[![Live Demo](https://img.shields.io/badge/🤗%20Live%20Demo-Hugging%20Face-yellow)](https://huggingface.co/spaces/DecryptVoid/Omnidex)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?logo=github)](https://github.com/DhanushPillay/Omnidex)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

*An intelligent, conversational chatbot that bridges the gap between structured Pokemon data and natural language using advanced Machine Learning and Google Gemini AI.*

[**Try Live Demo →**](https://decryptvoid-omnidex.hf.space)

</div>

---

## 📋 Overview

Omnidex is not just a Pokedex; it's a smart assistant capable of understanding complex queries, comparing Pokemon, and even identifying Pokemon from images. By combining traditional ML algorithms (KNN, TF-IDF) with modern Large Language Models (Gemini), Omnidex provides accurate, data-driven answers with a conversational touch.

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🧠 **Intelligent Querying** | Ask questions naturally (e.g., "Who beats Charizard?", "Strongest fire type?") |
| 📊 **Deep Analysis** | Get detailed stats, type effectiveness, and competitive insights |
| ⚔️ **Matchup Engine** | Instant weakness/resistance calculations and side-by-side comparisons |
| 🧬 **Smart Recommendations** | ML-powered suggestions based on stats and types (K-Nearest Neighbors) |
| 👁️ **Visual Recognition** | Upload an image to identify any Pokemon using Gemini Vision |
| �️ **Voice Interaction** | Speak to Omnidex and hear responses (Web Speech API) |

---

## � Dependencies & Technology

We use a robust stack of Python libraries to power Omnidex. Here's why each component is essential:

### Core Framework
- **`flask`**: A lightweight WSGI web application framework. It serves our web interface and API endpoints.
- **`gunicorn`**: A production-grade WSGI server used to run the Flask app in deployment environments like Hugging Face Spaces.
- **`python-dotenv`**: Manages environment variables (like API keys) securely by loading them from a `.env` file.

### Data Processing & Machine Learning
- **`pandas`**: The backbone of our data handling. It loads, cleans, and queries the massive `pokemon_data.csv` dataset efficiently.
- **`numpy`**: Provides support for large, multi-dimensional arrays and matrices, essential for our numerical calculations.
- **`scikit-learn`**: Powers our recommendation engine (KNN) and intent classification (TF-IDF Vectorizer).
- **`sentence-transformers`**: Generates semantic embeddings for user queries, allowing the bot to understand *meaning* rather than just matching keywords.
- **`faiss-cpu`**: A library for efficient similarity search of dense vectors, enabling fast retrieval of relevant context.

### AI & External APIs
- **`google-generativeai`**: The official client for Google's Gemini API. This gives Omnidex its conversational personality and vision capabilities.
- **`duckduckgo-search`**: Allows the bot to search the web for real-time information and lore not present in the static database.
- **`requests`**: Used for making HTTP requests to fetch external resources like sprite images.

### Multimedia Handling
- **`Pillow`**: The Python Imaging Library, used for processing and handling uploaded images before analysis.
- **`SpeechRecognition`**: Enables the backend to process audio data if needed (though primary speech-to-text is handled client-side).
- **`openpyxl`**: Read/write Excel 2010 xlsx/xlsm files, used if data sources are in Excel format.

---

## 🚀 Quick Start Guide

### Prerequisites
- Python 3.9 or higher
- A free [Gemini API Key](https://aistudio.google.com/apikey)

### Local Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/DhanushPillay/Omnidex.git
   cd Omnidex
   ```

2. **Create a virtual environment (Recommended)**
   ```bash
   python -m venv .venv
   
   # Windows
   .venv\Scripts\activate
   
   # Linux/macOS
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment**
   Create a `.env` file in the root directory and add your API Key:
   ```env
   GEMINI_API_KEY=your_actual_api_key_here
   ```

5. **Launch Omnidex**
   ```bash
   python app.py
   ```
   Open your browser and navigate to `http://localhost:5000`.

---

## � Project Architecture

```
Omnidex/
├── app.py                  # Main Flask application entry point
├── requirements.txt        # Python package dependencies
├── Dockerfile              # Container configuration for deployment
├── .env                    # Environment variables (not committed)
├── backend/
│   └── pokemon_chatbot.py  # Core Logic: hybrid ML engine & Gemini integration
├── data/
│   ├── pokemon_data.csv    # Structured dataset of 800+ Pokemon
│   └── learned_cache.json  # Persistent cache for web-learned knowledge
└── frontend/
    ├── templates/
    │   └── index.html      # Responsive Chat Interface
    └── static/
        ├── style.css       # Modern CSS variables & animations
        └── script.js       # Client-side logic, Speech API & WebSocket
```

---

## � Contributing

Contributions are welcome! Whether it's adding new features, improving the ML models, or refining the UI.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

<div align="center">

**Built with 💻 and ☕ by [DhanushPillay](https://github.com/DhanushPillay)**

</div>
