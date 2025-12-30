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

# Omnidex

### AI-Powered Pokemon Assistant

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Hugging%20Face-yellow)](https://huggingface.co/spaces/DecryptVoid/Omnidex)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?logo=github)](https://github.com/DhanushPillay/Omnidex)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

*An intelligent, conversational chatbot that bridges the gap between structured Pokemon data and natural language using advanced Machine Learning and Google Gemini AI.*

[**Try Live Demo →**](https://decryptvoid-omnidex.hf.space)

</div>

---

## Overview

Omnidex provides an interface for interacting with Pokemon data through natural language. By combining structured data analysis with Large Language Models, it allows users to query statistics, compare entities, and retrieve information without needing exact keyword matches.

## Key Features

| Feature | Description |
|---------|-------------|
| **Intelligent Querying** | Understanding of complex natural language queries (e.g., "Who beats Charizard?", "Strongest fire type?") |
| **Deep Analysis** | Detailed statistics, type effectiveness calculations, and competitive usage insights |
| **Matchup Engine** | Calculation of weaknesses and resistances with direct side-by-side comparisons |
| **Smart Recommendations** | Machine Learning-powered suggestions based on statistical similarity (K-Nearest Neighbors) |
| **Visual Recognition** | Identification of Pokemon from user-uploaded images using computer vision capabilities |
| **Voice Interaction** | Speech-to-text and text-to-speech support via the Web Speech API |

---

## Dependencies & Technology

The system relies on the following Python libraries:

### Core Framework
- **`flask`**: Serves the web interface and API endpoints.
- **`gunicorn`**: A production-grade WSGI server for deployment.
- **`python-dotenv`**: Manages sensitive environment variables.

### Data Processing & Machine Learning
- **`pandas`**: Handles efficient data loading and querying of the dataset.
- **`numpy`**: Supports numerical operations for stat calculations.
- **`scikit-learn`**: Implements the K-Nearest Neighbors algorithm for recommendations and TF-IDF for intent classification.
- **`sentence-transformers`**: Generates semantic embeddings to interpret user query intent.
- **`faiss-cpu`**: Enables efficient vector search for retrieving relevant context.

### AI & External APIs
- **`google-generativeai`**: Integrates Google's Gemini API for natural language generation and vision analysis.
- **`duckduckgo-search`**: Facilitates web search for information not present in the local database.
- **`requests`**: Handles HTTP requests for external resources.

---

## Quick Start Guide

### Prerequisites
- Python 3.9 or higher
- A Google Gemini API Key

### Local Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/DhanushPillay/Omnidex.git
   cd Omnidex
   ```

2. **Create a virtual environment**
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

5. **Launch Application**
   ```bash
   python app.py
   ```
   Navigate to `http://localhost:5000` in your web browser.

---

## Project Architecture

```
Omnidex/
├── app.py                  # Main application entry point
├── requirements.txt        # Package dependencies
├── Dockerfile              # Deployment configuration
├── .env                    # Environment variables
├── backend/
│   └── pokemon_chatbot.py  # Core logic and ML engine
├── data/
│   ├── pokemon_data.csv    # Structured dataset
│   └── learned_cache.json  # Search result cache
└── frontend/
    ├── templates/
    │   └── index.html      # User Interface
    └── static/
        ├── style.css       # Stylesheets
        └── script.js       # Client-side logic
```

---

## Limitations and Constraints

While Omnidex aims to provide accurate responses, users should be aware of the following limitations:

1.  **Dependency on External APIs**: The conversational features and image recognition rely on the Google Gemini API. Service interruptions or rate limits on the API will affect these functionalities.
2.  **Data Freshness**: The core statistics are derived from a static CSV dataset (`pokemon_data.csv`). New Pokemon generations or balance changes introduced in recent games may not be reflected immediately unless the dataset is manually updated or the web search fallback successfully retrieves the new data.
3.  **Hallucinations**: As with all Large Language Models, the AI may occasionally generate incorrect or "hallucinated" information, particularly for lore-based questions that are not strictly defined in the structured database.
4.  **Initial Load Time**: The first startup requires downloading machine learning models (Sentence Transformers), which may take several minutes depending on internet speed.
5.  **Browser Compatibility**: Voice interaction features rely on the Web Speech API, which has varying levels of support across different web browsers (currently best supported in Chrome and Edge).

---

## Contributing

Contributions are welcome. Please follow standard pull request procedures:

1. Fork the Project
2. Create your Feature Branch
3. Commit your Changes
4. Push to the Branch
5. Open a Pull Request

---

## License

Distributed under the MIT License. See `LICENSE` for more information.

---

<div align="center">

**Developed by [DhanushPillay](https://github.com/DhanushPillay)**

</div>
