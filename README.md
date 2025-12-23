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

### AI-Powered Pokemon Assistant

[![Live Demo](https://img.shields.io/badge/🤗%20Live%20Demo-Hugging%20Face-yellow)](https://huggingface.co/spaces/DecryptVoid/Omnidex)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?logo=github)](https://github.com/DhanushPillay/Omnidex)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

*An intelligent chatbot that answers questions about Pokemon using Machine Learning and Google Gemini AI*

[**Try Live Demo →**](https://decryptvoid-omnidex.hf.space)

</div>

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔍 **Pokemon Info** | Get stats, types, and details for 800+ Pokemon |
| ⚔️ **Type Matchups** | Discover weaknesses and strengths |
| 📊 **Comparisons** | Side-by-side Pokemon stat comparisons |
| 🧬 **Evolutions** | View evolution chains with images |
| 🎯 **Recommendations** | Find similar Pokemon using ML |
| 💬 **Natural Chat** | Conversational AI powered by Gemini |

---

## 🛠️ Tech Stack

<table>
<tr>
<td><b>Backend</b></td>
<td>Python, Flask, Pandas, Scikit-learn</td>
</tr>
<tr>
<td><b>AI/ML</b></td>
<td>Google Gemini API, TF-IDF, KNN, Sentence Transformers</td>
</tr>
<tr>
<td><b>Frontend</b></td>
<td>HTML5, CSS3, Vanilla JavaScript</td>
</tr>
<tr>
<td><b>Deployment</b></td>
<td>Docker, Hugging Face Spaces, GitHub Actions</td>
</tr>
</table>

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- [Gemini API Key](https://aistudio.google.com/apikey) (free)

### Installation

```bash
# Clone the repository
git clone https://github.com/DhanushPillay/Omnidex.git
cd Omnidex

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Set API key
set GEMINI_API_KEY=your_key_here  # Windows
export GEMINI_API_KEY=your_key_here  # Linux/Mac

# Run
python app.py
```

Open **http://localhost:5000** in your browser.

---

## 💬 Example Queries

```
"Tell me about Pikachu"
"What is Charizard weak to?"
"Compare Mewtwo and Mew"
"Pokemon similar to Dragonite"
"How does Eevee evolve?"
"Which Pokemon has the highest attack?"
```

---

## 📁 Project Structure

```
Omnidex/
├── app.py                  # Flask server
├── Dockerfile              # Docker deployment
├── requirements.txt        # Dependencies
├── backend/
│   └── pokemon_chatbot.py  # AI/ML engine (46 methods)
├── data/
│   └── pokemon_data.csv    # Pokemon database
└── frontend/
    ├── templates/
    │   └── index.html      # Chat UI
    └── static/
        ├── style.css       # Styling
        └── script.js       # Frontend logic
```

---

## 🤖 ML Capabilities

- **Intent Classification** — TF-IDF + Semantic Embeddings
- **Recommendations** — K-Nearest Neighbors
- **Fuzzy Matching** — Handles typos and partial names
- **Conversational AI** — Google Gemini integration

---

## 📄 License

This project is licensed under the MIT License.

---

<div align="center">

**Made with ❤️ by [DhanushPillay](https://github.com/DhanushPillay)**

</div>
