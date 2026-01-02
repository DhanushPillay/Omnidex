# Omnidex Project File Guide

This document provides a detailed breakdown of the file structure and the specific role of each file in the Omnidex codebase.

---

## 📂 Root Directory

| File | Type | Description |
|------|------|-------------|
| **`app.py`** | Python (Flask) | **The Application Entry Point.** Initializes the Flask server, defines API routes (`/ask`, `/upload`), handles session management for user context, and serves the HTML frontend. |
| **`requirements.txt`** | Config | Lists all Python dependencies required to run the project (e.g., `flask`, `openai`, `pandas`, `scikit-learn`). |
| **`.env`** | Config | Stores sensitive environment variables like `OPENAI_API_KEY`. **Never share this file.** |
| **`README.md`** | Markdown | The main landing page for the project. Contains setup instructions, feature overview, and installation guide. |

---

## 📂 Backend (`backend/`)

This directory contains the core logic, AI models, and service orchestrators.

| File | Description |
|------|-------------|
| **`pokemon_chatbot.py`** | **The Brain (Orchestrator).** This class initializes all services (`DataService`, `IntentService`, `ExternalService`) and coordinates the flow of a conversation. It decides whether to answer a question using local data, external APIs, or AI generation. |

### 📂 Services (`backend/services/`)

Refactored micro-modules that handle specific responsibilities.

| File | Description |
|------|-------------|
| **`data_service.py`** | **Database Manager.** <br>• Loads `pokemon_data.csv` and JSON files.<br>• Manages the **KNN Recommendation Engine** (finding similar Pokemon).<br>• Handles **Self-Learning persistence** (saving new Pokemon to CSV). |
| **`intent_service.py`** | **NLP Classifier.**<br>• Loads `intents.json` training data.<br>• Uses **Sentence-Transformers** (or TF-IDF) to understand what the user wants (e.g., "pokemon_info", "weakness"). |
| **`external_service.py`** | **Gateway to the World.**<br>• **OpenAI Integration**: Generates personas ("Professor") and analyzes images.<br>• **PokeAPI**: Fetches data for unknown Pokemon.<br>• **DuckDuckGo**: Searches the web for lore and myths. |

---

## 📂 Data (`data/`)

Static and dynamic datasets used by the backend.

| File | Description |
|------|-------------|
| **`pokemon_data.csv`** | **The Knowledge Base.** A CSV file containing stats, types, and descriptions for hundreds of Pokemon. This is the source of truth for the bot. |
| **`intents.json`** | **NLP Training Data.** Contains training phrases for the intent classifier (e.g., "Tell me about..." -> `pokemon_info`). |
| **`type_chart.json`** | **Game Rules.** A dictionary defining type effectiveness (e.g., Water is strong against Fire). |
| **`evolution.json`** | **Evolution Parsers.** Stores data about evolution chains and requirements. |

---

## 📂 Frontend (`frontend/`)

The user interface code.

### 📂 Static (`frontend/static/`)
| File | Description |
|------|-------------|
| **`script.js`** | **Frontend Logic.**<br>• Handles User Input & Typing Indicators.<br>• Manages **Auto-Scrolling** logic.<br>• Renders Chat Bubbles, Images, and Comparison Cards.<br>• Handles Web Speech API (Voice-to-Text). |
| **`style.css`** | **Styling.** Defines the dark-mode aesthetic, glassmorphism effects, and responsive layout for mobile/desktop. |

### 📂 Templates (`frontend/templates/`)
| File | Description |
|------|-------------|
| **`index.html`** | **The Structure.** The main HTML skeleton including the chat container, input area, and file upload modal. |

---

## 📂 Docs (`docs/`)

Project documentation.

| File | Description |
|------|-------------|
| **`AI_and_ML_Guide.md`** | Deep dive into the Artificial Intelligence architecture (Algorithms, Models, Strategies). |
| **`Project_File_Guide.md`** | (This File) A dictionary of all files in the project. |
