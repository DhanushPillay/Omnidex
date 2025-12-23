# Contributing to Omnidex

Thank you for your interest in contributing to Omnidex! 🎮

---

## Table of Contents
1. [Getting Started](#getting-started)
2. [Development Setup](#development-setup)
3. [Code Structure](#code-structure)
4. [Making Changes](#making-changes)
5. [Pull Request Process](#pull-request-process)
6. [Code Style](#code-style)
7. [Adding New Features](#adding-new-features)

---

## Getting Started

### Prerequisites
- Python 3.9+
- Git
- A Gemini API key (free at https://aistudio.google.com/apikey)

### Fork and Clone

```bash
# Fork the repository on GitHub first, then:
git clone https://github.com/YOUR_USERNAME/Omnidex.git
cd Omnidex
```

---

## Development Setup

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Set API key
set GEMINI_API_KEY=your_key_here  # Windows
export GEMINI_API_KEY=your_key_here  # Linux/Mac

# Run locally
python app.py
```

---

## Code Structure

```
📁 Omnidex/
├── 📄 app.py                    # Flask routes - ADD NEW ROUTES HERE
├── 📁 backend/
│   └── pokemon_chatbot.py      # AI logic - ADD NEW INTENTS HERE
├── 📁 frontend/
│   ├── templates/index.html    # UI structure
│   └── static/
│       ├── style.css           # Styling
│       └── script.js           # Frontend logic
└── 📁 data/
    └── pokemon_data.csv        # Pokemon database
```

---

## Making Changes

### Branch Naming

```bash
git checkout -b feature/your-feature-name
git checkout -b fix/bug-description
git checkout -b docs/documentation-update
```

### Commit Messages

Use clear, descriptive commit messages:

```bash
git commit -m "Add: Voice input feature"
git commit -m "Fix: Typo in Pikachu's stats"
git commit -m "Update: Improve intent classification accuracy"
git commit -m "Docs: Add API examples"
```

---

## Pull Request Process

1. **Fork** the repository
2. Create a **feature branch**
3. Make your changes
4. **Test** locally
5. **Push** to your fork
6. Create a **Pull Request**

### PR Checklist
- [ ] Code runs without errors
- [ ] Added tests (if applicable)
- [ ] Updated documentation
- [ ] No console.log or print statements left in

---

## Code Style

### Python

- Use PEP 8 style guide
- 4-space indentation
- Descriptive variable names
- Docstrings for functions

```python
def get_pokemon_info(self, name):
    """
    Get detailed information about a Pokemon.
    
    Args:
        name: Pokemon name (case-insensitive)
    
    Returns:
        str: Formatted Pokemon information
    """
    ...
```

### JavaScript

- Use `const` and `let`, avoid `var`
- camelCase for variables
- Use async/await for API calls

```javascript
async function sendMessage() {
    const response = await fetch('/ask', {...});
    const data = await response.json();
}
```

---

## Adding New Features

### Adding a New Intent

1. **Define training examples** in `pokemon_chatbot.py`:

```python
# In _init_intent_classifier
training_data.extend([
    ("What moves can Pikachu learn?", "moves"),
    ("Show me Charizard's moveset", "moves"),
    ("Pikachu's attacks", "moves"),
])
```

2. **Add handler** in `answer_question`:

```python
elif intent == "moves":
    pokemon_name = self._extract_pokemon_name(question_lower)
    if pokemon_name:
        data = self._get_pokemon_moves(pokemon_name)
        return self._make_conversational(data, original_question, context)
```

3. **Implement the method**:

```python
def _get_pokemon_moves(self, pokemon_name):
    """Get moveset for a Pokemon"""
    # Fetch from PokeAPI or return placeholder
    ...
```

### Adding a New Route

In `app.py`:

```python
@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'ok', 'pokemon_count': len(chatbot.df)})
```

### Adding Frontend Features

1. **HTML** - Add elements to `index.html`
2. **CSS** - Style in `style.css`
3. **JS** - Add logic in `script.js`

---

## Testing

### Manual Testing

```bash
python app.py
# Open http://localhost:5000
# Try various queries
```

### Test Queries

Test these before submitting:
- "Tell me about Pikachu"
- "What is Charizard weak to?"
- "Compare Mewtwo and Mew"
- "Pokemon similar to Dragonite"
- "How does Eevee evolve?"
- "pkachu" (typo test)

---

## Questions?

- Open an issue on GitHub
- Contact: DhanushPillay

Thank you for contributing! 🙏
