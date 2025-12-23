# API Reference

This document describes the REST API endpoints for Omnidex.

---

## Base URL

- **Local**: `http://localhost:5000`
- **Production**: `https://decryptvoid-omnidex.hf.space`

---

## Endpoints

### 1. GET `/`

**Description**: Render the main chat interface.

**Response**: HTML page

---

### 2. POST `/ask`

**Description**: Send a question to the chatbot.

#### Request

```http
POST /ask
Content-Type: application/json

{
    "question": "Tell me about Pikachu"
}
```

#### Response

```json
{
    "success": true,
    "response": "⚡ Pikachu is an Electric-type Pokemon from Generation 1! It has 35 HP and 55 Attack...",
    "image_url": "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/other/official-artwork/25.png",
    "comparison_images": null,
    "pokemon_name": "Pikachu",
    "pokemon_context": {
        "name": "Pikachu",
        "type1": "Electric",
        "type2": null,
        "hp": 35,
        "attack": 55,
        "defense": 40,
        "speed": 90,
        "generation": 1,
        "legendary": false,
        "weak_to": ["Ground"],
        "strong_against": ["Water", "Flying"]
    },
    "lore_info": null,
    "evolution_chain": null
}
```

#### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `success` | boolean | Whether the request succeeded |
| `response` | string | The chatbot's natural language response |
| `image_url` | string | URL to Pokemon official artwork |
| `comparison_images` | array | For VS comparisons (see below) |
| `pokemon_name` | string | Detected Pokemon name |
| `pokemon_context` | object | Rich Pokemon data |
| `lore_info` | array | Web search results for lore queries |
| `evolution_chain` | array | Evolution data (see below) |

---

### Comparison Response

When user asks "Compare Charizard and Blastoise":

```json
{
    "comparison_images": [
        {
            "name": "Charizard",
            "image": "https://...charizard.png"
        },
        {
            "name": "Blastoise",
            "image": "https://...blastoise.png"
        }
    ]
}
```

---

### Evolution Chain Response

When user asks "How does Charmander evolve?":

```json
{
    "evolution_chain": [
        {
            "name": "Charmander",
            "level": 0,
            "sprite": "https://...charmander.png"
        },
        {
            "name": "Charmeleon",
            "level": 16,
            "sprite": "https://...charmeleon.png"
        },
        {
            "name": "Charizard",
            "level": 36,
            "sprite": "https://...charizard.png"
        }
    ]
}
```

---

### Error Response

```json
{
    "success": false,
    "error": "Error message here"
}
```

---

### 3. GET `/stats`

**Description**: Get overall Pokemon statistics.

#### Response

```json
{
    "total_pokemon": 800,
    "legendary_count": 70,
    "generations": 6,
    "unique_types": 18
}
```

---

## Example Usage

### cURL

```bash
curl -X POST http://localhost:5000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is Mewtwo weak to?"}'
```

### JavaScript

```javascript
const response = await fetch('/ask', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question: 'What is Mewtwo weak to?' })
});
const data = await response.json();
console.log(data.response);
```

### Python

```python
import requests

response = requests.post('http://localhost:5000/ask', 
    json={'question': 'What is Mewtwo weak to?'})
data = response.json()
print(data['response'])
```

---

## Query Types

The chatbot supports various query types:

| Query Type | Example |
|------------|---------|
| Pokemon Info | "Tell me about Gengar" |
| Weakness | "What is Charizard weak to?" |
| Strength | "What is Pikachu strong against?" |
| Evolution | "How does Eevee evolve?" |
| Comparison | "Compare Mewtwo and Mew" |
| Recommendation | "Pokemon similar to Dragonite" |
| Type Query | "List all Fire type Pokemon" |
| Legendary | "How many legendary Pokemon are there?" |
| Stats | "Which Pokemon has the highest attack?" |
| Lore | "What is the story behind Mewtwo?" |
