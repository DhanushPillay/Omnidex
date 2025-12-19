"""
Vercel Serverless Function - Lightweight Version
Uses PokeAPI directly instead of local CSV + ML
"""
from flask import Flask, request, jsonify, send_from_directory
import requests
import os
import json

app = Flask(__name__)

# Pokemon type chart for weaknesses/strengths
TYPE_CHART = {
    'normal': {'weak': ['fighting'], 'strong': []},
    'fire': {'weak': ['water', 'ground', 'rock'], 'strong': ['grass', 'ice', 'bug', 'steel']},
    'water': {'weak': ['electric', 'grass'], 'strong': ['fire', 'ground', 'rock']},
    'electric': {'weak': ['ground'], 'strong': ['water', 'flying']},
    'grass': {'weak': ['fire', 'ice', 'poison', 'flying', 'bug'], 'strong': ['water', 'ground', 'rock']},
    'ice': {'weak': ['fire', 'fighting', 'rock', 'steel'], 'strong': ['grass', 'ground', 'flying', 'dragon']},
    'fighting': {'weak': ['flying', 'psychic', 'fairy'], 'strong': ['normal', 'ice', 'rock', 'dark', 'steel']},
    'poison': {'weak': ['ground', 'psychic'], 'strong': ['grass', 'fairy']},
    'ground': {'weak': ['water', 'grass', 'ice'], 'strong': ['fire', 'electric', 'poison', 'rock', 'steel']},
    'flying': {'weak': ['electric', 'ice', 'rock'], 'strong': ['grass', 'fighting', 'bug']},
    'psychic': {'weak': ['bug', 'ghost', 'dark'], 'strong': ['fighting', 'poison']},
    'bug': {'weak': ['fire', 'flying', 'rock'], 'strong': ['grass', 'psychic', 'dark']},
    'rock': {'weak': ['water', 'grass', 'fighting', 'ground', 'steel'], 'strong': ['fire', 'ice', 'flying', 'bug']},
    'ghost': {'weak': ['ghost', 'dark'], 'strong': ['psychic', 'ghost']},
    'dragon': {'weak': ['ice', 'dragon', 'fairy'], 'strong': ['dragon']},
    'dark': {'weak': ['fighting', 'bug', 'fairy'], 'strong': ['psychic', 'ghost']},
    'steel': {'weak': ['fire', 'fighting', 'ground'], 'strong': ['ice', 'rock', 'fairy']},
    'fairy': {'weak': ['poison', 'steel'], 'strong': ['fighting', 'dragon', 'dark']},
}

# Conversation context
context = {'last_pokemon': None}

def get_pokemon_from_api(name):
    """Fetch Pokemon data from PokeAPI"""
    try:
        clean_name = name.lower().replace(' ', '-').replace('.', '').replace("'", '')
        resp = requests.get(f"https://pokeapi.co/api/v2/pokemon/{clean_name}", timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            types = [t['type']['name'] for t in data['types']]
            stats = {s['stat']['name']: s['base_stat'] for s in data['stats']}
            sprite = data['sprites']['other']['official-artwork']['front_default']
            
            return {
                'name': data['name'].title(),
                'id': data['id'],
                'types': types,
                'hp': stats.get('hp', 0),
                'attack': stats.get('attack', 0),
                'defense': stats.get('defense', 0),
                'speed': stats.get('speed', 0),
                'sprite': sprite,
                'weak_to': TYPE_CHART.get(types[0], {}).get('weak', []),
                'strong_against': TYPE_CHART.get(types[0], {}).get('strong', [])
            }
    except Exception as e:
        print(f"PokeAPI error: {e}")
    return None

def generate_response(pokemon, question):
    """Generate a natural response about the Pokemon"""
    if not pokemon:
        return "I couldn't find that Pokemon. Try checking the spelling!"
    
    q = question.lower()
    name = pokemon['name']
    types = '/'.join([t.title() for t in pokemon['types']])
    
    # Weakness query
    if any(w in q for w in ['weak', 'weakness', 'vulnerable', 'counter']):
        weak = ', '.join([w.title() for w in pokemon['weak_to']]) if pokemon['weak_to'] else 'nothing specifically'
        return f"⚡ {name} ({types}) is weak to: {weak}. Watch out for those types in battle!"
    
    # Strength query
    if any(w in q for w in ['strong', 'effective', 'beat', 'counter']):
        strong = ', '.join([s.title() for s in pokemon['strong_against']]) if pokemon['strong_against'] else 'various types'
        return f"💪 {name} is super effective against: {strong}. Great offensive coverage!"
    
    # Stats query
    if any(w in q for w in ['stat', 'hp', 'attack', 'defense', 'speed']):
        return f"📊 {name} Stats: HP {pokemon['hp']}, Attack {pokemon['attack']}, Defense {pokemon['defense']}, Speed {pokemon['speed']}"
    
    # General info
    return f"🔥 {name} is a {types} type Pokemon! HP: {pokemon['hp']}, Attack: {pokemon['attack']}, Defense: {pokemon['defense']}, Speed: {pokemon['speed']}. It's weak to {', '.join([w.title() for w in pokemon['weak_to'][:3]])}."

def extract_pokemon_name(text):
    """Extract Pokemon name from text"""
    # Common Pokemon for quick lookup
    common = ['pikachu', 'charizard', 'mewtwo', 'mew', 'gengar', 'dragonite', 'blastoise', 
              'venusaur', 'eevee', 'snorlax', 'gyarados', 'alakazam', 'machamp', 'raichu',
              'bulbasaur', 'squirtle', 'charmander', 'jigglypuff', 'meowth', 'psyduck']
    
    text_lower = text.lower()
    
    # Check common Pokemon first
    for poke in common:
        if poke in text_lower:
            return poke
    
    # Try each word
    words = text_lower.replace('?', '').replace('!', '').split()
    skip_words = ['tell', 'me', 'about', 'what', 'is', 'who', 'the', 'a', 'an', 'weak', 'to', 
                  'strong', 'against', 'compare', 'and', 'evolution', 'evolve', 'type', 'stats']
    
    for word in words:
        if len(word) > 2 and word not in skip_words:
            # Test if it's a valid Pokemon via API
            pokemon = get_pokemon_from_api(word)
            if pokemon:
                return word
    
    return None

@app.route('/ask', methods=['POST'])
def ask():
    """Handle question requests"""
    try:
        data = request.get_json()
        question = data.get('question', '')
        
        if not question:
            return jsonify({'error': 'No question provided'}), 400
        
        # Extract Pokemon name
        pokemon_name = extract_pokemon_name(question)
        
        # Use context if no Pokemon found
        if not pokemon_name and context.get('last_pokemon'):
            pokemon_name = context['last_pokemon']
        
        # Get Pokemon data from API
        pokemon = get_pokemon_from_api(pokemon_name) if pokemon_name else None
        
        # Update context
        if pokemon:
            context['last_pokemon'] = pokemon['name'].lower()
        
        # Generate response
        response = generate_response(pokemon, question)
        
        return jsonify({
            'success': True,
            'response': response,
            'image_url': pokemon['sprite'] if pokemon else None,
            'pokemon_name': pokemon['name'] if pokemon else None,
            'pokemon_context': {
                'name': pokemon['name'],
                'type1': pokemon['types'][0].title() if pokemon else None,
                'type2': pokemon['types'][1].title() if pokemon and len(pokemon['types']) > 1 else None,
                'hp': pokemon['hp'] if pokemon else 0,
                'attack': pokemon['attack'] if pokemon else 0,
                'defense': pokemon['defense'] if pokemon else 0,
                'speed': pokemon['speed'] if pokemon else 0,
                'weak_to': [w.title() for w in pokemon['weak_to']] if pokemon else [],
                'strong_against': [s.title() for s in pokemon['strong_against']] if pokemon else []
            } if pokemon else None,
            'lore_info': None,
            'evolution_chain': None
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Vercel handler
def handler(environ, start_response):
    return app(environ, start_response)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
