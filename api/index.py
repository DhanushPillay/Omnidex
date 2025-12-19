"""
Vercel Serverless Function for Omnidex Pokemon Chatbot
"""
import sys
import os

# Add paths for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'backend'))

from flask import Flask, request, jsonify, send_from_directory
import json

# Initialize Flask app
app = Flask(__name__, 
            static_folder='../frontend/static',
            template_folder='../frontend/templates')

# Lazy load chatbot to avoid cold start timeout
chatbot = None

def get_chatbot():
    global chatbot
    if chatbot is None:
        from pokemon_chatbot import PokemonChatbot
        csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'pokemon_data.csv')
        chatbot = PokemonChatbot(csv_path)
    return chatbot

@app.route('/')
def home():
    """Serve the main chat interface"""
    return send_from_directory('../frontend/templates', 'index.html')

@app.route('/static/<path:filename>')
def static_files(filename):
    """Serve static files"""
    return send_from_directory('../frontend/static', filename)

@app.route('/ask', methods=['POST'])
def ask():
    """Handle question requests from the frontend"""
    try:
        data = request.get_json()
        question = data.get('question', '')
        
        if not question:
            return jsonify({'error': 'No question provided'}), 400
        
        bot = get_chatbot()
        response = bot.answer_question(question)
        
        # Try to extract Pokemon name and get context
        image_url = None
        pokemon_name = bot._extract_pokemon_name(question.lower())
        pokemon_context = None
        lore_info = None
        
        # Story keywords for lore search
        story_keywords = ['story', 'lore', 'backstory', 'origin', 'history', 'found', 'created', 'discovered', 'legend']
        is_story_query = any(kw in question.lower() for kw in story_keywords)
        
        # If no Pokemon name found, try context
        if not pokemon_name:
            pokemon_name = bot.conversation_context.get('last_pokemon')
        
        if pokemon_name:
            image_url = bot._get_sprite_url(pokemon_name)
            bot.conversation_context['last_pokemon'] = pokemon_name
            
            # Get rich context from CSV
            matched = bot._fuzzy_find_pokemon(pokemon_name)
            if matched:
                try:
                    poke = bot.df[bot.df['Name'] == matched].iloc[0]
                    pokemon_context = {
                        'name': matched,
                        'type1': poke['Type1'],
                        'type2': poke['Type2'] if poke['Type2'] else None,
                        'hp': int(poke['HP']),
                        'attack': int(poke['Attack']),
                        'defense': int(poke['Defense']),
                        'speed': int(poke['Speed']),
                        'generation': int(poke['Generation']),
                        'legendary': bool(poke['Legendary'])
                    }
                    if poke['Type1'] in bot.TYPE_CHART:
                        pokemon_context['weak_to'] = bot.TYPE_CHART[poke['Type1']]['weak']
                        pokemon_context['strong_against'] = bot.TYPE_CHART[poke['Type1']]['strong']
                    
                    # Web search for lore queries
                    if is_story_query:
                        try:
                            from duckduckgo_search import DDGS
                            with DDGS() as ddgs:
                                results = list(ddgs.text(f"{matched} Pokemon lore origin backstory", max_results=3))
                            if results:
                                lore_info = [{'title': r.get('title', ''), 'body': r.get('body', '')} for r in results]
                        except Exception as e:
                            print(f"Web search error: {e}")
                except:
                    pass
        
        # Get evolution chain if available
        evolution_chain = bot.conversation_context.get('evolution_chain')
        if evolution_chain:
            bot.conversation_context['evolution_chain'] = None
        
        return jsonify({
            'success': True,
            'response': response,
            'image_url': image_url,
            'pokemon_name': pokemon_name,
            'pokemon_context': pokemon_context,
            'lore_info': lore_info,
            'evolution_chain': evolution_chain
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Vercel requires this handler
def handler(environ, start_response):
    return app(environ, start_response)

# For local testing
if __name__ == '__main__':
    app.run(debug=True, port=5000)
