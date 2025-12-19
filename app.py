import sys
import os

# Add backend folder to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from flask import Flask, render_template, request, jsonify
from pokemon_chatbot import PokemonChatbot

# Configure Flask with new folder structure
app = Flask(__name__, 
            template_folder='.',
            static_folder='static')

# Initialize the chatbot with new data path
chatbot = PokemonChatbot('data/pokemon_data.csv')

@app.route('/')
def home():
    """Render the main chat interface"""
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    """Handle question requests from the frontend"""
    try:
        data = request.get_json()
        question = data.get('question', '')
        
        if not question:
            return jsonify({'error': 'No question provided'}), 400
        
        # Get response from chatbot
        response = chatbot.answer_question(question)
        
        # Try to extract Pokemon name and get rich context
        image_url = None
        pokemon_name = chatbot._extract_pokemon_name(question.lower())
        pokemon_context = None
        lore_info = None
        
        # Detect if this is a story/lore query
        story_keywords = ['story', 'lore', 'backstory', 'origin', 'history', 'found', 'created', 'discovered', 'legend']
        is_story_query = any(kw in question.lower() for kw in story_keywords)
        
        # If no Pokemon name found, try to use context from response or last_pokemon
        if not pokemon_name:
            # Check if there's a Pokemon name in the conversation context
            pokemon_name = chatbot.conversation_context.get('last_pokemon')
        
        if pokemon_name:
            image_url = chatbot._get_sprite_url(pokemon_name)
            chatbot.conversation_context['last_pokemon'] = pokemon_name
            
            # Get rich context from CSV for Grok
            matched = chatbot._fuzzy_find_pokemon(pokemon_name)
            if matched:
                try:
                    poke = chatbot.df[chatbot.df['Name'] == matched].iloc[0]
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
                    if poke['Type1'] in chatbot.TYPE_CHART:
                        pokemon_context['weak_to'] = chatbot.TYPE_CHART[poke['Type1']]['weak']
                        pokemon_context['strong_against'] = chatbot.TYPE_CHART[poke['Type1']]['strong']
                    
                    # If story query, search for lore
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
        evolution_chain = chatbot.conversation_context.get('evolution_chain')
        # Clear it after reading so it doesn't persist to next query
        if evolution_chain:
            chatbot.conversation_context['evolution_chain'] = None
        
        return jsonify({
            'success': True,
            'response': response,
            'image_url': image_url,
            'pokemon_name': pokemon_name,
            'pokemon_context': pokemon_context,
            'lore_info': lore_info,
            'evolution_chain': evolution_chain  # Evolution sprites
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/stats')
def stats():
    """Get Pokemon statistics"""
    try:
        total_pokemon = len(chatbot.df)
        legendary_count = len(chatbot.df[chatbot.df['Legendary'] == True])
        generations = chatbot.df['Generation'].nunique()
        types = set(chatbot.df['Type1'].tolist() + chatbot.df['Type2'].dropna().tolist())
        
        return jsonify({
            'total_pokemon': total_pokemon,
            'legendary_count': legendary_count,
            'generations': generations,
            'unique_types': len(types)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🔮 Omnidex - Pokemon AI Server Starting...")
    print("="*60)
    print(f"\nLoaded {len(chatbot.df)} Pokemon!")
    print("\n🌐 Open your browser and go to: http://localhost:5000")
    print("\nPress CTRL+C to stop the server\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
