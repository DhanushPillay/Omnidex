import sys
import os
import secrets
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add backend folder to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from flask import Flask, render_template, request, jsonify, session
from pokemon_chatbot import PokemonChatbot

# Configure Flask with correct folder structure for deployment
app = Flask(__name__, 
            template_folder='frontend/templates',
            static_folder='frontend/static')

# Set secret key for sessions
app.secret_key = secrets.token_hex(16)

# Initialize the chatbot with new data path
chatbot = PokemonChatbot('data/pokemon_data.csv')

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    """Render the main chat interface"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle image uploads for analysis"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Analyze image
        analysis = chatbot.analyze_image(filepath)

        # Clean up
        try:
            os.remove(filepath)
        except:
            pass

        return jsonify(analysis)

@app.route('/ask', methods=['POST'])
def ask():
    """Handle question requests from the frontend"""
    try:
        data = request.get_json()
        question = data.get('question', '')
        
        if not question:
            return jsonify({'error': 'No question provided'}), 400
        
        # Initialize context in session if not present
        if 'context' not in session:
            session['context'] = {
                'last_pokemon': None,
                'last_intent': None,
                'conversation_history': [],      
                'mentioned_pokemon': [],          
                'compared_pokemon': [],           
                'current_topic': None,            
                'evolution_chain': None           
            }

        # Initialize user profile in session
        if 'user_profile' not in session:
            session['user_profile'] = {
                'name': None,
                'fav_pokemon': None
            }
        
        # Get context from session
        context = session['context']
        user_profile = session['user_profile']

        # Get response from chatbot, passing context and user profile
        response = chatbot.answer_question(question, context, user_profile)
        
        # Try to extract Pokemon name and get rich context
        image_url = None
        comparison_images = []
        pokemon_name = chatbot._extract_pokemon_name(question.lower())
        pokemon_context = None
        lore_info = None
        
        # Detect if this is a story/lore query
        story_keywords = ['story', 'lore', 'backstory', 'origin', 'history', 'found', 'created', 'discovered', 'legend']
        is_story_query = any(kw in question.lower() for kw in story_keywords)
        
        # Check if it was a comparison
        if context.get('last_intent') == 'compare':
            compared_list = context.get('compared_pokemon', [])
            if len(compared_list) >= 2:
                img1 = chatbot._get_sprite_url(compared_list[0])
                img2 = chatbot._get_sprite_url(compared_list[1])
                comparison_images = [
                    {'name': compared_list[0], 'image': img1},
                    {'name': compared_list[1], 'image': img2}
                ]
                # Default image to first one
                image_url = img1
                pokemon_name = compared_list[0]
        
        # If no Pokemon name found (and not comparing), try to use context
        if not pokemon_name and not comparison_images:
            # Check if there's a Pokemon name in the conversation context
            # BUT only use it if the user implies it (pronouns or follow-up intents)
            last_poke = context.get('last_pokemon')
            if last_poke:
                # Intents that naturally follow a subject
                follow_up_intents = ['weakness', 'strength', 'evolution', 'stats', 'moves']
                intent = context.get('last_intent')
                
                # Pronouns in question
                has_pronoun = any(w in question.lower().split() for w in ['it', 'its', 'he', 'she', 'this', 'that'])
                
                if (intent in follow_up_intents) or has_pronoun or (intent and intent.startswith('context_')):
                    pokemon_name = last_poke
        
        if pokemon_name:
            if not image_url:
                image_url = chatbot._get_sprite_url(pokemon_name)
            # Update last_pokemon in context (if not already set by answer_question)
            context['last_pokemon'] = pokemon_name
            
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
        evolution_chain = context.get('evolution_chain')
        # Clear it after reading
        if evolution_chain:
            context['evolution_chain'] = None
        
        # Save context back to session
        session['context'] = context
        session['user_profile'] = user_profile
        session.modified = True

        return jsonify({
            'success': True,
            'response': response,
            'image_url': image_url,
            'comparison_images': comparison_images,
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
