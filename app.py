import sys
import os

# Add backend folder to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from flask import Flask, render_template, request, jsonify
from pokemon_chatbot import PokemonChatbot

# Configure Flask with new folder structure
app = Flask(__name__, 
            template_folder='frontend/templates',
            static_folder='frontend/static')

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
        
        # Try to extract Pokemon name and get sprite URL
        image_url = None
        pokemon_name = chatbot._extract_pokemon_name(question.lower())
        if pokemon_name:
            image_url = chatbot._get_sprite_url(pokemon_name)
            # Update conversation context
            chatbot.conversation_context['last_pokemon'] = pokemon_name
        
        return jsonify({
            'success': True,
            'response': response,
            'image_url': image_url,
            'pokemon_name': pokemon_name
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
