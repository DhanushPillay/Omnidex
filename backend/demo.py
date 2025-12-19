"""
Demo script showing the Pokemon Chatbot capabilities
"""
from pokemon_chatbot import PokemonChatbot

def run_demo():
    """Run a demo of the chatbot with example questions"""
    print("🔥 Pokemon Chatbot Demo\n")
    
    # Initialize chatbot
    chatbot = PokemonChatbot('pokemon_data.csv')
    
    # Example questions to demonstrate functionality
    demo_questions = [
        "List all fire type Pokemon",
        "Tell me about Charizard",
        "How many legendary Pokemon are there?",
        "Who has the highest attack?",
        "Show me generation 1 Pokemon",
        "Compare Charizard and Blastoise",
        "Name all water type Pokemon",
        "List all legendary Pokemon"
    ]
    
    print("\n" + "="*70)
    print("RUNNING DEMO QUESTIONS")
    print("="*70 + "\n")
    
    for question in demo_questions:
        print(f"❓ Question: {question}")
        print("-" * 70)
        response = chatbot.answer_question(question)
        print(f"🤖 Answer: {response}")
        print("\n" + "="*70 + "\n")
    
    print("Demo completed! Now you can run pokemon_chatbot.py to interact yourself!")

if __name__ == "__main__":
    run_demo()
