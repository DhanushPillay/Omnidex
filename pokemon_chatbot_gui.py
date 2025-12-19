import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from pokemon_chatbot import PokemonChatbot
import threading

class PokemonChatbotGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🔥 Pokemon Chatbot")
        self.root.geometry("900x700")
        self.root.configure(bg="#2b2d42")
        
        # Initialize chatbot
        try:
            self.chatbot = PokemonChatbot('pokemon_data.csv')
            self.chatbot_ready = True
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load Pokemon data: {str(e)}")
            self.chatbot_ready = False
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the user interface"""
        
        # Title Frame
        title_frame = tk.Frame(self.root, bg="#8d99ae", pady=15)
        title_frame.pack(fill=tk.X)
        
        title_label = tk.Label(
            title_frame,
            text="🔥 Pokemon Chatbot 🔥",
            font=("Arial", 24, "bold"),
            bg="#8d99ae",
            fg="#2b2d42"
        )
        title_label.pack()
        
        subtitle_label = tk.Label(
            title_frame,
            text="Ask me anything about Pokemon!",
            font=("Arial", 12),
            bg="#8d99ae",
            fg="#2b2d42"
        )
        subtitle_label.pack()
        
        # Main container
        main_frame = tk.Frame(self.root, bg="#2b2d42")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Chat display area
        chat_frame = tk.Frame(main_frame, bg="#2b2d42")
        chat_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Chat history with scrollbar
        self.chat_display = scrolledtext.ScrolledText(
            chat_frame,
            wrap=tk.WORD,
            font=("Consolas", 11),
            bg="#edf2f4",
            fg="#2b2d42",
            relief=tk.FLAT,
            padx=10,
            pady=10,
            state=tk.DISABLED
        )
        self.chat_display.pack(fill=tk.BOTH, expand=True)
        
        # Configure text tags for styling
        self.chat_display.tag_config("user", foreground="#ef233c", font=("Arial", 11, "bold"))
        self.chat_display.tag_config("bot", foreground="#06a77d", font=("Arial", 11, "bold"))
        self.chat_display.tag_config("message", foreground="#2b2d42", font=("Consolas", 10))
        self.chat_display.tag_config("info", foreground="#8d99ae", font=("Arial", 9, "italic"))
        
        # Input frame
        input_frame = tk.Frame(main_frame, bg="#2b2d42")
        input_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Question input
        self.question_entry = tk.Entry(
            input_frame,
            font=("Arial", 12),
            bg="#edf2f4",
            fg="#2b2d42",
            relief=tk.FLAT,
            insertbackground="#2b2d42"
        )
        self.question_entry.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10), ipady=8)
        self.question_entry.bind('<Return>', lambda e: self.send_message())
        
        # Send button
        send_button = tk.Button(
            input_frame,
            text="Send",
            font=("Arial", 12, "bold"),
            bg="#06a77d",
            fg="white",
            relief=tk.FLAT,
            cursor="hand2",
            command=self.send_message,
            padx=20
        )
        send_button.pack(side=tk.RIGHT, ipady=8)
        
        # Quick questions frame
        quick_frame = tk.Frame(main_frame, bg="#2b2d42")
        quick_frame.pack(fill=tk.X)
        
        quick_label = tk.Label(
            quick_frame,
            text="Quick Questions:",
            font=("Arial", 10, "bold"),
            bg="#2b2d42",
            fg="#8d99ae"
        )
        quick_label.pack(anchor=tk.W, pady=(0, 5))
        
        # Quick question buttons
        buttons_frame = tk.Frame(quick_frame, bg="#2b2d42")
        buttons_frame.pack(fill=tk.X)
        
        quick_questions = [
            "List all fire type Pokemon",
            "Show legendary Pokemon",
            "Who has highest attack?",
            "Tell me about Pikachu"
        ]
        
        for i, question in enumerate(quick_questions):
            btn = tk.Button(
                buttons_frame,
                text=question,
                font=("Arial", 9),
                bg="#8d99ae",
                fg="#2b2d42",
                relief=tk.FLAT,
                cursor="hand2",
                command=lambda q=question: self.quick_question(q)
            )
            btn.pack(side=tk.LEFT, padx=5, pady=5, ipady=3, ipadx=5)
        
        # Clear button
        clear_button = tk.Button(
            buttons_frame,
            text="Clear Chat",
            font=("Arial", 9),
            bg="#ef233c",
            fg="white",
            relief=tk.FLAT,
            cursor="hand2",
            command=self.clear_chat
        )
        clear_button.pack(side=tk.RIGHT, padx=5, pady=5, ipady=3, ipadx=5)
        
        # Display welcome message
        self.display_welcome_message()
        
    def display_welcome_message(self):
        """Display welcome message in chat"""
        welcome_msg = """Welcome to Pokemon Chatbot! 🎮

I can answer questions about Pokemon! Try asking:
• "List all fire type Pokemon"
• "Tell me about Charizard"
• "How many legendary Pokemon are there?"
• "Who has the highest attack?"
• "Compare Charizard and Blastoise"
• "Show me generation 1 Pokemon"

Type your question below or click a quick question button!
"""
        self.chat_display.configure(state=tk.NORMAL)
        self.chat_display.insert(tk.END, welcome_msg, "info")
        self.chat_display.insert(tk.END, "\n" + "="*80 + "\n\n")
        self.chat_display.configure(state=tk.DISABLED)
        self.chat_display.see(tk.END)
        
    def send_message(self):
        """Handle sending a message"""
        if not self.chatbot_ready:
            messagebox.showwarning("Warning", "Chatbot is not ready!")
            return
            
        question = self.question_entry.get().strip()
        if not question:
            return
        
        # Clear input
        self.question_entry.delete(0, tk.END)
        
        # Display user question
        self.display_message("You", question, "user")
        
        # Get response in separate thread to avoid GUI freezing
        thread = threading.Thread(target=self.get_response, args=(question,))
        thread.daemon = True
        thread.start()
        
    def get_response(self, question):
        """Get response from chatbot"""
        try:
            response = self.chatbot.answer_question(question)
            self.root.after(0, self.display_message, "Bot", response, "bot")
        except Exception as e:
            self.root.after(0, self.display_message, "Bot", f"Error: {str(e)}", "bot")
    
    def display_message(self, sender, message, tag):
        """Display a message in the chat area"""
        self.chat_display.configure(state=tk.NORMAL)
        
        # Add sender label
        self.chat_display.insert(tk.END, f"{sender}: ", tag)
        
        # Add message
        self.chat_display.insert(tk.END, f"{message}\n\n", "message")
        
        self.chat_display.configure(state=tk.DISABLED)
        self.chat_display.see(tk.END)
    
    def quick_question(self, question):
        """Handle quick question button click"""
        self.question_entry.delete(0, tk.END)
        self.question_entry.insert(0, question)
        self.send_message()
    
    def clear_chat(self):
        """Clear the chat display"""
        self.chat_display.configure(state=tk.NORMAL)
        self.chat_display.delete(1.0, tk.END)
        self.chat_display.configure(state=tk.DISABLED)
        self.display_welcome_message()


def main():
    """Main function to run the GUI"""
    root = tk.Tk()
    app = PokemonChatbotGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
