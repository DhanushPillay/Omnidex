// Load stats on page load
window.addEventListener('DOMContentLoaded', () => {
    loadStats();
});

// Load Pokemon statistics
async function loadStats() {
    try {
        const response = await fetch('/stats');
        const data = await response.json();

        document.getElementById('total-pokemon').textContent = data.total_pokemon;
        document.getElementById('legendary-count').textContent = data.legendary_count;
        document.getElementById('generations').textContent = data.generations;
        document.getElementById('unique-types').textContent = data.unique_types;
    } catch (error) {
        console.error('Error loading stats:', error);
    }
}

// Send question to the chatbot
async function sendQuestion() {
    const input = document.getElementById('question-input');
    const question = input.value.trim();

    if (!question) {
        return;
    }

    // Clear input
    input.value = '';

    // Display user message
    addMessage(question, 'user');

    // Show loading state
    setLoading(true);

    try {
        // First get data from our Python backend
        const response = await fetch('/ask', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ question: question })
        });

        const data = await response.json();

        if (data.success) {
            // Try to enhance response with Grok AI via Puter.js
            let finalResponse = data.response;

            try {
                if (typeof puter !== 'undefined' && puter.ai) {
                    const grokResponse = await puter.ai.chat(
                        `You are Omnidex, an all-knowing Pokemon AI expert. Make this response conversational and engaging (2-3 sentences, use 1-2 emoji):

User asked: "${question}"
Data: ${data.response}

Give a natural, friendly response based on this data.`,
                        { model: 'x-ai/grok-4.1-fast', max_tokens: 150 }
                    );
                    if (grokResponse && grokResponse.message && grokResponse.message.content) {
                        finalResponse = grokResponse.message.content;
                    }
                }
            } catch (grokError) {
                console.log('Grok enhancement skipped:', grokError.message);
                // Use original response if Grok fails
            }

            addMessage(finalResponse, 'bot', data.image_url);
        } else {
            addMessage('Error: ' + (data.error || 'Unknown error'), 'bot');
        }
    } catch (error) {
        addMessage('Error: Failed to connect to the server', 'bot');
        console.error('Error:', error);
    } finally {
        setLoading(false);
    }
}

// Add message to chat (with optional image)
function addMessage(text, sender, imageUrl = null) {
    const chatMessages = document.getElementById('chat-messages');

    // Remove welcome message if it exists
    const welcomeScreen = chatMessages.querySelector('.welcome-screen');
    if (welcomeScreen) {
        welcomeScreen.remove();
    }

    const messageDiv = document.createElement('div');
    messageDiv.className = `message message-${sender}`;

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';

    const label = document.createElement('div');
    label.className = 'message-label';
    label.textContent = sender === 'user' ? 'You' : 'Omnidex';

    // Add Pokemon image if available (for bot messages)
    if (sender === 'bot' && imageUrl) {
        const imageContainer = document.createElement('div');
        imageContainer.className = 'pokemon-image-container';

        const img = document.createElement('img');
        img.src = imageUrl;
        img.alt = 'Pokemon';
        img.className = 'pokemon-sprite';
        img.onerror = () => { imageContainer.style.display = 'none'; };

        imageContainer.appendChild(img);
        contentDiv.appendChild(imageContainer);
    }

    const messageText = document.createElement('div');
    messageText.className = 'message-text';
    // Handle newlines if needed, or stick to textContent for safety
    messageText.textContent = text;

    contentDiv.appendChild(label);
    contentDiv.appendChild(messageText);
    messageDiv.appendChild(contentDiv);

    chatMessages.appendChild(messageDiv);

    // Scroll to bottom
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Handle Enter key press
function handleKeyPress(event) {
    if (event.key === 'Enter') {
        sendQuestion();
    }
}

// Quick question buttons
function askQuestion(question) {
    const input = document.getElementById('question-input');
    input.value = question;
    sendQuestion();
}

// Clear chat
function clearChat() {
    const chatMessages = document.getElementById('chat-messages');
    chatMessages.innerHTML = `
        <div class="welcome-screen">
            <div class="welcome-icon">🎮</div>
            <h2>Welcome to PokéBot</h2>
            <p>I'm your advanced AI assistant for all things Pokémon. Ask me about stats, matchups, or lore!</p>
        </div>
    `;
}

// Set loading state
function setLoading(isLoading) {
    const sendBtn = document.getElementById('send-btn');
    const sendText = document.getElementById('send-text');
    const spinner = document.getElementById('loading-spinner');

    if (isLoading) {
        sendBtn.disabled = true;
        sendText.style.display = 'none';
        spinner.style.display = 'inline-block';
    } else {
        sendBtn.disabled = false;
        sendText.style.display = 'inline';
        spinner.style.display = 'none';
    }
}
