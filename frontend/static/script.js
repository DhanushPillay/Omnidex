// Conversation history for multi-turn context
let conversationHistory = [];

// Send message on button click
async function sendMessage() {
    const input = document.getElementById('user-input');
    const question = input.value.trim();

    if (!question) return;

    input.value = '';
    hideWelcome();
    addMessage(question, 'user');
    showTyping();

    try {
        const response = await fetch('/ask', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question })
        });

        const data = await response.json();
        hideTyping();

        if (data.success) {
            // Backend now handles all personality/formatting
            // Only need to display the response
            addMessage(data.response, 'bot', data.image_url, data.evolution_chain);
        } else {
            addMessage('Something went wrong. Please try again.', 'bot');
        }
    } catch (error) {
        hideTyping();
        addMessage('Connection error. Please check if the server is running.', 'bot');
    }
}

// Initialize on load
document.addEventListener('DOMContentLoaded', () => {
    // No profile loading or avatar grid initialization needed
});

// Add message to chat (updated for avatar)
function addMessage(text, sender, imageUrl = null, evolutionChain = null) {
    const area = document.getElementById('messages-area');

    const msg = document.createElement('div');
    msg.className = `message message-${sender}`;

    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    if (sender === 'user') {
        // Hardcoded Ash Avatar
        avatar.innerHTML = `<img src="https://play.pokemonshowdown.com/sprites/trainers/red.png" alt="Ash">`;
    } else {
        avatar.innerHTML = '<img src="/static/pokeball.png" alt="Pokeball">';
    }

    const content = document.createElement('div');
    content.className = 'message-content';

    // Show evolution chain if available (multiple Pokemon images)
    if (sender === 'bot' && evolutionChain && evolutionChain.length > 0) {
        const evoContainer = document.createElement('div');
        evoContainer.className = 'evolution-chain';
        evolutionChain.forEach((evo, index) => {
            const evoItem = document.createElement('div');
            evoItem.className = 'evolution-item';
            const img = document.createElement('img');
            img.src = evo.sprite;
            img.alt = evo.name;
            img.title = evo.name;
            evoItem.appendChild(img);
            const nameLabel = document.createElement('span');
            nameLabel.textContent = evo.name;
            evoItem.appendChild(nameLabel);
            evoContainer.appendChild(evoItem);
            // Add arrow between evolutions (but not after the last one)
            if (index < evolutionChain.length - 1 && evolutionChain[index + 1].level > evo.level) {
                const arrow = document.createElement('span');
                arrow.className = 'evolution-arrow';
                arrow.textContent = '→';
                evoContainer.appendChild(arrow);
            }
        });
        content.appendChild(evoContainer);
    }
    // Otherwise show single image if available
    else if (sender === 'bot' && imageUrl) {
        const imgContainer = document.createElement('div');
        imgContainer.className = 'pokemon-image';
        const img = document.createElement('img');
        img.src = imageUrl;
        img.alt = 'Pokemon';
        img.onerror = () => imgContainer.remove();
        imgContainer.appendChild(img);
        content.appendChild(imgContainer);
    }

    const textDiv = document.createElement('div');
    textDiv.className = 'message-text';
    textDiv.textContent = text;
    content.appendChild(textDiv);

    msg.appendChild(avatar);
    msg.appendChild(content);
    area.appendChild(msg);

    area.scrollTop = area.scrollHeight;
}

// Show typing indicator
function showTyping() {
    const area = document.getElementById('messages-area');
    const typing = document.createElement('div');
    typing.id = 'typing';
    typing.className = 'message message-bot';
    typing.innerHTML = `
        <div class="message-avatar"><img src="/static/pokeball.png" alt="Pokeball"></div>
        <div class="message-content">
            <div class="typing-indicator">
                <span></span><span></span><span></span>
            </div>
        </div>
    `;
    area.appendChild(typing);
    area.scrollTop = area.scrollHeight;
}

// Hide typing indicator
function hideTyping() {
    document.getElementById('typing')?.remove();
}

// Hide welcome message
function hideWelcome() {
    document.getElementById('welcome')?.remove();
}

// Handle suggestion click
function askSuggestion(text) {
    document.getElementById('user-input').value = text;
    sendMessage();
}

// Handle Enter key
function handleKeyPress(event) {
    if (event.key === 'Enter') {
        sendMessage();
    }
}

// Clear chat
function clearChat() {
    // Reset conversation history
    conversationHistory = [];
    const area = document.getElementById('messages-area');
    area.innerHTML = `
        <div class="welcome-message" id="welcome">
            <div class="welcome-icon"><img src="/static/pokeball.png" alt="Pokeball"></div>
            <h1>Omnidex</h1>
            <p>Your Pokemon AI Assistant</p>
            <div class="suggestions">
                <button class="suggestion" onclick="askSuggestion('Tell me about Pikachu')">Tell me about Pikachu</button>
                <button class="suggestion" onclick="askSuggestion('What is Charizard weak to?')">What is Charizard weak to?</button>
                <button class="suggestion" onclick="askSuggestion('Compare Mewtwo and Mew')">Compare Mewtwo and Mew</button>
                <button class="suggestion" onclick="askSuggestion('Pokemon similar to Gengar')">Pokemon similar to Gengar</button>
            </div>
        </div>
    `;
}
