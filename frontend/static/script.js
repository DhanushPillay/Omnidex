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
            let finalResponse = data.response;

            // Enhance with Grok AI using rich context + enhanced personality
            try {
                if (typeof puter !== 'undefined' && puter.ai) {
                    // Build rich context string if we have Pokemon data
                    let contextStr = data.response;
                    if (data.pokemon_context) {
                        const ctx = data.pokemon_context;
                        contextStr = `
Pokemon: ${ctx.name}
Type: ${ctx.type1}${ctx.type2 ? '/' + ctx.type2 : ''}
Stats: HP ${ctx.hp}, Attack ${ctx.attack}, Defense ${ctx.defense}, Speed ${ctx.speed}
Generation: ${ctx.generation}
Legendary: ${ctx.legendary ? 'Yes' : 'No'}
${ctx.weak_to ? 'Weak to: ' + ctx.weak_to.join(', ') : ''}
${ctx.strong_against ? 'Strong against: ' + ctx.strong_against.join(', ') : ''}`;
                    }

                    // Add lore/story info from web search
                    let loreStr = '';
                    if (data.lore_info && data.lore_info.length > 0) {
                        loreStr = '\n\nLore from Internet:\n' + data.lore_info.map(l =>
                            `- ${l.title}: ${l.body}`
                        ).join('\n');
                    }

                    // Add conversation history for context
                    let historyStr = '';
                    if (conversationHistory.length > 0) {
                        const recentHistory = conversationHistory.slice(-4);
                        historyStr = '\n\nRecent Conversation:\n' + recentHistory.map(h =>
                            `${h.role}: ${h.content}`
                        ).join('\n');
                    }

                    // Track this exchange
                    conversationHistory.push({ role: 'User', content: question });

                    const grokResponse = await puter.ai.chat(
                        `You are Omnidex, an enthusiastic Pokemon Professor AI assistant. 

PERSONALITY:
- Speak like a friendly Pokemon trainer and researcher
- Show genuine enthusiasm about Pokemon, especially rare and legendary ones
- Share fun facts and trivia when relevant
- Reference Pokemon regions (Kanto, Johto, Unova, etc.) naturally
- Use 1-2 Pokemon-themed emojis 
- Be warm, helpful, and engaging
- Remember what was discussed earlier if history is provided
${historyStr}

User asked: "${question}"

Pokemon Database:
${contextStr}
${loreStr}

RESPONSE GUIDELINES:
- 2-4 sentences, natural and conversational
- If comparing Pokemon, give battle insights
- If discussing evolution, mention how exciting the transformation is
- If discussing lore/story, be a storyteller
- If discussing stats, be analytical but fun
- Always sound like you genuinely love Pokemon!`,
                        { model: 'x-ai/grok-4.1-fast', max_tokens: 300 }
                    );
                    if (grokResponse?.message?.content) {
                        finalResponse = grokResponse.message.content;
                        // Track bot response
                        conversationHistory.push({ role: 'Omnidex', content: finalResponse.substring(0, 100) });
                        // Keep only last 10 exchanges
                        if (conversationHistory.length > 10) {
                            conversationHistory = conversationHistory.slice(-10);
                        }
                    }
                }
            } catch (e) {
                console.log('Grok skipped:', e.message);
            }

            addMessage(finalResponse, 'bot', data.image_url, data.evolution_chain);
        } else {
            addMessage('Something went wrong. Please try again.', 'bot');
        }
    } catch (error) {
        hideTyping();
        addMessage('Connection error. Please check if the server is running.', 'bot');
    }
}

// User Profile State
let currentUser = {
    name: 'Trainer',
    avatar: '' // Empty means default or U
};

// Trainer Sprites (Pokemon Showdown)
const TRAINERS = [
    'red', 'blue', 'ethan', 'lyra', 'brendan',
    'may', 'lucas', 'dawn', 'hilbert', 'hilda'
];

// Initialize on load
document.addEventListener('DOMContentLoaded', () => {
    loadProfile();
    initAvatarGrid();
});

function loadProfile() {
    const stored = localStorage.getItem('omnidex_profile');
    if (stored) {
        currentUser = JSON.parse(stored);
    }
}

function initAvatarGrid() {
    const grid = document.getElementById('avatar-grid');
    grid.innerHTML = '';

    TRAINERS.forEach(trainer => {
        const url = `https://play.pokemonshowdown.com/sprites/trainers/${trainer}.png`;
        const div = document.createElement('div');
        div.className = 'avatar-option';
        div.onclick = () => selectAvatar(url, div);
        div.innerHTML = `<img src="${url}" alt="${trainer}">`;
        grid.appendChild(div);
    });
}

function openProfileModal() {
    const modal = document.getElementById('profile-modal');
    modal.classList.add('active');

    // Set current values
    document.getElementById('username-input').value = currentUser.name;
    document.getElementById('preview-avatar').src = currentUser.avatar || 'https://ui-avatars.com/api/?name=T&background=random';

    // Highlight selected if in grid
    document.querySelectorAll('.avatar-option').forEach(opt => {
        opt.classList.remove('selected');
        if (opt.querySelector('img').src === currentUser.avatar) {
            opt.classList.add('selected');
        }
    });
}

function closeProfileModal() {
    document.getElementById('profile-modal').classList.remove('active');
}

let tempSelectedAvatar = '';

function selectAvatar(url, element) {
    tempSelectedAvatar = url;
    document.getElementById('preview-avatar').src = url;

    // Update selection UI
    document.querySelectorAll('.avatar-option').forEach(el => el.classList.remove('selected'));
    if (element) element.classList.add('selected');
}

function handleFileUpload(input) {
    if (input.files && input.files[0]) {
        const reader = new FileReader();
        reader.onload = function (e) {
            tempSelectedAvatar = e.target.result; // Base64 string
            document.getElementById('preview-avatar').src = tempSelectedAvatar;
            // Clear grid selection
            document.querySelectorAll('.avatar-option').forEach(el => el.classList.remove('selected'));
        };
        reader.readAsDataURL(input.files[0]);
    }
}

function saveProfile() {
    const name = document.getElementById('username-input').value.trim() || 'Trainer';
    // Use new selection, or keep old if not changed
    const avatar = tempSelectedAvatar || currentUser.avatar;

    currentUser = { name, avatar };
    localStorage.setItem('omnidex_profile', JSON.stringify(currentUser));

    closeProfileModal();
    // Maybe show a toast notification?
}

// Add message to chat (updated for avatar)
function addMessage(text, sender, imageUrl = null, evolutionChain = null) {
    const area = document.getElementById('messages-area');

    const msg = document.createElement('div');
    msg.className = `message message-${sender}`;

    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    if (sender === 'user') {
        if (currentUser.avatar) {
            avatar.innerHTML = `<img src="${currentUser.avatar}" alt="${currentUser.name}">`;
        } else {
            avatar.textContent = currentUser.name.charAt(0).toUpperCase();
            avatar.style.background = 'var(--accent)';
        }
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
