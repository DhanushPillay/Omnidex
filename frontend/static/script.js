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
            addMessage(data.response, 'bot', data.image_url, data.evolution_chain, data.comparison_images, data.card_data);
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
function addMessage(text, sender, imageUrl = null, evolutionChain = null, comparisonImages = null, cardData = null) {
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

    // ============ HOLOGRAPHIC CARD RENDERER ============
    if (sender === 'bot' && cardData) {
        const cardContainer = document.createElement('div');
        cardContainer.className = 'holo-card-container';

        // Determine type colors
        const type = cardData.type || 'default';
        const typeColor1 = `var(--type-${type})`;
        const typeColor2 = `var(--type-${type}-2)`;

        const card = document.createElement('div');
        card.className = 'holo-card';
        card.style.setProperty('--c1', typeColor1);
        card.style.setProperty('--c2', typeColor2);

        card.innerHTML = `
            <div class="holo-card-content">
                <div class="card-header">
                    <span>${cardData.name}</span>
                    <span class="card-hp">${cardData.hp} HP</span>
                </div>
                <div class="card-image">
                    <img src="${cardData.image}" alt="${cardData.name}">
                </div>
                <div class="card-stats">
                    <div class="stat-row">
                        <span>Type</span>
                        <span>${type.toUpperCase()}</span>
                    </div>
                    <div class="stat-row">
                        <span>Attack</span>
                        <span>${cardData.attack}</span>
                    </div>
                    <div class="stat-row">
                        <span>Defense</span>
                        <span>${cardData.defense}</span>
                    </div>
                </div>
            </div>
        `;

        // 3D Tilt Logic
        card.addEventListener('mousemove', function (e) {
            const rect = card.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;

            const centerX = rect.width / 2;
            const centerY = rect.height / 2;

            const rotateX = ((y - centerY) / centerY) * -10; // Max rotation deg
            const rotateY = ((x - centerX) / centerX) * 10;

            card.style.transform = `perspective(1000px) rotateX(${rotateX}deg) rotateY(${rotateY}deg) scale3d(1.05, 1.05, 1.05)`;

            // Move gradient
            const percentX = (x / rect.width) * 100;
            const percentY = (y / rect.height) * 100;

            card.style.setProperty('--bg-x', `${percentX}%`);
            card.style.setProperty('--bg-y', `${percentY}%`);
        });

        card.addEventListener('mouseleave', function () {
            card.style.transform = 'perspective(1000px) rotateX(0) rotateY(0) scale3d(1, 1, 1)';
        });

        cardContainer.appendChild(card);
        content.appendChild(cardContainer);
    }
    // Show comparison images (VS View)
    else if (sender === 'bot' && comparisonImages && comparisonImages.length >= 2) {
        const compContainer = document.createElement('div');
        compContainer.className = 'comparison-container';

        // First Pokemon
        const p1 = comparisonImages[0];
        const card1 = document.createElement('div');
        card1.className = 'pokemon-vs-card';
        card1.innerHTML = `<img src="${p1.image}" alt="${p1.name}"><span>${p1.name}</span>`;

        // VS Badge
        const vsBadge = document.createElement('div');
        vsBadge.className = 'vs-badge';
        vsBadge.textContent = 'VS';

        // Second Pokemon
        const p2 = comparisonImages[1];
        const card2 = document.createElement('div');
        card2.className = 'pokemon-vs-card';
        card2.innerHTML = `<img src="${p2.image}" alt="${p2.name}"><span>${p2.name}</span>`;

        compContainer.appendChild(card1);
        compContainer.appendChild(vsBadge);
        compContainer.appendChild(card2);

        content.appendChild(compContainer);
    }
    // Show evolution chain if available (multiple Pokemon images)
    else if (sender === 'bot' && evolutionChain && evolutionChain.length > 0) {
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
