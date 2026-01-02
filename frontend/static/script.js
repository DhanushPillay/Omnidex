// Conversation history for multi-turn context
let conversationHistory = [];
let recognition = null;
let isRecording = false;
let synth = window.speechSynthesis;

// Initialize Speech Recognition
if ('webkitSpeechRecognition' in window) {
    recognition = new webkitSpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.lang = 'en-US';

    recognition.onstart = function () {
        isRecording = true;
        document.getElementById('mic-btn').classList.add('recording');
    };

    recognition.onend = function () {
        isRecording = false;
        document.getElementById('mic-btn').classList.remove('recording');
    };

    recognition.onresult = function (event) {
        const transcript = event.results[0][0].transcript;
        document.getElementById('user-input').value = transcript;
        sendMessage();
    };
} else {
    console.log("Speech Recognition not supported in this browser.");
    document.getElementById('mic-btn').style.display = 'none';
}

// Toggle Recording
function toggleRecording() {
    if (!recognition) return;

    if (isRecording) {
        recognition.stop();
    } else {
        recognition.start();
    }
}

// Speak text using TTS
function speakResponse(text) {
    if (synth.speaking) {
        synth.cancel();
    }

    // Clean text (remove emoji and markdown-like chars for better reading)
    const cleanText = text.replace(/[*#]/g, '').replace(/[\u{1F600}-\u{1F64F}]/gu, '');

    const utterance = new SpeechSynthesisUtterance(cleanText);
    utterance.rate = 1.0;
    utterance.pitch = 1.0;

    // Try to find a good English voice
    const voices = synth.getVoices();
    const preferredVoice = voices.find(v => v.name.includes('Google US English') || v.name.includes('Samantha'));
    if (preferredVoice) utterance.voice = preferredVoice;

    synth.speak(utterance);
}

// Handle Image Upload
async function handleImageUpload(input) {
    if (input.files && input.files[0]) {
        const file = input.files[0];

        hideWelcome();
        // Show user uploaded image
        const reader = new FileReader();
        reader.onload = function (e) {
            addMessage("What is this Pokemon?", 'user', e.target.result);
        };
        reader.readAsDataURL(file);

        showTyping();

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();
            hideTyping();

            if (data.error) {
                addMessage(`Error: ${data.error}`, 'bot');
            } else if (data.name) {
                let reply = `That looks like **${data.name}**! `;
                reply += `\n${data.description}`;
                if (data.db_match) {
                    reply += `\n\nI have it in my database! Do you want to know its stats?`;
                }
                addMessage(reply, 'bot');
                // Automatic speaking removed based on user feedback
                // speakResponse(`That looks like ${data.name}!`);
            } else {
                addMessage("I couldn't identify that Pokemon. Try a clearer image!", 'bot');
            }
        } catch (error) {
            hideTyping();
            addMessage('Error uploading image.', 'bot');
        }

        // Reset input
        input.value = '';
    }
}

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
            addMessage(data.response, 'bot', data.image_url, data.evolution_chain, data.comparison_images);
            // Automatic speaking removed based on user feedback
            // speakResponse(data.response);
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
function addMessage(text, sender, imageUrl = null, evolutionChain = null, comparisonImages = null) {
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

    // Show comparison images (VS View)
    if (sender === 'bot' && comparisonImages && comparisonImages.length >= 2) {
        const compContainer = document.createElement('div');
        compContainer.className = 'comparison-container';

        // First Pokemon
        const p1 = comparisonImages[0];
        const card1 = document.createElement('div');
        card1.className = 'pokemon-vs-card';
        card1.innerHTML = `<img src="${p1.image}" alt="${p1.name}" onload="document.getElementById('messages-area').scrollTop = document.getElementById('messages-area').scrollHeight"><span>${p1.name}</span>`;

        // VS Badge
        const vsBadge = document.createElement('div');
        vsBadge.className = 'vs-badge';
        vsBadge.textContent = 'VS';

        // Second Pokemon
        const p2 = comparisonImages[1];
        const card2 = document.createElement('div');
        card2.className = 'pokemon-vs-card';
        card2.innerHTML = `<img src="${p2.image}" alt="${p2.name}" onload="document.getElementById('messages-area').scrollTop = document.getElementById('messages-area').scrollHeight"><span>${p2.name}</span>`;

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
    // Show single image if available
    else if (sender === 'bot' && imageUrl) {
        const imgContainer = document.createElement('div');
        imgContainer.className = 'pokemon-image';
        const img = document.createElement('img');
        img.src = imageUrl;
        img.alt = 'Pokemon';
        img.onload = () => {
            const area = document.getElementById('messages-area');
            area.scrollTop = area.scrollHeight;
        };
        img.onerror = () => imgContainer.remove();
        imgContainer.appendChild(img);
        content.appendChild(imgContainer);
    }

    const textDiv = document.createElement('div');
    textDiv.className = 'message-text';
    textDiv.textContent = text;
    content.appendChild(textDiv);

    // Add Speak button for bot messages
    if (sender === 'bot') {
        const actionsDiv = document.createElement('div');
        actionsDiv.className = 'message-actions';
        actionsDiv.style.display = 'flex';
        actionsDiv.style.justifyContent = 'flex-end';
        actionsDiv.style.marginTop = '0.25rem';

        const speakBtn = document.createElement('button');
        speakBtn.className = 'icon-btn';
        speakBtn.title = 'Read aloud';
        speakBtn.style.padding = '0.25rem'; // Smaller padding
        speakBtn.innerHTML = `
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon>
                <path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07"></path>
            </svg>
        `;
        speakBtn.onclick = () => speakResponse(text);

        actionsDiv.appendChild(speakBtn);
        content.appendChild(actionsDiv);
    }

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
