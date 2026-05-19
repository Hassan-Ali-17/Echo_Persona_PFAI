document.addEventListener('DOMContentLoaded', () => {
    const uploadBtn = document.getElementById('upload-btn');
    const fileUpload = document.getElementById('file-upload');
    const recordBtn = document.getElementById('record-btn');
    const micIcon = document.getElementById('mic-icon');
    const visualizer = document.getElementById('visualizer');
    
    const interactionArea = document.querySelector('.interaction-area');
    const loadingState = document.getElementById('loading-state');
    const resultsArea = document.getElementById('results-area');
    const resetBtn = document.getElementById('reset-btn');
    const mainPanel = document.getElementById('main-panel');
    
    // Custom Audio Player Selectors
    const resultAudio = document.getElementById('result-audio');
    const playPauseBtn = document.getElementById('play-pause-btn');
    const playerProgressBar = document.getElementById('player-progress-bar');
    const playerProgressFill = document.getElementById('player-progress-fill');
    const playerCurrentTime = document.getElementById('player-current-time');
    const playerTotalDuration = document.getElementById('player-total-duration');

    // Insights Selectors
    const insightsContainer = document.getElementById('insights-container');
    const insightsText = document.getElementById('insights-text');

    // Model Selection Buttons
    const modelBtns = document.querySelectorAll('.toggle-btn');
    let selectedModel = 'dl';
    
    modelBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            modelBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            selectedModel = btn.dataset.model;
        });
    });

    // History elements
    const historyPanel = document.getElementById('history-panel');
    const historyList = document.getElementById('history-list');
    const clearHistoryBtn = document.getElementById('clear-history-btn');

    // Load history on start
    renderHistory();
    
    // Emotion Colors
    const emotionColors = {
        'Happy': '#f59e0b',
        'Sad': '#3b82f6',
        'Angry': '#ef4444',
        'Fearful': '#8b5cf6',
        'Disgusted': '#10b981',
        'Surprised': '#ec4899',
        'Neutral': '#94a3b8',
        'Calm': '#0ea5e9',
        'Uncertain': '#64748b'
    };

    // Personality Insight Texts
    const personalityInsights = {
        'Happy': "Your communication style radiates positivity, energy, and warmth. You tend to speak with highly expressive pitch variations, making your interactions collaborative, engaging, and motivating to those around you.",
        'Calm': "Your vocal style is poised, grounded, and peaceful. You maintain a stable, measured pitch that projects confidence, level-headedness, and emotional stability under pressure.",
        'Sad': "Your style is reflective, empathetic, and gentle. Speaking with softer vocal resonance and a gentle pace suggests you are an active listener who processes thoughts deeply before responding.",
        'Angry': "Your vocal signature is commanding, assertive, and direct. You project strong presence and authority, communicating with intense, focused energy that commands immediate attention.",
        'Fearful': "Your style is highly alert, cautious, and sensitive. Your voice registers responsiveness to surroundings, indicating detailed awareness, high care, and a protective orientation.",
        'Disgusted': "Your tone is discerning, analytical, and highly critical. You communicate with selective emphasis, showing that you prioritize accuracy, quality, and direct evaluation.",
        'Surprised': "Your style is highly dynamic, curious, and reactive. The spontaneous variations in your voice indicate a highly adaptive, creative, and enthusiastic approach to new ideas.",
        'Neutral': "Your communication is objective, balanced, and diplomatic. By keeping emotional inflections neutral and matter-of-fact, you excel at objective mediation and rational reasoning.",
        'Uncertain': "Your voice shows vulnerability, reflection, and open exploration. You speak with a flexible tone that seeks collaboration and feedback."
    };

    // --- Custom Audio Player Setup ---
    playPauseBtn.addEventListener('click', () => {
        if (resultAudio.paused) {
            resultAudio.play();
            playPauseBtn.textContent = '❚❚';
        } else {
            resultAudio.pause();
            playPauseBtn.textContent = '▶';
        }
    });

    resultAudio.addEventListener('timeupdate', () => {
        if (resultAudio.duration) {
            const pct = (resultAudio.currentTime / resultAudio.duration) * 100;
            playerProgressFill.style.width = `${pct}%`;
            playerCurrentTime.textContent = formatTime(resultAudio.currentTime);
        }
    });

    resultAudio.addEventListener('loadedmetadata', () => {
        playerTotalDuration.textContent = formatTime(resultAudio.duration);
    });

    resultAudio.addEventListener('ended', () => {
        playPauseBtn.textContent = '▶';
        playerProgressFill.style.width = '0%';
        playerCurrentTime.textContent = '0:00';
    });

    playerProgressBar.addEventListener('click', (e) => {
        const rect = playerProgressBar.getBoundingClientRect();
        const clickPosition = (e.clientX - rect.left) / rect.width;
        if (resultAudio.duration) {
            resultAudio.currentTime = clickPosition * resultAudio.duration;
        }
    });

    function formatTime(seconds) {
        if (isNaN(seconds) || seconds === Infinity) return '0:00';
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs < 10 ? '0' : ''}${secs}`;
    }

    // --- Upload Flow ---
    uploadBtn.addEventListener('click', () => {
        fileUpload.click();
    });

    fileUpload.addEventListener('change', async (e) => {
        if (e.target.files.length > 0) {
            const file = e.target.files[0];
            await processAudio(file);
        }
    });

    // --- Recording Flow with Live Audio Visualizer ---
    let isRecording = false;
    let mediaRecorder;
    let audioChunks = [];
    let audioCtx;
    let analyser;
    let sourceNode;
    let animationFrameId;

    recordBtn.addEventListener('click', async () => {
        if (!isRecording) {
            // Start recording
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                mediaRecorder = new MediaRecorder(stream);
                
                mediaRecorder.ondataavailable = (event) => {
                    audioChunks.push(event.data);
                };

                mediaRecorder.onstop = async () => {
                    const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                    audioChunks = [];
                    
                    try {
                        const arrayBuffer = await audioBlob.arrayBuffer();
                        const audioContext = new (window.AudioContext || window.webkitAudioContext)();
                        const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
                        
                        const wavBlob = audioBufferToWav(audioBuffer);
                        const file = new File([wavBlob], "recording.wav", { type: 'audio/wav' });
                        await processAudio(file);
                    } catch (e) {
                        alert("Error converting audio format. Please try again.");
                        console.error(e);
                    }
                };

                // Setup Web Audio API for visualizer
                audioCtx = new (window.AudioContext || window.webkitAudioContext)();
                analyser = audioCtx.createAnalyser();
                sourceNode = audioCtx.createMediaStreamSource(stream);
                sourceNode.connect(analyser);
                analyser.fftSize = 64; // Small fft for simple visual representation

                // Show visualizer and hide mic icon
                micIcon.classList.add('hidden');
                visualizer.classList.remove('hidden');
                
                // Draw loop
                drawVisualizer();

                mediaRecorder.start();
                isRecording = true;
                recordBtn.textContent = "Stop Recording";
                recordBtn.style.background = "rgba(239, 68, 68, 0.2)";
                recordBtn.style.color = "#ef4444";
                
            } catch (err) {
                alert("Microphone access denied or unavailable.");
                console.error(err);
            }
        } else {
            // Stop recording
            stopRecordingAndVisualizer();
        }
    });

    function stopRecordingAndVisualizer() {
        if (mediaRecorder && mediaRecorder.state !== 'inactive') {
            mediaRecorder.stop();
            mediaRecorder.stream.getTracks().forEach(track => track.stop());
        }
        
        // Stop animation
        if (animationFrameId) {
            cancelAnimationFrame(animationFrameId);
        }
        
        // Close audio context
        if (audioCtx && audioCtx.state !== 'closed') {
            audioCtx.close();
        }

        // Hide visualizer and show mic icon
        visualizer.classList.add('hidden');
        micIcon.classList.remove('hidden');

        isRecording = false;
        recordBtn.textContent = "Start Recording";
        recordBtn.style.background = "";
        recordBtn.style.color = "";
    }

    function drawVisualizer() {
        if (!analyser) return;

        const bufferLength = analyser.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);
        const canvasCtx = visualizer.getContext('2d');
        
        const width = visualizer.width;
        const height = visualizer.height;

        function draw() {
            animationFrameId = requestAnimationFrame(draw);
            analyser.getByteFrequencyData(dataArray);

            canvasCtx.fillStyle = 'rgba(11, 15, 25, 0.3)'; // translucent bg for motion blur
            canvasCtx.fillRect(0, 0, width, height);

            const barWidth = (width / bufferLength) * 1.5;
            let barHeight;
            let x = 0;

            for (let i = 0; i < bufferLength; i++) {
                barHeight = (dataArray[i] / 255) * height * 0.9;
                
                // Center bars vertically
                const y = (height - barHeight) / 2;

                // Gradient color
                const grad = canvasCtx.createLinearGradient(0, y, 0, y + barHeight);
                grad.addColorStop(0, '#ec4899'); // pink accent
                grad.addColorStop(1, '#6366f1'); // indigo accent

                canvasCtx.fillStyle = grad;
                
                // Rounded corner bars
                canvasCtx.beginPath();
                canvasCtx.roundRect(x, y, barWidth - 2, barHeight, 4);
                canvasCtx.fill();

                x += barWidth;
            }
        }
        draw();
    }

    // --- Processing & API Call ---
    async function processAudio(file) {
        showLoading();

        // Load the audio file into custom player
        const audioUrl = URL.createObjectURL(file);
        resultAudio.src = audioUrl;
        resultAudio.load();

        // Reset player UI controls
        playPauseBtn.textContent = '▶';
        playerProgressFill.style.width = '0%';
        playerCurrentTime.textContent = '0:00';
        playerTotalDuration.textContent = '0:00';

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch(`/api/predict?model_type=${selectedModel}`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.detail || 'Prediction failed');
            }

            const data = await response.json();
            showResults(data);
            saveToHistory(data);

        } catch (error) {
            alert(`Error: ${error.message}`);
            resetUI();
        }
    }

    // --- UI State Management ---
    function showLoading() {
        interactionArea.classList.add('hidden');
        resultsArea.classList.add('hidden');
        loadingState.classList.remove('hidden');
    }

    function showResults(data) {
        loadingState.classList.add('hidden');
        resultsArea.classList.remove('hidden');

        // Main Result
        const resultLabel = document.getElementById('result-emotion');
        resultLabel.textContent = data.emotion;
        resultLabel.style.color = emotionColors[data.emotion] || emotionColors['Neutral'];
        
        document.getElementById('result-confidence').textContent = `${(data.confidence * 100).toFixed(1)}%`;

        // Update model badge used
        document.getElementById('model-badge-used').textContent = data.model_used || "Deep Learning";

        // Set dynamic glow color
        mainPanel.className = 'glass-panel main-panel'; // reset classes
        const emotionClass = `glow-${data.emotion.toLowerCase()}`;
        mainPanel.classList.add(emotionClass);

        // Update Echo Persona Insights text and styling
        const activeColor = emotionColors[data.emotion] || emotionColors['Neutral'];
        insightsContainer.style.borderLeftColor = activeColor;
        const insightMsg = personalityInsights[data.emotion] || personalityInsights['Neutral'];
        insightsText.textContent = insightMsg;

        // Render Bars
        const barsContainer = document.getElementById('emotion-bars');
        barsContainer.innerHTML = '';

        // Sort scores descending
        const sortedScores = Object.entries(data.all_scores)
            .sort((a, b) => b[1] - a[1])
            .slice(0, 4); // Show top 4

        sortedScores.forEach(([emotion, score]) => {
            const percentage = (score * 100).toFixed(1);
            const color = emotionColors[emotion] || emotionColors['Neutral'];
            
            const row = document.createElement('div');
            row.className = 'emotion-row';
            
            row.innerHTML = `
                <div class="emotion-name">${emotion}</div>
                <div class="bar-container">
                    <div class="bar-fill" style="width: 0%; background: ${color}"></div>
                </div>
                <div class="emotion-score">${percentage}%</div>
            `;
            
            barsContainer.appendChild(row);
            
            // Animate bar
            setTimeout(() => {
                row.querySelector('.bar-fill').style.width = `${percentage}%`;
            }, 50);
        });
    }

    function resetUI() {
        // Pause audio if playing
        resultAudio.pause();
        resultAudio.src = '';
        
        interactionArea.classList.remove('hidden');
        loadingState.classList.add('hidden');
        resultsArea.classList.add('hidden');
        mainPanel.className = 'glass-panel main-panel'; // reset glow classes
        fileUpload.value = '';
        stopRecordingAndVisualizer();
    }

    resetBtn.addEventListener('click', resetUI);

    // --- History Management ---
    function saveToHistory(data) {
        let history = JSON.parse(localStorage.getItem('echo_persona_history') || '[]');
        
        // Add new item to beginning
        const newItem = {
            emotion: data.emotion,
            confidence: data.confidence,
            model: data.model_used || 'Deep Learning',
            timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
        };
        
        history.unshift(newItem);
        
        // Limit to 5 items
        history = history.slice(0, 5);
        
        localStorage.setItem('echo_persona_history', JSON.stringify(history));
        renderHistory();
    }

    function renderHistory() {
        const history = JSON.parse(localStorage.getItem('echo_persona_history') || '[]');
        if (history.length === 0) {
            historyPanel.classList.add('hidden');
            return;
        }

        historyPanel.classList.remove('hidden');
        historyList.innerHTML = '';

        history.forEach(item => {
            const color = emotionColors[item.emotion] || emotionColors['Neutral'];
            const itemEl = document.createElement('div');
            itemEl.className = 'history-item';
            
            itemEl.innerHTML = `
                <div class="history-item-left">
                    <span class="history-item-time">${item.timestamp}</span>
                    <span class="history-model-type">${item.model}</span>
                </div>
                <div class="history-item-right">
                    <span class="history-emotion-badge" style="background: ${color}20; color: ${color}; border: 1px solid ${color}40">${item.emotion}</span>
                    <span class="history-score">${(item.confidence * 100).toFixed(0)}%</span>
                </div>
            `;
            historyList.appendChild(itemEl);
        });
    }

    clearHistoryBtn.addEventListener('click', () => {
        localStorage.removeItem('echo_persona_history');
        renderHistory();
    });

    // --- Audio Utility ---
    function audioBufferToWav(buffer) {
        const numOfChan = buffer.numberOfChannels;
        const length = buffer.length * numOfChan * 2 + 44;
        const bufferArray = new ArrayBuffer(length);
        const view = new DataView(bufferArray);
        const channels = [], sampleRate = buffer.sampleRate;
        let offset = 0, pos = 0;

        function setUint16(data) {
            view.setUint16(offset, data, true);
            offset += 2;
        }
        function setUint32(data) {
            view.setUint32(offset, data, true);
            offset += 4;
        }

        setUint32(0x46464952); // "RIFF"
        setUint32(length - 8); // file length - 8
        setUint32(0x45564157); // "WAVE"

        setUint32(0x20746d66); // "fmt " chunk
        setUint32(16); // length = 16
        setUint16(1); // PCM (uncompressed)
        setUint16(numOfChan);
        setUint32(sampleRate);
        setUint32(sampleRate * 2 * numOfChan); // avg. bytes/sec
        setUint16(numOfChan * 2); // block-align
        setUint16(16); // 16-bit (hardcoded in this export)

        setUint32(0x61746164); // "data" - chunk
        setUint32(length - 44); // chunk length

        for(let i = 0; i < buffer.numberOfChannels; i++)
            channels.push(buffer.getChannelData(i));

        while(pos < buffer.length) {
            for(let i = 0; i < numOfChan; i++) {
                let sample = Math.max(-1, Math.min(1, channels[i][pos]));
                sample = (0.5 + sample < 0 ? sample * 32768 : sample * 32767)|0;
                view.setInt16(offset, sample, true);
                offset += 2;
            }
            pos++;
        }

        return new Blob([bufferArray], {type: "audio/wav"});
    }
});
