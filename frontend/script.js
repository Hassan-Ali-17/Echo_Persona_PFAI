document.addEventListener('DOMContentLoaded', () => {
    const uploadBtn = document.getElementById('upload-btn');
    const fileUpload = document.getElementById('file-upload');
    const recordBtn = document.getElementById('record-btn');
    const micIcon = document.getElementById('mic-icon');
    
    const interactionArea = document.querySelector('.interaction-area');
    const loadingState = document.getElementById('loading-state');
    const resultsArea = document.getElementById('results-area');
    const resetBtn = document.getElementById('reset-btn');
    
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

    // --- Recording Flow (Mocked for demo purposes, as real MediaRecorder needs HTTPS/localhost) ---
    let isRecording = false;
    let mediaRecorder;
    let audioChunks = [];

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

                mediaRecorder.start();
                isRecording = true;
                recordBtn.textContent = "Stop Recording";
                recordBtn.style.background = "rgba(239, 68, 68, 0.2)";
                recordBtn.style.color = "#ef4444";
                micIcon.classList.add('recording');
                
            } catch (err) {
                alert("Microphone access denied or unavailable.");
            }
        } else {
            // Stop recording
            mediaRecorder.stop();
            mediaRecorder.stream.getTracks().forEach(track => track.stop());
            isRecording = false;
            recordBtn.textContent = "Start Recording";
            recordBtn.style.background = "";
            recordBtn.style.color = "";
            micIcon.classList.remove('recording');
        }
    });

    // --- Processing & API Call ---
    async function processAudio(file) {
        showLoading();

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.detail || 'Prediction failed');
            }

            const data = await response.json();
            showResults(data);

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
        interactionArea.classList.remove('hidden');
        loadingState.classList.add('hidden');
        resultsArea.classList.add('hidden');
        fileUpload.value = '';
    }

    resetBtn.addEventListener('click', resetUI);

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
