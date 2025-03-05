let audioCtx;
let source;
let clickTimeout;

// function is taken from http://apiologia.zymologia.fi/static/beeapp/js/basic_data.js 
// with added lines
// source = null;
// document.getElementById('audioDebugInfo').textContent = "Sound stopped";
function stopLoop() {
    if (source) {
        source.stop(); 
        source = null;
	document.getElementById('audioDebugInfo').textContent = "Sound stopped";
	console.log('Sound stopped');
    }
}


// function is taken from http://apiologia.zymologia.fi/static/beeapp/js/basic_data.js 
// with added line
// if (audioCtx) { audioCtx.close(); }
function generate_wave(frequencies, amplitudes, duration = 2.0, sampleRate = 44100){
    if (audioCtx) { audioCtx.close(); }
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();

    const totalSamples = sampleRate * duration;
    const audioBuffer = audioCtx.createBuffer(1, totalSamples, sampleRate);
    const outputData = audioBuffer.getChannelData(0); 

    const timeStep = 1 / sampleRate;

    frequencies.forEach((frequency, index) => {
        const phaseShift = Math.random() * 2 * Math.PI; // Random start phase
        for (let i = 0; i < totalSamples; i++) {
            const t = i * timeStep; 
            outputData[i] += amplitudes[index] * Math.sin(2 * Math.PI * frequency * t + phaseShift);
        }
    });

    // Smooth transition between end and start of the signal 
    const fadeDuration = 0.01; //  fade-in/fade-out duration in sec
    const fadeSamples = Math.floor(fadeDuration * sampleRate);

    // Fade-in for the first fadeSamples
    for (let i = 0; i < fadeSamples; i++) {
        const fadeFactor = i / fadeSamples;
        outputData[i] *= fadeFactor;
    }

    // Fade-out for the last fadeSamples
    for (let i = totalSamples - fadeSamples; i < totalSamples; i++) {
        const fadeFactor = (totalSamples - i) / fadeSamples;
        outputData[i] *= fadeFactor;
    }

    // Normalization
    const maxAmplitude = Math.max(...outputData.map(Math.abs));
    if (maxAmplitude > 0) {
        for (let i = 0; i < totalSamples; i++) {
            outputData[i] /= maxAmplitude; 
        }
    }

    source = audioCtx.createBufferSource();
    source.buffer = audioBuffer;
    source.loop = true;  
    source.connect(audioCtx.destination);
    source.start();

    console.log('Sound generated and looped smoothly');

    /*
    // Convert the audio buffer to WAV format
    const wavData = audioBufferToWav(audioBuffer);
    const blob = new Blob([wavData], { type: 'audio/wav' });

    // Create a download link
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.style.display = 'none';
    a.href = url;
    a.download = 'composed_sound.wav';
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url); // Clean up the URL
    */
}

function formatDate(date) {
    if (!date) return "error";
        return date.toISOString().replace('T', ' ').substr(0, 19);
}

// Function to play sound from current acoustic spectrum
function playSound() {
    const sensor = document.getElementById('audioSensorSelector').value;
    console.log(`Preparing sound for sensor ${sensor}`);

    const spectraPlot = document.querySelector('#acoustic_spectra_plot .js-plotly-plot');
    const timeSlider = document.querySelector('#time_slider_plot .js-plotly-plot');
    if (!spectraPlot) {
        console.error("Acoustic spectra plot not found");
        return;
    }
    
    let sensorData = null;
    for (let i = 0; i < spectraPlot.data.length; i++) {
        const trace = spectraPlot.data[i];
        if (trace.name && trace.name === `${sensor}`) {
            sensorData = trace;
            break;
        }
    }
    if (!sensorData) {
        console.error(`Error: data for sensor ${sensor} was not found`);
        return;
    }
    
    const frequencies = sensorData.x;
    const amplitudes = sensorData.y;
    
    if (!frequencies || !amplitudes || frequencies.length === 0 || amplitudes.length === 0) {
        console.error("No acoustic data available for selected time range");
        document.getElementById('audioDebugInfo').textContent = "No acoustic data available for selected time range";
        return;
    }

    // Assert that all amplitudes are sane
    const hasInvalidValues = amplitudes.some(amp => isNaN(amp) || !isFinite(amp));
    if (hasInvalidValues) {
        const error = "Error: spectrum contains NaN or Infinity values";
        console.error(error);
        return;
    }
    const normalizedAmplitudes = amplitudes.map(amp => amp / 100);

    // Assert that values are in [0, 1] range
    const isInRange = normalizedAmplitudes.every(amp => amp >= 0 && amp <= 1);
    if (!isInRange) {
        const error = "Error: normalized amplitudes outside [0, 1] range";
        console.error(error);
        return;
    }
    
    let timeRange = null;
    let numDatapoints = 0;
    
    if (timeSlider && timeSlider.layout && timeSlider.layout.xaxis) {
	timeRange = timeSlider.layout.xaxis.range;
    }
    
    if (sensorData.meta && sensorData.meta.raw_times && timeRange) {
	const startTime = new Date(timeRange[0]);
	const endTime = new Date(timeRange[1]);
	numDatapoints = sensorData.meta.raw_times.filter(t => {
	    const time = new Date(t);
	    return time >= startTime && time <= endTime;
	}).length;
    }

    // Stop any currently playing sound and generate new one
    stopLoop();
    generate_wave(frequencies, normalizedAmplitudes);
    
    // Format debug information
    let debugInfo = `Playing sound for sensor ${sensor}`;
    
    if (timeRange) {
        debugInfo += `\nSelected time range:`;
        debugInfo += `\nFrom ${formatDate(new Date(timeRange[0]))} (UTC)`;
        debugInfo += `\nTo\u00A0\u00A0\u00A0\u00A0 ${formatDate(new Date(timeRange[1]))} (UTC)`;
	debugInfo += `\nNumber of datapoints inside range: ${numDatapoints}`;
    }
    
    document.getElementById('audioDebugInfo').textContent = debugInfo;
}

document.addEventListener('DOMContentLoaded', function() {
    const playSpectrumButton = document.getElementById('playSpectrumButton');
    const stopButton = document.getElementById('stopButton');
    const audioSensorSelector = document.getElementById('audioSensorSelector');
    
    if (playSpectrumButton) {
        playSpectrumButton.addEventListener('click', playSound);
    }
    
    if (stopButton) {
        stopButton.addEventListener('click', stopLoop);
    }
});

