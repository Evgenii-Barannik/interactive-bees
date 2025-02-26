// spectra_callback.js
// This script listens for changes in the time slider plot and updates the acoustic spectra plot
// by re-computing the average of the raw spectra within the selected time range.

let currentTimeSlider = null;
let spectraPlotElem = null;
let isInteracting = false;
let lastRange = null;

function initSpectraHandlers() {
    // Remove old handlers if they exist
    if(currentTimeSlider) {
        currentTimeSlider.removeListener('plotly_relayout', handleRelayout);
        currentTimeSlider.removeEventListener('mousedown', handleInteractionStart);
        currentTimeSlider.removeEventListener('touchstart', handleInteractionStart);
    }

    // Find fresh elements after redraw
    const timeSliderContainer = document.getElementById('time_slider_plot');
    currentTimeSlider = timeSliderContainer?.querySelector('.js-plotly-plot');
    spectraPlotElem = document.querySelector('#acoustic_spectra_plot .js-plotly-plot');

    if(!currentTimeSlider || !spectraPlotElem) return;

    // Binding new handlers
    currentTimeSlider.on('plotly_relayout', handleRelayout);
    currentTimeSlider.addEventListener('mousedown', handleInteractionStart);
    currentTimeSlider.addEventListener('touchstart', handleInteractionStart);
    document.addEventListener('mouseup', handleInteractionEnd);
    document.addEventListener('touchend', handleInteractionEnd);
}

function handleInteractionStart() {
    isInteracting = true;
}

function handleInteractionEnd() {
    isInteracting = false;
    const newRange = getValidRange({});
    if(newRange) updateSpectraPlot(newRange, spectraPlotElem);
}

function getValidRange(eventData) {
    return eventData['xaxis.range'] || 
           currentTimeSlider?.layout?.xaxis?.range || 
           lastRange;
}

function handleRelayout(eventData) {
    const newRange = getValidRange(eventData);
    if(!newRange || (lastRange?.[0] === newRange[0] && lastRange?.[1] === newRange[1])) return;
    
    lastRange = newRange;
    const continuousUpdate = document.getElementById('continuous_update')?.checked;
    
    if(!isInteracting || continuousUpdate) {
        updateSpectraPlot(newRange, spectraPlotElem);
    }
}

function updateSpectraPlot(newRange, spectraPlot) {
    var data = spectraPlot.data;
    var updateY = [];
    var startTime = new Date(newRange[0]);
    var endTime = new Date(newRange[1]);
    
    let totalSpectra = 0; 

    // Process each sensor trace in the acoustic spectra plot.
    for (var i = 0; i < data.length; i++) {
        var meta = data[i].meta;
        if (!meta || !meta.raw_spectra || !meta.raw_times) {
            // If no meta data, keep original y values.
            updateY.push(data[i].y);
            continue;
        }
        var rawTimes = meta.raw_times.map(function(t) { return new Date(t); });
        var rawSpectra = meta.raw_spectra; // array of spectra arrays
        var selectedSpectra = [];
        // Filter raw spectra based on time range.
        for (var j = 0; j < rawTimes.length; j++) {
            if (rawTimes[j] >= startTime && rawTimes[j] <= endTime) {
                selectedSpectra.push(rawSpectra[j]);
            }
        }
        if (selectedSpectra.length > 0) {
            // Compute element-wise average of the selected spectra.
            var nBins = selectedSpectra[0].length;
            var sum = new Array(nBins).fill(0);
            for (var j = 0; j < selectedSpectra.length; j++) {
                for (var k = 0; k < nBins; k++) {
                    sum[k] += selectedSpectra[j][k];
                }
            }
            var avg = sum.map(function(val) { return val / selectedSpectra.length; });
            // Normalize the averaged spectrum to 100 * value / max_value
            var max_val = Math.max.apply(null, avg);
            if (max_val !== 0) {
                avg = avg.map(function(val) { return 100 * val / max_val; });
            }
            updateY.push(avg);
            totalSpectra += selectedSpectra.length; 
        } else {
            // If no spectra fall within the selected range for this sensor,
            // update with an empty array so that no curve (spline) is drawn.
            updateY.push([]);
        }
    }
    
    console.log(`Processing range ${new Date(newRange[0]).toISOString()} → ${new Date(newRange[1]).toISOString()}\n   Total spectra: ${totalSpectra}`);
    
    // Update all traces in the acoustic spectra plot.
    Plotly.restyle(spectraPlot, { y: updateY });
}

// Init on the first load
document.addEventListener('DOMContentLoaded', initSpectraHandlers);

