// spectra_callback.js
// This script listens for changes in the time slider plot and updates the acoustic spectra plot
// by re-computing the average of the raw spectra within the selected time range.

function updateSpectraPlot(newRange, spectraPlot) {
    var data = spectraPlot.data;
    var updateY = [];
    var startTime = new Date(newRange[0]);
    var endTime = new Date(newRange[1]);

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
        } else {
            // If no spectra fall within the selected range for this sensor,
            // update with an empty array so that no curve (spline) is drawn.
            updateY.push([]);
        }
    }
    // Update all traces in the acoustic spectra plot.
    Plotly.restyle(spectraPlot, { y: updateY });
}

function debounce(func, timeout = 300){
  let timer;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => { func.apply(this, args); }, timeout);
  };
}

// Attach the update callback to the time slider plot's relayout event.
var timeSliderContainer = document.getElementById('time_slider_plot');
// Select the inner Plotly graph element (which provides Plotly's event methods).
var timeSlider = timeSliderContainer.querySelector('.js-plotly-plot');
if (timeSlider) {
    const debouncedUpdate = debounce(function(eventData) {
        console.log("Debounced relayout eventData:", eventData);
        
        var newRange;
        if (eventData.hasOwnProperty('xaxis.range')) {
             newRange = eventData['xaxis.range'];
        } else if (eventData.hasOwnProperty('xaxis.range[0]') && eventData.hasOwnProperty('xaxis.range[1]')) {
             newRange = [eventData['xaxis.range[0]'], eventData['xaxis.range[1]']];
        } else {
             return;
        }
        
        console.log("Final range:", newRange);
        var spectraContainer = document.getElementById('acoustic_spectra_plot');
        var spectraPlot = spectraContainer.querySelector('.js-plotly-plot');
        if (spectraPlot) {
            updateSpectraPlot(newRange, spectraPlot);
        }
    }, 300);

    timeSlider.on('plotly_relayout', debouncedUpdate);
    timeSlider.on('plotly_sliderend', function(eventData) {
        debouncedUpdate(eventData);
    });
} else {
    console.error("Time slider graph element not found.");
} 
