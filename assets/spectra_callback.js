// spectra_callback.js
// This script listens for changes in the time slider plot and updates the acoustic spectra plot
// by re-computing the average of the raw spectra within the selected time range.

(function() {
    const timeSliderContainer = document.getElementById('time_slider_plot');
    const spectraPlotElem = document.querySelector('#acoustic_spectra_plot .js-plotly-plot');

    if (!timeSliderContainer || !spectraPlotElem) {
        console.error("Required elements not found");
        return;
    }

    const timeSlider = timeSliderContainer.querySelector('.js-plotly-plot');
    if (!timeSlider) {
        console.error("Time slider graph element not found");
        return;
    }

    let isInteracting = false;
    let lastRange = null;

    function getValidRange(eventData) {
        if (eventData['xaxis.range[0]'] && eventData['xaxis.range[1]']) {
            return [
                eventData['xaxis.range[0]'], 
                eventData['xaxis.range[1]']
            ];
        }
        
        const xaxis = timeSlider.layout.xaxis;
        if (xaxis && xaxis.range) {
            return [...xaxis.range];
        }
        
        return lastRange;
    }

    function handleRelayout(eventData) {
        const newRange = getValidRange(eventData);
        if (!newRange) return;
        
        if (lastRange && 
            newRange[0] === lastRange[0] && 
            newRange[1] === lastRange[1]) {
            return;
        }
        lastRange = newRange;

        if (!isInteracting) {
            console.log('Relayout event:', {
                eventData,
                isInteracting,
                newRange,
                lastRange
            });
            updateSpectraPlot(newRange, spectraPlotElem);
        }
    }

    timeSlider.addEventListener('mousedown', () => {
        isInteracting = true;
        const currentRange = getValidRange({});
        console.log("Mouse down on slider", {
            currentRange,
            timestamp: new Date().toISOString()
        });
    });

    document.addEventListener('mouseup', (event) => {
        if (isInteracting) {
            isInteracting = false;
            const newRange = getValidRange({});
            console.log("Mouse up (was dragging)", {
                newRange,
                timestamp: new Date().toISOString(),
                duration: newRange ? `${new Date(newRange[1]) - new Date(newRange[0])}ms` : null
            });
            
            if (newRange) {
                updateSpectraPlot(newRange, spectraPlotElem);
            }
        }
    });

    timeSlider.on('plotly_relayout', handleRelayout);

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
})();

