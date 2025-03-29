document.addEventListener('DOMContentLoaded', function() {
    const selector = document.getElementById('peakRelationsImageSelector');
    const image = document.getElementById('displayedPeakRelationsImage');
    
    if (selector && image) {
        selector.addEventListener('change', function() {
            const selectedSensor = selector.options[selector.selectedIndex].text.replace('Sensor ', '');
            if (imagePaths && imagePaths[selectedSensor] && imagePaths[selectedSensor]['peak_relations']) {
                image.src = imagePaths[selectedSensor]['peak_relations'];
            }
        });
    }
}); 
