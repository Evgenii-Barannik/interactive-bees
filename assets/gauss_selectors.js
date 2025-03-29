// Global variable to store image paths for different sensors and datetimes
let gaussImagePaths = {};

// Function to initialize the Gaussian image paths
function initGaussImagePaths(paths) {
    gaussImagePaths = paths;
    updateDatetimeSelector();
}

// Function to update the datetime selector based on the selected sensor
function updateDatetimeSelector() {
    const sensorSelector = document.getElementById('gaussImageSelector');
    const datetimeSelector = document.getElementById('gaussDatetimeSelector');
    const selectedSensor = sensorSelector.options[sensorSelector.selectedIndex].text.replace('Sensor ', '');
    
    // Clear existing options except "Averaged"
    while (datetimeSelector.options.length > 1) {
        datetimeSelector.remove(1);
    }
    
    // Make sure we have individual paths for this sensor
    if (gaussImagePaths[selectedSensor] && gaussImagePaths[selectedSensor].gauss_individual) {
        const individualPaths = gaussImagePaths[selectedSensor].gauss_individual;
        
        console.log("Individual paths for sensor", selectedSensor, ":", individualPaths);
        
        // Add options for each individual datetime
        individualPaths.forEach((path, index) => {
            console.log(`Processing path ${index}: ${path}`);
            
            // Match the format used in the filenames: YYYY-MM-DD-HH-MM-SS+ZZZZ
            const datetimeMatch = path.match(/(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}\+\d{4})/);
            
            let datetimeLabel;
            
            if (datetimeMatch) {
                console.log(`Found datetime: ${datetimeMatch[1]}`);
                // Format the datetime to be more readable
                // YYYY-MM-DD-HH-MM-SS+ZZZZ -> YYYY-MM-DD HH:MM:SS
                const dt = datetimeMatch[1];
                const formattedDt = `${dt.substring(0, 10)} ${dt.substring(11, 13)}:${dt.substring(14, 16)}:${dt.substring(17, 19)}`;
                datetimeLabel = formattedDt;
            } else {
                console.log(`No datetime found, using fallback`);
                datetimeLabel = `Datapoint ${index + 1}`;
            }
            
            const option = document.createElement('option');
            option.value = path;
            option.textContent = datetimeLabel;
            datetimeSelector.appendChild(option);
        });
    }
    
    // Reset to "Averaged" option
    datetimeSelector.value = "averaged";
    updateGaussImage();
}

// Function to update the displayed Gaussian image
function updateGaussImage() {
    const sensorSelector = document.getElementById('gaussImageSelector');
    const datetimeSelector = document.getElementById('gaussDatetimeSelector');
    const displayedImage = document.getElementById('displayedGaussImage');
    
    const selectedSensor = sensorSelector.options[sensorSelector.selectedIndex].text.replace('Sensor ', '');
    const selectedDatetime = datetimeSelector.value;
    
    if (selectedDatetime === "averaged") {
        // Show the averaged image for the selected sensor
        displayedImage.src = gaussImagePaths[selectedSensor].gauss_averaged;
    } else {
        // Show the individual image for the selected datetime
        displayedImage.src = selectedDatetime;
    }
}

// Add event listeners once DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    const sensorSelector = document.getElementById('gaussImageSelector');
    const datetimeSelector = document.getElementById('gaussDatetimeSelector');
    
    // Add event listener for sensor selector
    sensorSelector.addEventListener('change', function() {
        updateDatetimeSelector();
    });
    
    // Add event listener for datetime selector
    datetimeSelector.addEventListener('change', function() {
        updateGaussImage();
    });
}); 