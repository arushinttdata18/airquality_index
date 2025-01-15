async function predict() {
    const formData = {
        region: document.getElementById('region').value,
        temperature: document.getElementById('temperature').value,
        humidity: document.getElementById('humidity').value,
        pm25: document.getElementById('pm25').value,
        pm10: document.getElementById('pm10').value,
        no2: document.getElementById('no2').value,
        so2: document.getElementById('so2').value,
        co: document.getElementById('co').value,
        proximityToIndustrialAreas: document.getElementById('proximityToIndustrialAreas').value,
        populationDensity: document.getElementById('populationDensity').value,
        airQuality: document.getElementById('airQuality').value
    };

    const response = await fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(formData)
    });

    const result = await response.json();
    document.getElementById('result').innerText = `Predicted Air Quality: ${result.prediction}`;
}
