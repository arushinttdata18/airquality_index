from flask import Flask, request, jsonify, render_template
import pickle, joblib
from Project.preprocess import process_data, process_features, make_predictions

app = Flask(__name__, template_folder='templates', static_folder='static')

# Load your trained model
with open('Air_Quality.pkl', 'rb') as model_file:
    model = joblib.load(model_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from the JSON request
    data = request.json
    region = data['region']
    temperature = data['temperature']
    humidity = data['humidity']
    pm25 = data['pm25']
    pm10 = data['pm10']
    no2 = data['no2']
    so2 = data['so2']
    co = data['co']
    proximity_to_industrial_areas = data['proximityToIndustrialAreas']
    population_density = data['populationDensity']
    air_quality = data['airQuality']

    # Prepare the feature array for prediction
    features = [[region, temperature, humidity, pm25, pm10, no2, so2, co,
                proximity_to_industrial_areas, population_density, air_quality]]
    #required_features = [[temperature, no2, co, proximity_to_industrial_areas, so2]]

    scaled_data = process_data(data)

    preprocessed_data = process_features(scaled_data)

    prediction = make_predictions(preprocessed_data, data['airQuality']) 

    # Perform the prediction
    #prediction = model.predict(features)

    # Return the prediction result as JSON
    return jsonify({'prediction': prediction})#int(prediction[0])

if __name__ == "__main__":
    app.run(debug=True)
