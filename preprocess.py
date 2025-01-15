import joblib
import pickle
import numpy as np
from sklearn.metrics import accuracy_score

def process_data(data):
    features = [data['temperature'], data['humidity'],data['pm25'], data['pm10'], 
                data['no2'],data['so2'], data['co'],data['proximityToIndustrialAreas'],
                data['populationDensity']] #['Temperature', 'Humidity', 'PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'Proximity_to_Industrial_Areas', 'Population_Density']
    
    features = [list(map(int, features))]
    
    loaded_scaler = joblib.load('scaler.pkl')

    x_new_scaled = loaded_scaler.transform(features)

    return x_new_scaled

def process_features(sample):
    print(type(sample), sample.shape)
    temperatureHumidityInteration = sample[0][0]*sample[0][1]

    print(len(sample[0][2:7]))
    

    averagePollutants = sum(sample[0][2:7])/5

    preprocessedFeatures = np.array([sample[0][4], sample[0][6], sample[0][7], averagePollutants, temperatureHumidityInteration])

    return preprocessedFeatures

def check_consistency():
    with open('Air_Quality.pkl', 'rb') as file:
        loaded_model = pickle.load(file)

    print(loaded_model.feature_names)   

def make_predictions(inputSample, label):
    with open('Air_Quality.pkl', 'rb') as file:
        loaded_model = pickle.load(file)

    print(inputSample)

    #check_consistency()

    print(type(inputSample), len(inputSample))
    pred = loaded_model.predict(inputSample.reshape(1,-1))

    encoding = {'Good' : 1, 'Poor': 0}
    decoding = {1 : 'Good', 0: 'Poor'}

    expectedLabel = encoding[label]

    print(pred)

    score = accuracy_score(pred, [expectedLabel])

    print(f"Accuracy : {score}")

    #print(decoding[pred[0]])

    return decoding[pred[0]] #pred, score 