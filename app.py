from flask import Flask, render_template, request
import pandas as pd
import pickle
import json
import numpy as np

app = Flask(__name__)

# Load the trained model
model = pickle.load(open("real_estate_home_prices_prediction.pkl", 'rb'))

# Load the columns from the JSON file
with open('columns.json', 'r') as f:
    columns_data = json.load(f)

columns = columns_data.get('data_columns', [])

# Define the predict_price function
def predict_price(location, sqft, bath, bhk, model, columns):
    # Initialize an array of zeros for all features
    x = np.zeros(len(columns))

    # Set known feature values
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    
    # Find the index of the location feature and set it
    if location in columns:
        loc_index = columns.index(location)
        x[loc_index] = 1

    # Predict the price using the model
    return model.predict([x])[0]


@app.route('/')
def home():
    return render_template('real.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        cities = request.form.get('cities')
        squareFeet = request.form.get('squareFeet')
        bathrooms = request.form.get('bathrooms')
        bedrooms = request.form.get('bedrooms')
    else:  # For handling GET requests
        cities = request.args.get('cities')
        squareFeet = request.args.get('squareFeet')
        bathrooms = request.args.get('bathrooms')
        bedrooms = request.args.get('bedrooms')

    print(cities, squareFeet, bathrooms, bedrooms)

    # Prepare input features for prediction
    input_features = [cities, float(squareFeet), int(bathrooms), int(bedrooms)]

    # Call the predict_price function to get the predicted price
    predicted_price = predict_price(*input_features, model=model, columns=columns)

    print("Predicted Price: ", predicted_price)

    # Render the result page with the predicted price
    return render_template('after.html', predicted_price=predicted_price)


if __name__ == '__main__':
    app.run(debug=True)
