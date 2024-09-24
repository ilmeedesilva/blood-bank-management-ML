from flask import Flask, jsonify
import joblib
import pandas as pd
import datetime
from forecasting import fetch_data, preprocess_data, train_model, forecast_stock

from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "http://localhost:3000"}})


@app.route('/api/forecast', methods=['GET'])
def get_forecast():
    donations, stock = fetch_data()

    if donations.empty or len(donations) < 5:
        return jsonify({"error": "Not enough donation data available for forecasting."}), 400

    data = preprocess_data(donations, stock)

    if data.empty or len(data) < 5:
        return jsonify({"error": "Insufficient data after preprocessing for forecasting."}), 400

    model, training_columns = train_model(data)

    future_dates = pd.date_range(start=datetime.datetime.now(), periods=6, freq='ME')
    forecast = forecast_stock(model, future_dates, training_columns)

    response = {}
    for index, row in forecast.iterrows():
        blood_type = row['blood_type']
        month_year = row['donation_date'].strftime("%B %Y")

        if blood_type not in response:
            response[blood_type] = []

        response[blood_type].append({
            "predicted_date": month_year,
            "predicted_quantity": f"{row['predicted_quantity']} ml"
        })

    return jsonify(response)


@app.route('/api/forecast/<int:organization_id>', methods=['GET'])
def get_forecast_for_organization(organization_id):
    donations, stock = fetch_data(organization_id)

    if donations.empty or len(donations) < 5:
        return jsonify({"error": f"Not enough donation data available for organization ID {organization_id}."}), 400

    data = preprocess_data(donations, stock)

    if data.empty or len(data) < 5:
        return jsonify({"error": f"Insufficient data after preprocessing for organization ID {organization_id}."}), 400

    model, training_columns = train_model(data)

    future_dates = pd.date_range(start=datetime.datetime.now(), periods=6, freq='ME')
    forecast = forecast_stock(model, future_dates, training_columns)

    response = {}
    for index, row in forecast.iterrows():
        blood_type = row['blood_type']
        month_year = row['donation_date'].strftime("%B %Y")

        if blood_type not in response:
            response[blood_type] = []

        response[blood_type].append({
            "predicted_date": month_year,
            "predicted_quantity": f"{row['predicted_quantity']} ml"
        })

    return jsonify(response)


if __name__ == "__main__":
    app.run(debug=True)
