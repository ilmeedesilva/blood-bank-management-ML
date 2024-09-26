from flask import Flask, jsonify
import joblib
import pandas as pd
import datetime
from forecasting import fetch_data, preprocess_data, train_model, forecast_stock
from condition_forecasting import fetch_condition_data, preprocess_condition_data, train_condition_model, forecast_conditions
from medical_forecasting import fetch_medical_data, preprocess_medical_data, train_medical_model, forecast_medical


from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "http://localhost:3000"}})


@app.route('/api/forecast', methods=['GET'])
def get_forecast():
    donations, stock = fetch_data()

    if donations.empty or len(donations) < 5:
        return jsonify({"error": "Not enough donation data available for forecasting."}), 200

    data = preprocess_data(donations, stock)

    if data.empty or len(data) < 5:
        return jsonify({"error": "Insufficient data after preprocessing for forecasting."}), 200

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
        return jsonify({"error": f"Not enough donation data available for organization ID {organization_id}."}), 200

    data = preprocess_data(donations, stock)

    if data.empty or len(data) < 5:
        return jsonify({"error": f"Insufficient data after preprocessing for organization ID {organization_id}."}), 200

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


@app.route('/api/condition_forecast', methods=['GET'])
def get_condition_forecast():
    conditions = fetch_condition_data()

    if conditions.empty or len(conditions) < 5:
        return jsonify({"error": "Not enough condition data available for forecasting."}), 200

    data = preprocess_condition_data(conditions)

    if data.empty or len(data) < 5:
        return jsonify({"error": "Insufficient data after preprocessing for forecasting."}), 200

    model, training_columns = train_condition_model(data)

    future_dates = pd.date_range(start=datetime.datetime.now(), periods=6, freq='ME')
    forecast = forecast_conditions(model, future_dates, training_columns)

    response = {}
    for index, row in forecast.iterrows():
        condition_name = row['condition_name']
        month_year = row['date'].strftime("%B %Y")

        if condition_name not in response:
            response[condition_name] = []

        response[condition_name].append({
            "predicted_date": month_year,
            "predicted_blood_donation": f"{row['predicted_blood_donation']} ml"
        })

    return jsonify(response)


@app.route('/api/medical_forecast', methods=['GET'])
def get_medical_forecast():

    conditions = fetch_medical_data()

    if conditions.empty or len(conditions) < 5:
        return jsonify({"error": "Not enough condition data available for forecasting."}), 200


    data = preprocess_medical_data(conditions)

    if data.empty or len(data) < 5:
        return jsonify({"error": "Insufficient data after preprocessing for forecasting."}), 200

    model, training_columns = train_medical_model(data)

    future_dates = pd.date_range(start=datetime.datetime.now(), periods=6, freq='ME') 
    forecast = forecast_medical(model, future_dates, training_columns)

   
    response = {}
    for index, row in forecast.iterrows():
        condition_name = row['condition_name']
        month_year = row['date'].strftime("%B %Y")

        if condition_name not in response:
            response[condition_name] = []

        response[condition_name].append({
            "predicted_date": month_year,
            "predicted_occurrences": f"{row['predicted_occurrences']} cases"
        })

    return jsonify(response)

@app.route('/api/condition_forecast/<int:organization_id>', methods=['GET'])
def get_condition_forecast_for_organization(organization_id):
    conditions = fetch_condition_data(organization_id)

    if conditions.empty or len(conditions) < 5:
        return jsonify({"error": f"Not enough condition data available for organization ID {organization_id}."}), 200

    data = preprocess_condition_data(conditions)

    if data.empty or len(data) < 5:
        return jsonify({"error": f"Insufficient data after preprocessing for organization ID {organization_id}."}), 200

    model, training_columns = train_condition_model(data)

    future_dates = pd.date_range(start=datetime.datetime.now(), periods=6, freq='ME')
    forecast = forecast_conditions(model, future_dates, training_columns)

    response = {}
    for index, row in forecast.iterrows():
        condition_name = row['condition_name']
        month_year = row['date'].strftime("%B %Y")

        if condition_name not in response:
            response[condition_name] = []

        response[condition_name].append({
            "predicted_date": month_year,
            "predicted_blood_donation": f"{row['predicted_blood_donation']} ml"
        })

    return jsonify(response)

@app.route('/api/medical_forecast/<int:organization_id>', methods=['GET'])
def get_medical_forecast_for_organization(organization_id):
    conditions = fetch_medical_data(organization_id)

    if conditions.empty or len(conditions) < 5:
        return jsonify({"error": f"Not enough medical data available for organization ID {organization_id}."}), 200

    data = preprocess_medical_data(conditions)

    if data.empty or len(data) < 5:
        return jsonify({"error": f"Insufficient data after preprocessing for organization ID {organization_id}."}), 200

    model, training_columns = train_medical_model(data)

    future_dates = pd.date_range(start=datetime.datetime.now(), periods=6, freq='ME')
    forecast = forecast_medical(model, future_dates, training_columns)

    response = {}
    for index, row in forecast.iterrows():
        condition_name = row['condition_name']
        month_year = row['date'].strftime("%B %Y")

        if condition_name not in response:
            response[condition_name] = []

        response[condition_name].append({
            "predicted_date": month_year,
            "predicted_occurrences": f"{row['predicted_occurrences']} cases"
        })

    return jsonify(response)




if __name__ == "__main__":
    app.run(debug=True)
