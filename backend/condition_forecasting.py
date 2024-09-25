import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib
import datetime
import itertools
from db import get_db_connection

db_uri = get_db_connection()
engine = create_engine(db_uri)


def fetch_condition_data(organization_id=None):
    conditions_query = """
    SELECT pc.date, mc.condition_name, pc.blood_donation_count 
    FROM patient_conditions pc
    JOIN medical_conditions mc ON pc.condition_id = mc.id
    WHERE mc.status = 'ACTIVE'
    """
    if organization_id is not None:
        conditions_query += f" AND pc.organization_id = {organization_id}"

    conditions_query += " GROUP BY pc.date, mc.condition_name"
    
    conditions = pd.read_sql(conditions_query, engine)
    return conditions


def fetch_condition_names():
    condition_names_query = """
    SELECT mc.condition_name 
    FROM medical_conditions mc
    WHERE mc.status = 'ACTIVE'
    """
    condition_names = pd.read_sql(condition_names_query, engine)
    return condition_names['condition_name'].tolist()



def preprocess_condition_data(conditions):
    conditions['date'] = pd.to_datetime(conditions['date'])


    conditions_resampled = conditions.set_index('date').groupby('condition_name').resample('W').sum()
    conditions_resampled = conditions_resampled.drop(columns=['condition_name'])
    conditions_resampled = conditions_resampled.reset_index()

    return conditions_resampled


def create_time_features(df, date_column):
    df['year'] = df[date_column].dt.year
    df['month'] = df[date_column].dt.month
    df['weekofyear'] = df[date_column].dt.isocalendar().week
    df['dayofweek'] = df[date_column].dt.dayofweek
    df['dayofmonth'] = df[date_column].dt.day
    df['dayofyear'] = df[date_column].dt.dayofyear

    return df



def train_condition_model(data):
    data = create_time_features(data, 'date')

    X = pd.get_dummies(data[['condition_name', 'year', 'month', 'weekofyear', 'dayofweek']], drop_first=True)
    y = data['blood_donation_count']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")

    joblib.dump(model, 'condition_forecast_model.pkl')

    return model, X_train.columns


def forecast_conditions(model, future_dates, training_columns):
    condition_names = fetch_condition_names()

    future_data = pd.DataFrame(list(itertools.product(future_dates, condition_names)),
                               columns=['date', 'condition_name'])
    future_data['date'] = pd.to_datetime(future_data['date'])

    future_data = create_time_features(future_data, 'date')

    X_future = pd.get_dummies(future_data[['condition_name', 'year', 'month', 'weekofyear', 'dayofweek']], drop_first=True)

    for col in training_columns:
        if col not in X_future.columns:
            X_future[col] = 0
    X_future = X_future[training_columns]

    predictions = model.predict(X_future)
    future_data['predicted_blood_donation'] = predictions

    return future_data