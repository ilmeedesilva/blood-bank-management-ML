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


def fetch_data(organization_id=None):
    donations_query = """
    SELECT donation_date, blood_type, SUM(quantity) as quantity 
    FROM donations 
    WHERE status = 'ACTIVE'
    """
    if organization_id is not None:
        donations_query += f" AND organization_id = {organization_id}"

    donations_query += " GROUP BY donation_date, blood_type"
    stock_query = "SELECT last_updated, blood_type, quantity FROM stock"

    donations = pd.read_sql(donations_query, engine)
    stock = pd.read_sql(stock_query, engine)

    return donations, stock


def preprocess_data(donations, stock):
    donations['donation_date'] = pd.to_datetime(donations['donation_date'])
    stock['last_updated'] = pd.to_datetime(stock['last_updated'])

    donations_resampled = donations.set_index('donation_date').groupby('blood_type').resample('W').sum()
    donations_resampled = donations_resampled.drop(columns=['blood_type'])
    donations_resampled = donations_resampled.reset_index()

    return donations_resampled


# Feature Engineering for Date
def create_time_features(df, date_column):
    df['year'] = df[date_column].dt.year
    df['month'] = df[date_column].dt.month
    df['weekofyear'] = df[date_column].dt.isocalendar().week
    df['dayofweek'] = df[date_column].dt.dayofweek
    df['dayofmonth'] = df[date_column].dt.day
    df['dayofyear'] = df[date_column].dt.dayofyear

    return df


def train_model(data):
    data = create_time_features(data, 'donation_date')

    X = pd.get_dummies(data[['blood_type', 'year', 'month', 'weekofyear', 'dayofweek']], drop_first=True)

    y = data['quantity']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")

    joblib.dump(model, 'blood_stock_forecast_model.pkl')

    return model, X_train.columns


def forecast_stock(model, future_dates, training_columns):
    blood_types = ['A+', 'B+', 'O+', 'AB+', 'A-', 'B-', 'O-', 'AB-']

    future_data = pd.DataFrame(list(itertools.product(future_dates, blood_types)),
                               columns=['donation_date', 'blood_type'])
    future_data['donation_date'] = pd.to_datetime(future_data['donation_date'])

    future_data = create_time_features(future_data, 'donation_date')

    X_future = pd.get_dummies(future_data[['blood_type', 'year', 'month', 'weekofyear', 'dayofweek']], drop_first=True)

    # Align columns with the training set
    for col in training_columns:
        if col not in X_future.columns:
            X_future[col] = 0  # Add missing columns with 0 values
    X_future = X_future[training_columns]  # Reorder to match training column order

    predictions = model.predict(X_future)
    future_data['predicted_quantity'] = predictions

    return future_data


if __name__ == "__main__":
    donations, stock = fetch_data()
    data = preprocess_data(donations, stock)
    model, training_columns = train_model(data)

    future_dates = pd.date_range(start=datetime.datetime.now(), periods=6, freq='ME')
    forecast = forecast_stock(model, future_dates, training_columns)
    print(forecast)
