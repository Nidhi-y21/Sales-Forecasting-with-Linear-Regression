import pandas as pd				
import numpy as np				
from datetime import datetime, timedelta				
from sklearn.model_selection import train_test_split				
from sklearn.linear_model import LinearRegression				
import plotly.express as px				
				
def load_data(file_path):				
df = pd.read_csv(file_path)				
df['Date'] = pd.to_datetime(df['Date'])				
return df				
				
def preprocess_data(df):				
df['DayOfWeek'] = df['Date'].dt.dayofweek				
df['Month'] = df['Date'].dt.month				
df['Year'] = df['Date'].dt.year				
return df				
				
def train_model(df):				
X = df[['DayOfWeek', 'Month', 'Year']]				
y = df['Revenue']				
model = LinearRegression()				
model.fit(X, y)				
return model				
				
def generate_future_dates(last_date, periods=90):				
return pd.date_range(start=last_date + timedelta(days=1), periods=periods, freq='D')				
				
def create_forecast(model, future_dates):				
future_df = pd.DataFrame({'Date': future_dates})				
future_df['DayOfWeek'] = future_df['Date'].dt.dayofweek				
future_df['Month'] = future_df['Date'].dt.month				
future_df['Year'] = future_df['Date'].dt.year				
future_df['Predicted_Revenue'] = model.predict(future_df[['DayOfWeek', 'Month', 'Year']])				
return future_df				
				
def product_wise_forecast(df):				
products = df['Product'].unique()				
all_forecasts = pd.DataFrame()				
				
for product in products:				
product_data = df[df['Product'] == product]				
model = train_model(product_data)				
future_dates = generate_future_dates(df['Date'].max())				
future_df = create_forecast(model, future_dates)				
future_df['Product'] = product				
all_forecasts = pd.concat([all_forecasts, future_df])				
				
return all_forecasts				
				
def plot_historical_vs_predicted(df):				
fig = px.line(df, x='Date', y=['Revenue', 'Predicted_Revenue'],				
title='Actual vs Predicted Revenue',				
labels={'value': 'Revenue', 'variable': 'Type'})				
fig.write_html('plots/historical_vs_predicted.html')				
				
def plot_product_forecast(df):				
fig = px.line(df, x='Date', y='Predicted_Revenue', color='Product',				
title='Revenue Forecast by Product')				
fig.write_html('plots/product_forecast.html')				
