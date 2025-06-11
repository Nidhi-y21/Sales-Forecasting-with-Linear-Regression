import unittest						
import pandas as pd						
from datetime import datetime						
from main import load_data, preprocess_data, train_model, create_forecast, generate_future_dates						
						
class TestForecast(unittest.TestCase):						
def setUp(self):						
# Create sample data						
self.sample_data = pd.DataFrame({						
'Date': pd.date_range(start='2024-01-01', periods=10),						
'Product': ['A'] * 10,						
'Revenue': range(100, 1000, 100)						
})						
						
def test_preprocess_data(self):						
processed_df = preprocess_data(self.sample_data.copy())						
self.assertTrue('DayOfWeek' in processed_df.columns)						
self.assertTrue('Month' in processed_df.columns)						
self.assertTrue('Year' in processed_df.columns)						
						
def test_model_training(self):						
df = preprocess_data(self.sample_data.copy())						
model = train_model(df)						
self.assertIsNotNone(model)						
						
def test_forecast_generation(self):						
last_date = self.sample_data['Date'].max()						
future_dates = generate_future_dates(last_date, periods=5)						
self.assertEqual(len(future_dates), 5)						
						
if __name__ == '__main__':						
unittest.main()						
