import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib  # This library saves our model

# 1. Load Data
print("Loading data...")
df = pd.read_csv("lagos_all_years.csv")

# 2. Preprocessing
# Convert dates and handle missing values
df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
df = df.dropna(subset=['datetime'])
df = df.fillna(method='ffill') # Fill gaps with previous day's data

# 3. Create Targets & Features
# We want to predict TOMORROW's Max Temp
df['target_next_max'] = df['tempmax'].shift(-1)

# Drop the last row (since it has no tomorrow)
df = df.dropna()

# These are the inputs the user will provide in the GUI
features = ['tempmax', 'tempmin', 'precip', 'humidity', 'windspeed']

# 4. Train the Model
# We use all available data to make the best possible model for the app
X = df[features]
y = df['target_next_max']

# Split just to print an accuracy score for you
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = Ridge(alpha=0.1)
model.fit(X_train, y_train)

# Check accuracy
predictions = model.predict(X_test)
error = mean_absolute_error(y_test, predictions)
print(f"Model Training Complete. MAE: {error:.2f} degrees Celsius")

# 5. Save the Model
# This creates a file 'weather_model.pkl' that contains the trained logic
joblib.dump(model, 'weather_model.pkl')
print("Model saved as 'weather_model.pkl'!")