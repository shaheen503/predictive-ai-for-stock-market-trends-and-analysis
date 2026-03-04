import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Generate sample dataset
data = {
    'Opening Price': np.random.randint(500, 2000, 100),
    'Closing Price': np.random.randint(500, 2000, 100),
    'Volume Traded': np.random.randint(1000, 100000, 100),
    'Market Sentiment': np.random.uniform(-1, 1, 100),
    'Decision': np.random.choice([0, 1], 100)  # 1 = BUY, 0 = SELL
}
df = pd.DataFrame(data)

# Features and target
X = df[['Opening Price', 'Closing Price', 'Volume Traded', 'Market Sentiment']]
y = df['Decision']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
with open("model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

print("Model trained and saved as model.pkl")
