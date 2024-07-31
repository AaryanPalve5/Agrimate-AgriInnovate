import pandas as pd
from datetime import datetime
from pmdarima import auto_arima
import pickle
import bz2

def load_data(file_path):
    """Load the rainfall data from a CSV file."""
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):
    """Preprocess the rainfall data."""
    df1 = df.loc[df['SUBDIVISION'] == 'Vidarbha'].iloc[:, 2:16]
    df2 = pd.melt(df1, id_vars='YEAR', value_vars=df1.columns[1:-1])
    df2['Date'] = df2['variable'] + ' ' + df2['YEAR'].astype(str)
    df2.loc[:, 'Date'] = df2['Date'].apply(lambda x: datetime.strptime(x, '%b %Y'))
    df2.columns = ['Year', 'Month', 'Rainfall', 'Date']
    df2.sort_values(by='Date', inplace=True)
    df3 = df2.drop(columns=["Month", "Year"])
    df3.set_index("Date", inplace=True)
    return df3

def train_model(data):
    """Train an ARIMA model."""
    model = auto_arima(y=data.Rainfall, m=12)
    return model

def save_model(model, file_path):
    """Save the trained model using compression."""
    with bz2.BZ2File(file_path, 'w') as f:
        pickle.dump(model, f)

def load_model(file_path):
    """Load the compressed model."""
    with bz2.BZ2File(file_path, 'rb') as f:
        model = pickle.load(f)
    return model

def make_predictions(model, n_periods):
    """Make predictions using the trained model."""
    predictions = model.predict(n_periods=n_periods)
    return predictions

def main():
    # Load data
    file_path = "Rainfall-Prediction/Dataset/Rainfall_Data_LL.csv"
    df = load_data(file_path)
    
    # Preprocess data
    df_processed = preprocess_data(df)
    
    # Train model
    model = train_model(df_processed)
    
    # Save model
    model_file_path = 'model2.pbz2'
    save_model(model, model_file_path)
    
    # Load model
    loaded_model = load_model(model_file_path)
    
    # Make predictions
    predictions = make_predictions(loaded_model, n_periods=12)
    print(predictions)

if __name__ == "__main__":
    main()
