# Ex.No: 6               HOLT WINTERS METHOD
### Date: 07/10/2025



### AIM:

### ALGORITHM:
1. You import the necessary libraries
2. You load a CSV file containing daily sales data into a DataFrame, parse the 'date' column as
datetime, and perform some initial data exploration
3. You group the data by date and resample it to a monthly frequency (beginning of the month
4. You plot the time series data
5. You import the necessary 'statsmodels' libraries for time series analysis
6. You decompose the time series data into its additive components and plot them:
7. You calculate the root mean squared error (RMSE) to evaluate the model's performance
8. You calculate the mean and standard deviation of the entire sales dataset, then fit a Holt-
Winters model to the entire dataset and make future predictions
9. You plot the original sales data and the predictions
### PROGRAM:
Import libraries:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose
```
Load & clean dataset:
```
data = pd.read_csv('GoogleStockPrices.csv', parse_dates=['Date'], index_col='Date')
print(data.head())
data = data[['Close']].copy()
data_monthly = data.resample('MS').mean()
print(data_monthly.head())
data_monthly.plot(title='Monthly Google Stock Prices', figsize=(10,5))
plt.show()
```
Scale data & plot:
```
scaler = MinMaxScaler()
scaled_data = pd.Series(
    scaler.fit_transform(data_monthly.values.reshape(-1, 1)).flatten(),
    index=data_monthly.index
)
scaled_data.plot(title='Scaled Data')
plt.show()
```
Decomposition:
```
decomposition = seasonal_decompose(data_monthly, model="additive")
decomposition.plot()
plt.show()
```
Train/Test split:
```
scaled_data = scaled_data + 1
train_data = scaled_data[:int(len(scaled_data) * 0.8)]
test_data = scaled_data[int(len(scaled_data) * 0.8):]
```
Holt-Winters model & test prediction:
```
model_add = ExponentialSmoothing(train_data, trend='add', seasonal='mul', seasonal_periods=12).fit()
test_predictions_add = model_add.forecast(steps=len(test_data))
ax = train_data.plot(label='Train', figsize=(10,5))
test_data.plot(ax=ax, label='Test')
test_predictions_add.plot(ax=ax, label='Predictions')
ax.legend()
ax.set_title('Holt-Winters Model Evaluation')
plt.show()
```
Model performance metrics:
```
rmse = np.sqrt(mean_squared_error(test_data, test_predictions_add))
mean_actual = np.mean(test_data)
mean_pred = np.mean(test_predictions_add)
std_actual = np.std(test_data)
std_pred = np.std(test_predictions_add)

print("📊 Model Performance Metrics")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean of Actual Test Data: {mean_actual:.4f}")
print(f"Mean of Predicted Data: {mean_pred:.4f}")
print(f"Standard Deviation of Actual Data: {std_actual:.4f}")
print(f"Standard Deviation of Predicted Data: {std_pred:.4f}")
```
Final model & forecast:
```
final_model = ExponentialSmoothing(scaled_data, trend='add', seasonal='mul', seasonal_periods=12).fit()
future_steps = 12
final_predictions = final_model.forecast(steps=future_steps)
ax = scaled_data.plot(label='Original Data', figsize=(10,5))
final_predictions.plot(ax=ax, label='Future Forecast')
ax.legend()
ax.set_title('Google Stock Price Forecast')
plt.show()
```

### OUTPUT:
### Scaled_data plot:
<img width="715" height="576" alt="image" src="https://github.com/user-attachments/assets/fc48a28b-5f89-4b1b-8d4b-7446f717e139" />
Decomposed plot:<img width="813" height="618" alt="image" src="https://github.com/user-attachments/assets/693e5d4d-2231-43dc-9964-87cfebbbb3c5" />


### TEST_PREDICTION:

<img width="892" height="492" alt="image" src="https://github.com/user-attachments/assets/45d6ddb5-7f31-49d5-96d4-590edb14417a" />
Model performance metrics:


Root Mean Squared Error (RMSE): 0.6935

Mean of Actual Test Data: 1.7164

Mean of Predicted Data: 1.1065

Standard Deviation of Actual Data: 0.1685

Standard Deviation of Predicted Data: 0.1648

### FINAL_PREDICTION:
<img width="880" height="493" alt="image" src="https://github.com/user-attachments/assets/74c3602a-7e64-4ca5-93bb-ca7a3a568dc0" />
<img width="813" height="618" alt="image" src="https://github.com/user-attachments/assets/693e5d4d-2231-43dc-9964-87cfebbbb3c5" />
### RESULT:

Thus the program run successfully based on the Holt Winters Method model.
