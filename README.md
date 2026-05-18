# Ex.No: 6               HOLT WINTERS METHOD
### Date: 18-05-2026

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
```py
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

data_monthly = data['money'].resample('MS').sum()
data_monthly.head()
data_monthly.plot()

scaler = MinMaxScaler()
scaled_data = pd.Series(scaler.fit_transform(data_monthly.values.reshape(-1, 1)).flatten())
scaled_data.plot()
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(data_monthly, model="additive", period=6)
decomposition.plot()
plt.show()

scaled_data=scaled_data+1
train_data = scaled_data[:int(len(scaled_data) * 0.8)]
test_data = scaled_data[int(len(scaled_data) * 0.8):]
model_add = ExponentialSmoothing(train_data, trend='add', seasonal='mul', seasonal_periods=4).fit()
test_predictions_add = model_add.forecast(steps=len(test_data))
ax=train_data.plot()
test_predictions_add.plot(ax=ax)
test_data.plot(ax=ax)
ax.legend(["train_data", "test_predictions_add","test_data"])
ax.set_title('Visual evaluation')
rmse = np.sqrt(mean_squared_error(test_data, test_predictions_add))
print(f"RMSE for test predictions: {rmse}")
print(f"Standard Deviation of scaled data: {np.sqrt(scaled_data.var())}")
print(f"Mean of scaled data: {scaled_data.mean()}")

import numpy as np
data_monthly_positive = data_monthly.replace(0, np.finfo(float).eps)
final_model = ExponentialSmoothing(data_monthly_positive, trend='add', seasonal='mul', seasonal_periods=4).fit()
final_predictions = final_model.forecast(steps=int(len(data_monthly)/4))
ax=data_monthly.plot()
final_predictions.plot(ax=ax)
ax.legend(["data_monthly", "final_predictions"])
ax.set_xlabel('Number of monthly passengers')
ax.set_ylabel('Months')
ax.set_title('Prediction')
```
### OUTPUT:
SCALED_DATA PLOT:


<img width="582" height="449" alt="image 1" src="https://github.com/user-attachments/assets/68fb5819-ff8e-4835-9153-8bbdecff8591" />

DECOMPOSED PLOT:


<img width="630" height="470" alt="image 2" src="https://github.com/user-attachments/assets/734a7445-fa56-4dec-a060-60d4e3d6c127" />

TEST PREDICTION:


<img width="547" height="435" alt="image 3" src="https://github.com/user-attachments/assets/50979629-a026-45ee-8b73-d1138ed4d337" />

FINAL PREDICTION:


<img width="589" height="471" alt="image 4" src="https://github.com/user-attachments/assets/6b929692-8e0a-46fb-a83a-c4d1ee85a453" />


### RESULT:
Thus the program run successfully based on the Holt Winters Method model.
