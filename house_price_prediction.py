import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import math
from sklearn.metrics import mean_squared_error
import random
import warnings

warnings.filterwarnings("ignore")

df = pd.read_excel('dataset/housedata.xls', sheet_name='Sheet1')
df.head(5)

df.columns = [col.replace(',', '') for col in df.columns]
df = df.rename(columns={'location ': 'location', '#bedrooms': 'bedrooms', '#bathrooms': 'bathrooms', 'house area in 1000 square feet': 'house_area', '1 if condo 0 otherwise ': 'is_condo', 'selling price in 1000 dollars': 'price'})

df.dtypes
df['price'] = df['price'].replace({';': ''}, regex=True).astype(int)

X = df[['house_area','bedrooms']]
y = df['price']
np.random.seed(2)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=695, random_state=42)

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

fig, ax = plt.subplots(1, 2, figsize=(18, 6))
ax[0].scatter(X['house_area'], y, marker ='o', color ='blue', label='Actual Prices')
ax[0].plot([min(X['house_area']), max(X['house_area'])], [min(y_pred), max(y_pred)], color='red', linestyle='--', label='Predicted Prices')
ax[0].set_title('House Area vs. House Price')
ax[0].set_xlabel('House Area (sq.ft)')
ax[0].set_ylabel('House Price ($)')
ax[0].legend()

ax[1].scatter(X['bedrooms'], y, marker ='o', color ='blue', label='Actual Prices')
ax[1].plot([min(X['bedrooms']), max(X['bedrooms'])], [min(y_pred), max(y_pred)], color='red', linestyle='--', label='Predicted Prices')
ax[1].set_title('Bedrooms vs. House Price')
ax[1].set_xlabel('Bedrooms')
ax[1].set_ylabel('House Price ($)')
ax[1].legend()

plt.show()

test_data = {
    'house_area': [846, 1324, 1150, 3037, 3984],
    'bedrooms': [1, 2, 3, 4, 5],
    'price': [115000, 234500, 198000, 528000, 572500]
}
df_test = pd.DataFrame(test_data)

X_test = df_test[['house_area', 'bedrooms']]
y_test_pred = model.predict(X_test)

fig, ax = plt.subplots(1, 2, figsize=(18, 6))
ax[0].scatter(X_test['house_area'], df_test['price'], marker ='o', color ='blue', label='Actual Prices')
ax[0].plot([min(X_test['house_area']), max(X_test['house_area'])], [min(y_test_pred), max(y_test_pred)], color='red', linestyle='--', label='Predicted Prices')
ax[0].set_title('House Area vs. House Price')
ax[0].set_xlabel('House Area (sq.ft)')
ax[0].set_ylabel('House Price ($)')
ax[0].legend()
ax[1].scatter(X_test['bedrooms'], df_test['price'], marker ='o', color ='blue', label='Actual Prices')
ax[1].plot([min(X_test['bedrooms']), max(X_test['bedrooms'])], [min(y_test_pred), max(y_test_pred)], color='red', linestyle='--', label='Predicted Prices')
ax[1].set_title('Bedrooms vs. House Price')
ax[1].set_xlabel('Bedrooms')
ax[1].set_ylabel('House Price ($)')
ax[1].legend()
plt.show()

mse_train = mean_squared_error(y, y_pred)
rmse_train = math.sqrt(mse_train)

mse_test = mean_squared_error(df_test['price'], y_test_pred)
rmse_test = math.sqrt(mse_test)

print(f'Root Mean Square Error (Train): {rmse_train:.2f}')
print(f'Root Mean Square Error (Test): {rmse_test:.2f}')

X_complex = pd.DataFrame({
    'f1': np.ones_like(df['house_area']),
    'f2': df['house_area'],
    'f3': np.maximum(df['house_area'] - 1.5, 0),
    'f4': df['bedrooms'],
    'f5': df['is_condo'],
    'f6': np.where(df['location']==2, 1, 0),
    'f7': np.where(df['location']==3, 1, 0),
    'f8': np.where(df['location']==4, 1, 0),
})
y = df['price']

model = LinearRegression()
model.fit(X_complex, y)
y_pred = model.predict(X_complex)

fig, ax = plt.subplots(1, 2, figsize=(18, 6))
ax[0].scatter(X_complex['f2'], y, marker ='o', color ='blue', label='Actual Prices')
ax[0].plot([min(X_complex['f2']), max(X_complex['f2'])], [min(y_pred), max(y_pred)], color='red', linestyle='--', label='Predicted Prices')
ax[0].set_title('House Area vs. House Price')
ax[0].set_xlabel('House Area (sq.ft)')
ax[0].set_ylabel('House Price ($)')
ax[0].legend()
ax[1].scatter(X_complex['f4'], y, marker ='o', color ='blue', label='Actual Prices')
ax[1].plot([min(X_complex['f4']), max(X_complex['f4'])], [min(y_pred), max(y_pred)], color='red', linestyle='--', label='Predicted Prices')
ax[1].set_title('Bedrooms vs. House Price')
ax[1].set_xlabel('Bedrooms')
ax[1].set_ylabel('House Price ($)')
ax[1].legend()
plt.show()

X = df[['house_area','bedrooms']]
y = df['price']
np.random.seed(2)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=695, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

plt.figure(figsize=(8,6))
plt.scatter(y_test, y_test_pred, color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test_pred), max(y_test_pred)], color='red', linestyle='--')
plt.title("Actual vs. Predicted Sale Price")
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.show()

np.random.seed(2)
X_train, X_test, y_train, y_test = train_test_split(X_complex, y, train_size=695, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

plt.figure(figsize=(8,6))
plt.scatter(y_test, y_test_pred, color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test_pred), max(y_test_pred)], color='red', linestyle='--')
plt.title("Actual vs. Predicted Sale Price")
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.show()
