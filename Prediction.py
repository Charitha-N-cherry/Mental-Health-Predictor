# 1️⃣ IMPORT REQUIRED LIBRARIES
import pandas as pd  # for reading and handling data
from sklearn.model_selection import train_test_split  # to split dataset
from sklearn.ensemble import RandomForestRegressor  # machine learning model
from sklearn.metrics import mean_squared_error, r2_score  # evaluation metrics
import matplotlib.pyplot as plt

# 👉 Explanation:
# We use pandas for dataframes, scikit-learn for ML models and evaluation.

# 2️⃣ LOAD YOUR DATASET
file_path = 'MentalHealth.xlsx'  # update path if needed
df = pd.read_excel(file_path)

# 👉 This loads your Excel file into a pandas dataframe.

# 3️⃣ SEPARATE FEATURES AND TARGET
X = df.drop(columns=['Stress_Level'])  # features (input columns)
# Drop all non-numeric columns
X = X.select_dtypes(include=['int64', 'float64'])
y = df['Stress_Level']  # target (label)

# 👉 We separate predictors (X) from what we want to predict (y)

# 4️⃣ SPLIT DATA INTO TRAINING AND TESTING
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 👉 80% for training the model, 20% for testing
# random_state ensures same split every time you run

# 5️⃣ INITIALIZE AND TRAIN THE MODEL
model = RandomForestRegressor()  # create model object
model.fit(X_train, y_train)  # train the model on training data

# 👉 RandomForest is an ensemble model (multiple decision trees combined)

# 6️⃣ MAKE PREDICTIONS ON TEST DATA
y_pred = model.predict(X_test)

# 👉 Now we predict stress levels for unseen test data

# 7️⃣ EVALUATE MODEL PERFORMANCE
mse = mean_squared_error(y_test, y_pred)  # mean squared error
r2 = r2_score(y_test, y_pred)  # R-squared metric

print("Mean Squared Error:", mse)
print("R² Score:", r2)

# 👉 Lower MSE is better; R² closer to 1 is better

# 8️⃣ OPTIONAL: SHOW SOME PREDICTIONS VS ACTUAL
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(results.head(10))

# 👉 This prints the first 10 actual vs predicted stress levels

#Predicted vs Actual scatter plot
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Stress Level')
plt.ylabel('Predicted Stress Level')
plt.title('Actual vs Predicted Stress Level')
plt.plot([1,10], [1,10], color='red')  # perfect prediction line
plt.show()








