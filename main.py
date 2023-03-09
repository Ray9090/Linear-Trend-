import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.animation as animation
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from matplotlib import pyplot as plt
from scipy.stats import boxcox, linregress

dataset = pd.read_csv(r"C:\Users\Q610267\Downloads\exd_download\Test-Interview\Test-Interview\30_2022_01_Greifer_Schritt510_Anforderung_bis_Endschalter.csv")



#plot the distribution curve and see whether the plot follows the bell curve shape
sns.distplot(dataset.s_step510_ms)
plt.show()

sns.distplot(dataset.step510)
plt.show()

sns.distplot(dataset.step510,dataset.s_step510_ms)
plt.show()

# Define a function to convert LDAP time to milliseconds
def ldap_to_ms(ldap_time):
    return (ldap_time / 10000) - 11644473600000

dataset['ms_time'] = dataset['s_step510_ms'].apply(ldap_to_ms)

dataset['difference'] = 0

# Iterate through each row of the dataframe
for i in range(1, len(dataset)):
    # Calculate the difference and store it in the 'difference' column
    dataset.loc[i, 'difference'] = dataset.loc[i, 'ms_time'] - dataset.loc[i-1, 'ms_time']


print (stats.shapiro(dataset.s_step510_ms))
#ShapiroResult(statistic=0.0001507401466369629, pvalue=0.0)

# to see whether the distribution of data follows a normal distribution
fig = sm.qqplot(dataset.step510,line='s')
plt.show()

print (stats.shapiro(dataset.step510))
#ShapiroResult(statistic=0.9505149126052856, pvalue=0.0)

df = dataset

# Preprocess the data to make the residuals normally distributed
# You can use the Box-Cox transformation to achieve this
df['step510'], _ = boxcox(df['step510'])
df['step510'] = df['step510'] - df['step510'].mean()

# Perform the linear trend analysis
slope, intercept, rvalue, pvalue, stderr = linregress(df['s_step510_ms'], df['step510'])

# Print the results
print(f"Slope: {slope:.2f}")
print(f"Intercept: {intercept:.2f}")
print(f"R-value: {rvalue:.2f}")
print(f"P-value: {pvalue:.2f}")
print(f"Standard error: {stderr:.2f}")


#transform the data using box-cox
sample_transformed, lambd = stats.boxcox(df['step510'])

#plot the distribution curve and QQ-plot for transformed data
sns.distplot(sample_transformed)
plt.show()
fig = sm.qqplot(sample_transformed,line='s')
plt.show()


# Load the data
data = df

# Preprocess the data points
y = np.sqrt(data['step510']) # square root transformation
X = sm.add_constant(data['s_step510_ms']) # add constant term

# Fit the linear model
model = sm.OLS(y, X).fit()

# Validate the model
print(model.summary())



###
# read the dataset
data = pd.read_csv(r"C:\Users\Q610267\Downloads\exd_download\Test-Interview\Test-Interview\30_2022_01_Greifer_Schritt510_Anforderung_bis_Endschalter.csv")

# preprocess the data points to make the residuals normally distributed
residuals = data["step510"] - data["step510"].mean()
residuals_normaltest = stats.normaltest(residuals)
if residuals_normaltest.pvalue < 0.05:
    data["step510"], _ = stats.boxcox(data["step510"])

# fit the linear model
X = sm.add_constant(data["s_step510_ms"])
model = sm.OLS(data["step510"], X).fit()

# print the model summary
print(model.summary())

###
import pandas as pd
import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PowerTransformer
from sklearn.pipeline import Pipeline

# read the dataset
data = pd.read_csv(r"C:\Users\Q610267\Downloads\exd_download\Test-Interview\Test-Interview\30_2022_01_Greifer_Schritt510_Anforderung_bis_Endschalter.csv")


# create a pipeline for preprocessing the data and fitting the model
preprocess = Pipeline([
    ("transformer", PowerTransformer(method="yeo-johnson")),
    ("regressor", LinearRegression())
])
model = TransformedTargetRegressor(regressor=preprocess, transformer=PowerTransformer(method="yeo-johnson"))

# fit the model
model.fit(data[["s_step510_ms"]], data["step510"])

# print the coefficients and intercept of the model
print("Coefficients:", model.regressor_.named_steps["regressor"].coef_)
print("Intercept:", model.regressor_.named_steps["regressor"].intercept_)


# Step 1: Import necessary libraries and load the dataset
import pandas as pd
import numpy as np
from scipy.stats import boxcox
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# load the dataset
data = pd.read_csv(r"C:\Users\Q610267\Downloads\exd_download\Test-Interview\Test-Interview\30_2022_01_Greifer_Schritt510_Anforderung_bis_Endschalter.csv")

# Step 2: Preprocess the data by transforming the target feature
# Transform using Box-Cox transformation to obtain normally distributed residuals
data['step510'], _ = boxcox(data['step510'])

# Step 3: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['s_step510_ms'], data['step510'], test_size=0.2, random_state=42)

# Step 4: Train a suitable regression model on the training set
X_train = np.array(X_train).reshape(-1, 1)
y_train = np.array(y_train).reshape(-1, 1)
lr = LinearRegression().fit(X_train, y_train)

# Step 5: Evaluate the model on the testing set
X_test = np.array(X_test).reshape(-1, 1)
y_test = np.array(y_test).reshape(-1, 1)
y_pred = lr.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

# Step 6: Check the model parameters and errors to validate the findings
print('Coefficients:', lr.coef_)
print('Intercept:', lr.intercept_)
print('Mean Squared Error:', mse)

# compute R-squared and adjusted R-squared
ss_tot = ((y_test - y_test.mean())**2).sum()
ss_res = ((y_test - y_pred)**2).sum()
r_squared = 1 - (ss_res / ss_tot)
n = len(y_test)
p = 1
adj_r_squared = 1 - ((1 - r_squared)*(n - 1) / (n - p - 1))

# print the results
print('Coefficients:', lr.coef_)
print('Intercept:', lr.intercept_)
print('Mean Squared Error:', mse)
print('R-squared:', r_squared)
print('Adjusted R-squared:', adj_r_squared)

# plot residuals
import seaborn as sns
import matplotlib.pyplot as plt
residuals = y_test - y_pred
sns.scatterplot(x=X_test.ravel(), y=residuals.ravel())
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('s_step510_ms')
plt.ylabel('Residuals')
plt.show()



#####
# import necessary libraries
from sklearn.linear_model import Ridge

# preprocess the data using Box-Cox transformation
data['step510'], _ = boxcox(data['step510'])

# split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['s_step510_ms'], data['step510'], test_size=0.2, random_state=42)

# train a Ridge regression model on the training set
X_train = np.array(X_train).reshape(-1, 1)
y_train = np.array(y_train).reshape(-1, 1)
ridge = Ridge(alpha=0.1).fit(X_train, y_train)

# evaluate the model on the testing set
X_test = np.array(X_test).reshape(-1, 1)
y_test = np.array(y_test).reshape(-1, 1)
y_pred = ridge.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r_squared = ridge.score(X_test, y_test)

# print the results
print('Coefficients:', ridge.coef_)
print('Intercept:', ridge.intercept_)
print('Mean Squared Error:', mse)
print('R-squared:', r_squared)



# import necessary libraries
from sklearn.ensemble import RandomForestRegressor

# preprocess the data using Box-Cox transformation
data['step510'], _ = boxcox(data['step510'])

# split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['s_step510_ms'], data['step510'], test_size=0.2, random_state=42)

# train a Random Forest regressor on the training set
X_train = np.array(X_train).reshape(-1, 1)
y_train = np.array(y_train).ravel()
rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42).fit(X_train, y_train)

# evaluate the model on the testing set
X_test = np.array(X_test).reshape(-1, 1)
y_test = np.array(y_test).ravel()
y_pred = rf.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r_squared = rf.score(X_test, y_test)

# print the results
print('Mean Squared Error:', mse)
print('R-squared:', r_squared)


# import necessary libraries
import tensorflow as tf
from tensorflow.keras import layers

# preprocess the data using Box-Cox transformation
data['step510'], _ = boxcox(data['step510'])

# split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['s_step510_ms'], data['step510'], test_size=0.2, random_state=42)

# create a simple neural network with a single hidden layer
model = tf.keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(1,)),
    layers.Dense(1)
])

# compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')

# train the model on the training set
history = model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test), verbose=0)

# evaluate the model on the testing set
mse = model.evaluate(X_test, y_test)
y_pred = model.predict(X_test)
r_squared = 1 - mse / np.var(y_test)

# plot the learning curves
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.legend()
plt.show()

# plot the predicted vs actual values
plt.scatter(X_test, y_test, label='actual')
plt.scatter(X_test, y_pred, label='predicted')
plt.xlabel('s_step510_ms')
plt.ylabel('step510')
plt.legend()
plt.show()

# print the results
print('Mean Squared Error:', mse)
print('R-squared:', r_squared)

### creating a new feature
# preprocess the data by taking the log of the s_step510_ms column
data['s_step510_ms'] = np.log(data['s_step510_ms'])

# create a new feature that represents the time elapsed between consecutive cylinder openings
data['time_elapsed'] = data['s_step510_ms'].diff().fillna(0)

# split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data[['s_step510_ms', 'time_elapsed']], data['step510'], test_size=0.2, random_state=42)

# fit a linear regression model on the training set
lr = LinearRegression().fit(X_train, y_train)

# evaluate the model on the testing set
y_pred = lr.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r_squared = lr.score(X_test, y_test)

# print the results
print('Coefficients:', lr.coef_)
print('Intercept:', lr.intercept_)
print('Mean Squared Error:', mse)
print('R-squared:', r_squared)


##### create more features
data = pd.read_csv(r"C:\Users\Q610267\Downloads\exd_download\Test-Interview\Test-Interview\30_2022_01_Greifer_Schritt510_Anforderung_bis_Endschalter.csv")

from scipy import signal

# preprocess the data by taking the log of the s_step510_ms column
data['s_step510_ms'] = np.log(data['s_step510_ms'])


# create additional features
data['step510_mean'] = data['step510'].rolling(window=5, min_periods=1).mean()
data['step510_max'] = data['step510'].rolling(window=5, min_periods=1).max()
data['step510_peak'] = signal.find_peaks(data['step510'], height=0)[0].size
data['step510_rms'] = np.sqrt(np.mean(data['step510']**2))
data['step510_var'] = data['step510'].rolling(window=5, min_periods=1).var()
data['step510_std'] = data['step510'].rolling(window=5, min_periods=1).std()
data['step510_power'] = np.sum(data['step510']**2) / len(data['step510'])
data['step510_kurtosis'] = data['step510'].kurtosis()
data['step510_skewness'] = data['step510'].skew()

data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.dropna(inplace=True)

# split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data.drop(columns=['step510']), data['step510'], test_size=0.2,
                                                    random_state=42)

# fit a linear regression model on the training set
lr = LinearRegression().fit(X_train, y_train)

# evaluate the model on the testing set
y_pred = lr.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r_squared = lr.score(X_test, y_test)

# print the results
print('Coefficients:', lr.coef_)
print('Intercept:', lr.intercept_)
print('Mean Squared Error:', mse)
print('R-squared:', r_squared)


### using correlation analysis
# load the dataset
data = pd.read_csv(r"C:\Users\Q610267\Downloads\exd_download\Test-Interview\Test-Interview\30_2022_01_Greifer_Schritt510_Anforderung_bis_Endschalter.csv")

# preprocess the data by taking the log of the s_step510_ms column
data['s_step510_ms'] = np.log(data['s_step510_ms'])

# create additional features
data['step510_mean'] = data['step510'].rolling(window=10, min_periods=1).mean()
data['step510_max'] = data['step510'].rolling(window=10, min_periods=1).max()
data['step510_peak'] = signal.find_peaks(data['step510'], height=0)[0].size
data['step510_rms'] = np.sqrt(np.mean(data['step510']**2))
data['step510_var'] = data['step510'].rolling(window=10, min_periods=1).var()
data['step510_std'] = data['step510'].rolling(window=10, min_periods=1).std()
data['step510_power'] = np.sum(data['step510']**2) / len(data['step510'])
data['step510_kurtosis'] = data['step510'].kurtosis()
data['step510_skewness'] = data['step510'].skew()

# remove any rows containing NaN or infinite values
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.dropna(inplace=True)

# select the most important features using correlation analysis
corr_matrix = data.corr()
important_features = corr_matrix.index[abs(corr_matrix['step510']) >= 0.1]

# split the dataset into training and testing sets using the selected features
X_train, X_test, y_train, y_test = train_test_split(data[important_features], data['step510'], test_size=0.2, random_state=42)

# fit a linear regression model on the training set
lr = LinearRegression().fit(X_train, y_train)

# evaluate the model on the testing set
y_pred = lr.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r_squared = lr.score(X_test, y_test)

# print the results
print('Selected Features:', important_features)
print('Coefficients:', lr.coef_)
print('Intercept:', lr.intercept_)
print('Mean Squared Error:', mse)
print('R-squared:', r_squared)
