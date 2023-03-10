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


##### again linear trend wiht box cox

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

# transform the residuals using the Box-Cox transformation
from scipy.stats import boxcox
y_pred = lr.predict(X_test)
residuals = y_test - y_pred

# add a constant value to make the residuals positive
residuals_min = np.abs(np.min(residuals)) + 0.1
residuals_pos = residuals + residuals_min

# transform the positive residuals using the Box-Cox transformation
residuals_bc, lambda_ = boxcox(residuals_pos)

# fit a linear regression model on the transformed residuals
lr_res = LinearRegression().fit(X_test, residuals_bc)

# evaluate the model on the testing set
y_pred_res_bc = lr_res.predict(X_test)
mse_res_bc = mean_squared_error(residuals_bc, y_pred_res_bc)
r_squared_res_bc = lr_res.score(X_test, residuals_bc)

# print the results
print('Selected Features:', important_features)
print('Coefficients:', lr_res.coef_)
print('Intercept:', lr_res.intercept_)
print('Mean Squared Error (Box-Cox transformed residuals):', mse_res_bc)
print('R-squared (Box-Cox transformed residuals):', r_squared_res_bc)


### linear trend Yeo Johson transformation

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

# transform the residuals using the Yeo-Johnson transformation
from sklearn.preprocessing import PowerTransformer
pt = PowerTransformer(method='yeo-johnson', standardize=False)
y_pred = lr.predict(X_test)
residuals = y_test - y_pred
residuals_yj = pt.fit_transform(residuals.values.reshape(-1, 1))

# fit a linear regression model on the transformed residuals
lr_res = LinearRegression().fit(X_test, residuals_yj)

# evaluate the model on the testing set
y_pred_res_yj = lr_res.predict(X_test)
mse_res_yj = mean_squared_error(residuals_yj, y_pred_res_yj)
r_squared_res_yj = lr_res.score(X_test, residuals_yj)


# print the results
print('Selected Features:', important_features)
print('Coefficients:', lr_res.coef_)
print('Intercept:', lr_res.intercept_)
print('Mean Squared Error (Yeo-Johnson transformed residuals):', mse_res_yj)
print('R-squared (Yeo-Johnson transformed residuals):', r_squared_res_yj)


#### QuantileTransformer
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

# transform the residuals using the QuantileTransformer
from sklearn.preprocessing import QuantileTransformer
qt = QuantileTransformer(output_distribution='normal')
y_pred = lr.predict(X_test)
residuals = y_test - y_pred
residuals_qt = qt.fit_transform(residuals.values.reshape(-1, 1))

# fit a linear regression model on the transformed residuals
lr_res = LinearRegression().fit(X_test, residuals_qt)

# evaluate the model on the testing set
y_pred_res_qt = lr_res.predict(X_test)
mse_res_qt = mean_squared_error(residuals_qt, y_pred_res_qt)
r_squared_res_qt = lr_res.score(X_test, residuals_qt)

# print the results
print('Selected Features:', important_features)
print('Coefficients:', lr_res.coef_)
print('Intercept:', lr_res.intercept_)
print('Mean Squared Error (Quantile transformed residuals):', mse_res_qt)
print('R-squared (Quantile transformed residuals):', r_squared_res_qt)


### logit transformation
# load the dataset
data = pd.read_csv(r"C:\Users\Q610267\Downloads\exd_download\Test-Interview\
    Test-Interview\30_2022_01_Greifer_Schritt510_Anforderung_bis_Endschalter.csv")

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

# generate the X and y arrays from the preprocessed data
X = data[important_features]
y = data['step510']

# split the arrays into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# fit a linear regression model on the training set
lr = LinearRegression().fit(X_train, y_train)

# transform the residuals using the logit transformation
from scipy.special import logit
y_pred = lr.predict(X_test)
residuals = y_test - y_pred
residuals_logit = logit(residuals[~np.isnan(residuals) & ~np.isinf(residuals)])

# replace NaN and infinite values in residuals_logit with the mean value
residuals_logit[np.isnan(residuals_logit) | np.isinf(residuals_logit)] = np.mean(residuals_logit[~np.isnan(residuals_logit) & ~np.isinf(residuals_logit)])

# fit a linear regression model on the transformed residuals
lr_res = LinearRegression().fit(X_test, residuals_logit)

# evaluate the model on the testing set
y_pred_res_logit = lr_res.predict(X_test)
mse_res_logit = mean_squared_error(residuals_logit, y_pred_res_logit)
r_squared_res_logit = lr_res.score(X_test, residuals_logit)

# print the results
print('Selected Features:', important_features)
print('Coefficients:', lr_res.coef_)
print('Intercept:', lr_res.intercept_)
print('Mean Squared Error (logit transformed residuals):', mse_res_logit)
print('R-squared (logit transformed residuals):', r_squared_res_logit)


#### too remove out linears
from scipy.stats import zscore

# calculate the Z-score for each data point
data['step510_zscore'] = zscore(data['step510'])

# remove data points with a Z-score greater than 3 or less than -3
data = data[(data['step510_zscore'] > -3) & (data['step510_zscore'] < 3)]


##### plot with few features

import pandas as pd
import numpy as np
from scipy.stats import zscore
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from scipy.special import inv_boxcox
from scipy.stats import boxcox
from scipy import signal
import matplotlib.pyplot as plt


# load the dataset
data = pd.read_csv(r"C:\Users\Q610267\Downloads\exd_download\Test-Interview\Test-Interview\30_2022_01_Greifer_Schritt510_Anforderung_bis_Endschalter.csv")

# preprocess the data by taking the log of the s_step510_ms column
data['s_step510_ms'] = np.log(data['s_step510_ms'])

# create a new feature that represents the time elapsed between consecutive cylinder openings
data['time_elapsed'] = data['s_step510_ms'].diff().fillna(0)

# create rolling statistics with a window size of 10 seconds
window_size = 10000
data['step510_mean'] = data['step510'].rolling(window_size).mean().fillna(0)
data['step510_max'] = data['step510'].rolling(window_size).max().fillna(0)
data['step510_var'] = data['step510'].rolling(window_size).var().fillna(0)

# remove outliers using the Z-score method
data['step510_zscore'] = zscore(data['step510'])
data = data[(data['step510_zscore'] > -3) & (data['step510_zscore'] < 3)]

# create a line plot of the trends
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(data['s_step510_ms'], data['step510'], label='Original Data', alpha=0.5)
ax.plot(data['s_step510_ms'], data['step510_mean'], label='Rolling Mean ({} seconds)'.format(window_size/1000), linewidth=2)
ax.plot(data['s_step510_ms'], data['step510_max'], label='Rolling Max ({} seconds)'.format(window_size/1000), linewidth=2)
ax.set_xlabel('Time (log scale)')
ax.set_ylabel('step510')
ax.set_yscale('log')
ax.set_xscale('log')
ax.legend(loc='best')
plt.show()


## p value

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import yeojohnson
import statsmodels.api as sm

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
important_features = corr_matrix.index[abs(corr_matrix['step510']) >= 0.3]

# split the dataset into training and testing sets using the selected features
X_train, X_test, y_train, y_test = train_test_split(data[important_features], data['step510'], test_size=0.2, random_state=42)

# fit a linear regression model on the training set
lr = LinearRegression().fit(X_train, y_train)

# transform the residuals using the Yeo-Johnson transformation
y_pred = lr.predict(X_test)
residuals = y_test - y_pred
residuals_yj, lambda_ = yeojohnson(residuals)

# fit a linear regression model on the transformed residuals
X_test_yj = sm.add_constant(X_test)
lr_res = sm.OLS(residuals_yj, X_test_yj).fit()

# evaluate the model on the testing set
y_pred_res_yj = lr_res.predict(X_test_yj)
mse_res_yj = mean_squared_error(residuals_yj, y_pred_res_yj)
r_squared_res_yj = lr_res.rsquared
p_value = lr_res.f_pvalue

# print the results
print('Selected Features:', important_features)
print('Coefficients:', lr_res.params)
print('Intercept:', lr_res.params[0])
print('Mean Squared Error (Yeo-Johnson transformed residuals):', mse_res_yj)
print('R-squared (Yeo-Johnson transformed residuals):', r_squared_res_yj)
print('P-value:', p_value)

# plot the linear trend
plt.plot(data['s_step510_ms'], data['step510'])
plt.show()


### different approch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import yeojohnson, ttest_ind
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import statsmodels.api as sm
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# load the dataset
data = pd.read_csv(r"C:\Users\Q610267\Downloads\exd_download\Test-Interview\Test-Interview\30_2022_01_Greifer_Schritt510_Anforderung_bis_Endschalter.csv")


# preprocess the data by taking the log of the s_step510_ms column
data['s_step510_ms'] = np.log(data['s_step510_ms'])

# create a new feature that represents the time elapsed between consecutive cylinder openings
data['time_elapsed'] = data['s_step510_ms'].diff().fillna(0)

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

# perform t-test and ANOVA to determine the significance of the selected features
t_test = pd.Series()
for feature in data.columns[:-1]:
    t, p = ttest_ind(data.loc[data['step510'] == 0, feature],
                     data.loc[data['step510'] == 1, feature],
                     equal_var=False)
    t_test[feature] = p

t_test.sort_values(inplace=True)
print('T-Test Results:')
print(t_test)

anova = pd.Series(f_classif(data.iloc[:,:-1], data['step510'])[1])
anova.index = data.columns[:-1]
anova.sort_values(inplace=True)
print('ANOVA Results:')
print(anova)

# perform feature selection using recursive feature elimination
lr = LinearRegression()
rfe = RFE(lr, n_features_to_select=5)
X_train, X_test, y_train, y_test = train_test_split(data.iloc[:,:-1], data['step510'], test_size=0.2, random_state=42)
rfe.fit(X_train, y_train)
rfe_selected = pd.Series(rfe.support_, index=X_train.columns)
rfe_selected_features = rfe_selected[rfe_selected==True].index.tolist()
print('Recursive Feature Elimination Results:')
print(rfe_selected_features)

# perform feature selection using feature importance ranking
rf = RandomForestRegressor()
pipe = make_pipeline(StandardScaler(), SelectFromModel(estimator=rf), lr)
pipe.fit(X_train, y_train)
importance = lr.coef_

# evaluate the model on the testing set
X_test_rfe = X


###

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import yeojohnson
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm

# load the dataset
data = pd.read_csv(r"C:\Users\Q610267\Downloads\exd_download\Test-Interview\Test-Interview\30_2022_01_Greifer_Schritt510_Anforderung_bis_Endschalter.csv")


# preprocess the data by taking the log of the s_step510_ms column
data['s_step510_ms'] = np.log(data['s_step510_ms'])

# create a new feature that represents the time elapsed between consecutive cylinder openings
data['time_elapsed'] = data['s_step510_ms'].diff().fillna(0)

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

# perform t-test and ANOVA to determine the significance of the selected features
t_test = pd.Series()
for feature in data.columns[:-1]:
    t, p = stats.ttest_ind(data.loc[data['step510'] == 0, feature],
                           data.loc[data['step510'] == 1, feature],
                           equal_var=False)
    t_test[feature] = p
t_test.sort_values(inplace=True)
print('T-Test Results:')
print(t_test)

anova = pd.Series(f_classif(data.iloc[:,:-1], data['step510'])[1])
anova.index = data.columns[:-1]
anova.sort_values(inplace=True)
print('ANOVA Results:')
print(anova)

# select features with p-value > 0.5
selected_features = anova[anova > 0.5].index.tolist()
print('Selected Features:')
print(selected_features)

# create X and y for modeling
X = data[selected_features]
y = data['step510']

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# fit a linear regression model
lr = LinearRegression()
lr.fit(X_train, y_train)

# predict on test data
y_pred = lr.predict(X_test)

# calculate mean squared error
mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', mse)

# plot residual, normal distribution, and linear trend
residual = y_test - y_pred
sns.histplot(residual, kde=True)
plt.title('Residual Normal Distribution')
plt.show()

sns.scatterplot(y_test, y_pred)
sns.lineplot






#### finding better p value

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import boxcox
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the data
data = pd.read_csv(r"C:\Users\Q610267\Downloads\exd_download\Test-Interview\Test-Interview\30_2022_01_Greifer_Schritt510_Anforderung_bis_Endschalter.csv")

# Create a new feature that represents the time elapsed between consecutive cylinder openings
data['time_elapsed'] = data['s_step510_ms'].diff().fillna(0)

# Box-Cox transformation on the target feature
data['step510'], _ = boxcox(data['step510'])

# Fit a linear regression model
X = data['time_elapsed'].values.reshape(-1, 1)
y = data['step510']
model = LinearRegression().fit(X, y)

# Predict the target feature using the fitted model
y_pred = model.predict(X)

# Calculate the R-squared value
r_squared = model.score(X, y)

# Calculate the mean squared error
mse = mean_squared_error(y, y_pred)

# Plot the original data and the predicted values
plt.scatter(data['time_elapsed'], y, label='Original Data')
plt.plot(data['time_elapsed'], y_pred, color='red', label='Linear Model')
plt.xlabel('Time Elapsed (s)')
plt.ylabel('Transformed Target Feature')
plt.legend()
plt.show()

# Plot the residuals
residuals = y - y_pred
plt.hist(residuals)
plt.title('Histogram of Residuals')
plt.xlabel('Residual')
plt.ylabel('Frequency')
plt.show()

# Test for normality of residuals using Shapiro-Wilk test
from scipy.stats import shapiro

_, p_value = shapiro(residuals)
if p_value > 0.05:
    print('Residuals are normally distributed')
else:
    print('Residuals are not normally distributed')



####

import pandas as pd
import numpy as np
from scipy.stats import shapiro
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv(r"C:\Users\Q610267\Downloads\exd_download\Test-Interview\Test-Interview\30_2022_01_Greifer_Schritt510_Anforderung_bis_Endschalter.csv")

# Compute the time elapsed between consecutive cylinder openings
data['time_elapsed'] = data['s_step510_ms'].diff().fillna(0)

# Compute the time in ms per process (= time for 1 closing)
data['time_per_process'] = data['step510'] / data['time_elapsed']

# Check the distribution of the residuals
residuals = data['time_per_process'] - data['time_per_process'].mean()
shapiro_results = shapiro(residuals)
print(f"Shapiro test results: p-value={shapiro_results.pvalue:.2f}")

# Try different methods to preprocess the data and check for a linear trend

# Log transformation
data['log_time_per_process'] = np.log(data['time_per_process'])
residuals = data['log_time_per_process'] - data['log_time_per_process'].mean()
shapiro_results = shapiro(residuals)
print(f"Shapiro test results after log transformation: p-value={shapiro_results.pvalue:.2f}")
plt.scatter(data.index, data['log_time_per_process'])
plt.show()
lr = LinearRegression()
lr.fit(data.index.values.reshape(-1, 1), data['log_time_per_process'])
plt.plot(data.index, lr.predict(data.index.values.reshape(-1, 1)), color='red')
plt.show()

# Square root transformation
data['sqrt_time_per_process'] = np.sqrt(data['time_per_process'])
residuals = data['sqrt_time_per_process'] - data['sqrt_time_per_process'].mean()
shapiro_results = shapiro(residuals)
print(f"Shapiro test results after square root transformation: p-value={shapiro_results.pvalue:.2f}")
plt.scatter(data.index, data['sqrt_time_per_process'])
plt.show()
lr = LinearRegression()
lr.fit(data.index.values.reshape(-1, 1), data['sqrt_time_per_process'])
plt.plot(data.index, lr.predict(data.index.values.reshape(-1, 1)), color='red')
plt.show()

# Box-Cox transformation
from scipy.stats import boxcox
data['bc_time_per_process'], _ = boxcox(data['time_per_process'])
residuals = data['bc_time_per_process'] - data['bc_time_per_process'].mean()
shapiro_results = shapiro(residuals)
print(f"Shapiro test results after Box-Cox transformation: p-value={shapiro_results.pvalue:.2f}")
plt.scatter(data.index, data['bc_time_per_process'])
plt.show()
lr = LinearRegression()
lr.fit(data.index.values.reshape(-1, 1), data['bc_time_per_process'])
plt.plot(data.index, lr.predict(data.index.values.reshape(-1, 1)), color='red')
plt.show()

## plot

# calculate time elapsed between consecutive cylinder openings
data['time_elapsed'] = data['s_step510_ms'].diff().fillna(0)

# apply log transform to 'step510' column
data['step510_log'] = np.log(data['step510'])


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import normaltest, linregress
from sklearn.linear_model import LinearRegression
from statsmodels.graphics.gofplots import qqplot

# plot histogram and QQ plot of transformed data
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
sns.histplot(data['step510_log'], kde=True, ax=ax[0])
ax[0].set_title('Histogram of Log Transformed Data')
sns.qqplot(data['step510_log'], line='s', ax=ax[1])
ax[1].set_title('QQ Plot of Log Transformed Data')

# perform linear regression on transformed data
x = data['time_elapsed']
y = data['step510_log']
slope, intercept, r_value, p_value, std_err = linregress(x, y)

# plot scatterplot with linear fit
fig, ax = plt.subplots()
sns.scatterplot(x=x, y=y, ax=ax)
sns.lineplot(x=x, y=intercept + slope*x, color='red', ax=ax)
ax.set_xlabel('Time Elapsed (ms)')
ax.set_ylabel('Log of Time for 1 Closing (ms)')
ax.set_title(f'Linear Fit: y = {intercept:.2f} + {slope:.2f}x, p-value={p_value:.4f}')

# back-transform predicted values and calculate residuals
predicted_log_values = intercept + slope*x
predicted_values = np.exp(predicted_log_values)
residuals = data['step510'] - predicted_values

# plot histogram and QQ plot of residuals
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
sns.histplot(residuals, kde=True, ax=ax[0])
ax[0].set_title('Histogram of Residuals')
normaltest_result = normaltest(residuals)
ax[0].text(0.05, 0.95, f'p-value: {normaltest_result.pvalue:.4f}', transform=ax[0].transAxes, fontsize=10,
            verticalalignment='top')
sns.qqplot(residuals, line='s', ax=ax[1])
ax[1].set_title('QQ Plot of Residuals')
plt.show()



#### other test for p value

import numpy as np
import pandas as pd
import mlxtend
import matplotlib.pyplot as plt
from scipy.stats import shapiro, zscore, norm
from scipy.stats import shapiro
from mlxtend.evaluate import permutation_test

data = pd.read_csv(r"C:\Users\Q610267\Downloads\exd_download\Test-Interview\Test-Interview\30_2022_01_Greifer_Schritt510_Anforderung_bis_Endschalter.csv")

# calculate the Z-score of the data
z = np.abs(zscore(data['step510']))

# calculate the p-value using the Z-test
p_value = 2 * norm.sf(z)

# print the results
if p_value < 0.05:
    print("The data is not normally distributed.")
else:
    print("The data is normally distributed.")

# plot a histogram and a Q-Q plot of the data
plt.hist(data['step510'], bins=50)
plt.show()
stats.probplot(data['step510'], dist="norm", plot=plt)
plt.show()

# run a permutation test to test for normality
p_value = permutation_test(data['step510'], np.random.normal, method='approximate', num_rounds=10000, seed=42)

# print the results
if p_value < 0.05:
    print("The data is not normally distributed.")
else:
    print("The data is normally distributed.")

# plot a histogram and a Q-Q plot of the data
plt.hist(data['step510'], bins=50)
plt.show()
stats.probplot(data['step510'], dist="norm", plot=plt)
plt.show()


#### error




#####


import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.stats import normaltest
from scipy.stats import ttest_ind



# load the dataset
data = pd.read_csv(r"C:\Users\Q610267\Downloads\exd_download\Test-Interview\Test-Interview\30_2022_01_Greifer_Schritt510_Anforderung_bis_Endschalter.csv")


# Convert s_step510_ms to datetime
data['s_step510_ms'] = pd.to_datetime(data['s_step510_ms'], unit='ms')

# Calculate the time elapsed between consecutive cylinder openings
data['time_elapsed'] = (data['s_step510_ms'] - data['s_step510_ms'].shift()).fillna(0)

# Convert time_elapsed to seconds
data['time_elapsed'] = data['time_elapsed'].dt.total_seconds()

# Create a scatter plot of time_elapsed vs step510
plt.scatter(data['time_elapsed'], data['step510'])
plt.xlabel('Time Elapsed (s)')
plt.ylabel('Step510')
plt.title('Scatter Plot of Time Elapsed vs Step510')
plt.show()

# Check for normality of residuals
model = sm.OLS(data['step510'], sm.add_constant(data['time_elapsed']))
results = model.fit()
residuals = results.resid
p_value = normaltest(residuals).pvalue

if p_value < 0.05:
    print('Residuals are not normally distributed.')
else:
    print('Residuals are normally distributed.')

# Perform t-test to determine if the slope of the regression line is significantly different from zero
t_stat, p_value = ttest_ind(data['step510'], data['time_elapsed'])

if p_value < 0.05:
    print('Slope of regression line is significantly different from zero.')
else:
    print('Slope of regression line is not significantly different from zero.')

# Plot the regression line
plt.scatter(data['time_elapsed'], data['step510'])
plt.plot(data['time_elapsed'], results.predict(), color='red')
plt.xlabel('Time Elapsed (s)')
plt.ylabel('Step510')
plt.title('Scatter Plot of Time Elapsed vs Step510 with Regression Line')
plt.show()
