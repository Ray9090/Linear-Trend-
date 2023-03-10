

def outlier_treatment(data, column):
    data_col = data[column]
    sorted(data_col)
    Q1, Q3 = np.percentile(data_col, [10, 90])
    # print(Q1, Q3)
    IQR = Q3 - Q1
    lower_range = Q1 - (2 * IQR)
    upper_range = Q3 + (2 * IQR)
    # print(lower_range)
    # print(upper_range)

    return data[(data[column] > lower_range) & (data[column] < upper_range)]


def li(s):
    s = s.dropna(how='any')
    X = range(len(s))
    y = s
    fit_reg = np.polyfit(X, y, 1)
    fit_log = np.flipud(fit_reg)
    y_pred = [fit_log[0] + fit_log[1] * i for i in range(len(y))]
    y = y.tolist()
    res = [y[x] - y_pred[x] for x in range(len(y))]
    test = sms.jarque_bera(res)
    pvalue = test[1]
    r2 = r2_score(y, y_pred)
    wear = y_pred[-1] - y_pred[0]
    return pvalue, r2, wear
    # plt.plot(res)
    # plt.show()


if __name__ == "__main__":
    df = pd.read_csv(r"C:\Users\simpc\Desktop\Greifer\Desktop\Station30\30_2022_01\30_2022_01_Greifer_Schritt510_Anforderung_bis_Endschalter.csv")
    clean_df = outlier_treatment(df, 'step510')
    clean_df['n'] = range(len(clean_df))
    step510 = pd.DataFrame(clean_df['step510'].copy())
    for w in trange(1000, 300000, 100):  # [135000]
        for i in [4006]:  # 1, 10000, 100trange(4000, 4200, 1)
            step510['SMA'] = step510['step510'][i:].rolling(w, min_periods=1).mean()
            p, r, wear = li(step510['SMA'])
            if p > 0.01:
                print(f'{w} - {i} SMA - Jarque Bera Chi-squared(2) p-value: ', p, ' R^2', r, ' Wear', wear)
            # The Cumulative Moving Average
            step510['CMA'] = step510['SMA'].expanding().mean()
            p, r, wear = li(step510['CMA'])
            if p > 0.01:
                print(f'{w} - {i} CMA - Jarque Bera Chi-squared(2) p-value: ', p, ' R^2', r, ' Wear', wear)
            # smoothing factor - 0.1
            step510['EMA_0.1'] = step510['SMA'].ewm(alpha=0.1, adjust=False).mean()
            p, r, wear = li(step510['EMA_0.1'])
            if p > 0.01:
                print(f'{w} - {i} EMA_0.1 - Jarque Bera Chi-squared(2) p-value: ', p, ' R^2', r, ' Wear', wear)
            # smoothing factor - 0.3
            step510['EMA_0.3'] = step510['SMA'].ewm(alpha=0.3, adjust=False).mean()
            p, r, wear = li(step510['EMA_0.3'])
            if p > 0.01:
                print(f'{w} - {i} EMA_0.3 - Jarque Bera Chi-squared(2) p-value: ', p, ' R^2', r, ' Wear', wear)

#
# df = pd.read_csv(r"C:\Users\simpc\Desktop\Greifer\Desktop\Station30\30_2022_01\30_2022_01_Greifer_Schritt510_Anforderung_bis_Endschalter.csv")
# clean_df = outlier_treatment(df, 'step510')
# clean_df['n'] = range(len(clean_df))
# step510 = pd.DataFrame(clean_df['step510'].copy())
# step510['SMA'] = step510['step510'][4101:].rolling(135000, min_periods=1).mean()
# step510['CMA'] = step510['SMA'].expanding().mean()
# step510['EMA_0.1'] = step510['SMA'].ewm(alpha=0.1, adjust=False).mean()
# step510['EMA_0.3'] = step510['SMA'].ewm(alpha=0.3, adjust=False).mean()
# s = step510['SMA'].copy()
# s = s.dropna(how='any')
# X = range(len(s))
# y = s
# fit_reg = np.polyfit(X, y, 1)
# fit_log = np.flipud(fit_reg)
# y_pred = [fit_log[0] + fit_log[1] * i for i in range(len(y))]
# y = y.tolist()
# res = [y[x] - y_pred[x] for x in range(len(y))]
# test = sms.jarque_bera(res)
# print('Jarque Bera Chi-squared(2) p-value: ', test[1], ' R^2', r2_score(y, y_pred))
# plt.plot(res)
# plt.plot(y)
# print(y_pred[-1] - y_pred[0])
# import matplotlib as mpl
# # parameters for graphic
# cm = 1 / 2.54                   # convert inch to cm
# dpi = 500                       # dpi of graphic
# font = {'family': 'Open Sans',
#         'size': 11}             # text in graphic, font and size
# mpl.rc('font', **font)
# datapoints_color = '#133155'    # color of datapoints: dark blue
# reg_line_color = 'r'            # color of regression line: red
#
# # axis label
# # Schließen Anforderung bis Endschalter
# step510_ledend_label = 'Schritt 510 Anforderung bis Endschalter'
# step510_x_axis_time_label = 'Zeit [d]'
# step510_x_axis_ncut_label = 'Schnittanzahl [-]'
# step510_y_axis_label = 'Schließzeit des Zylinders [ms]'
#
# # folder path for graphic
# fig_save_dir = 'C:/Users/simpc/Desktop/Greifer/Desktop/Station30'
# fig33, ax33 = plt.subplots(1, 1, figsize=(16 * cm, 7 * cm))
# ax33.plot(range(len(s)), s, color=datapoints_color,
#           label='Gleitender Durchschnitt')
# ax33.plot(range(len(s)), y_pred, color=reg_line_color,
#           label='Lineare Regressionsgerade')
# ax33.set_ylabel(step510_y_axis_label, style='italic')
# ax33.set_xlabel(step510_x_axis_ncut_label, style='italic')
# ax33_x_day_labels = ax33.get_xticks().tolist()
# ax33.legend(loc='upper left')
# plt.grid()
# plt.tight_layout()
# plt.savefig(r'C:\Users\simpc\Desktop\Greifer\Desktop\Station30\30_2022_01\window135000_from4101_p01_wear27_periods1.svg',
#     dpi=dpi)
# plt.close()
#
# from statsmodels.tsa.seasonal import seasonal_decompose
# from matplotlib import pyplot
# clean_df.index = pd.date_range('2022-01-01 00:00:00', periods=len(clean_df), freq='ms')
# result = seasonal_decompose(clean_df['step510'], model='multiplicative', period=1)
# result.plot()
# pyplot.title('Decomposition')
# pyplot.show()

# plt.style.use('seaborn')
#
# # line plot - the yearly average air temperature in Barcelona
# step510.plot(color='green', linewidth=3, figsize=(12,6))
#
# # modify ticks size
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)
# plt.legend('')
#
# # title and labels
# plt.title('The time for one process (closing)', fontsize=20)
# plt.xlabel('Schnittanzahl [-]', fontsize=16)
# plt.ylabel('Schließzeit des Zylinders [ms]', fontsize=16)

# The Simple Moving Average
# step510['SMA_1000'] = step510['step510'].rolling(1000).mean()
# step510['SMA_1000_periods'] = step510['step510'].rolling(1000, min_periods=1).mean()
#
# # the simple moving average over a period of 20 year
# step510['SMA_5000'] = step510['step510'].rolling(5000).mean()
# step510['SMA_5000_periods'] = step510['step510'].rolling(5000, min_periods=1).mean()
#
# window_size = 1000
# step510['SMA'] = step510['step510'][10000:].rolling(window_size).mean()
# colors for the line plot
# colors = ['green', 'red', 'darkred', 'violet', 'purple']
#
# # line plot - the yearly average air temperature in Barcelona
# step510.plot(color=colors, linewidth=3, figsize=(12, 6))
#
# # modify ticks size
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)
# plt.legend(labels =['Schließzeit', '1000 SMA', '1000 SMA fillna', '5000 SMA', '5000 SMA fillna'], fontsize=14)
#
# # title and labels
# plt.title('The time for one process (closing)', fontsize=20)
# plt.xlabel('Schnittanzahl [-]', fontsize=16)
# plt.ylabel('Schließzeit des Zylinders [ms]', fontsize=16)

# The Cumulative Moving Average
# step510['CMA'] = step510['SMA'].expanding().mean()
# li(step510['CMA'])
# colors for the line plot
# colors = ['green', 'orange']
#
# # line plot - the yearly average air temperature in Barcelona
# step510[['SMA', 'CMA']].plot(color=colors, linewidth=3, figsize=(12, 6))
#
# # modify ticks size
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)
# plt.legend(labels =['step510', 'CMA'], fontsize=14)
#
# # title and labels
# plt.title('The time for one process (closing)', fontsize=20)
# plt.xlabel('Schnittanzahl [-]', fontsize=16)
# plt.ylabel('Schließzeit des Zylinders [ms]', fontsize=16)

# The Exponential Moving average
# cumulative moving average
# ambient air temperature
# smoothing factor - 0.1
# step510['EMA_0.1'] = step510['SMA'].ewm(alpha=0.1, adjust=False).mean()
# li(step510['EMA_0.1'])
# smoothing factor - 0.3
# step510['EMA_0.3'] = step510['SMA'].ewm(alpha=0.3, adjust=False).mean()
# li(step510['EMA_0.3'])
# colors for the line plot
# colors = ['green', 'orchid', 'orange']
#
# # line plot - the yearly average air temperature in Barcelona
# step510[['SMA', 'EMA_0.1', 'EMA_0.3']].plot(color=colors, linewidth=3, figsize=(12, 6), alpha=0.8)
#
# # modify ticks size
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)
# plt.legend(labels=['Average air temperature', 'EMA - alpha=0.1', 'EMA - alpha=0.3'], fontsize=14)
#
# # title and labels
# plt.title('The time for one process (closing)', fontsize=20)
# plt.xlabel('Schnittanzahl [-]', fontsize=16)
# plt.ylabel('Schließzeit des Zylinders [ms]', fontsize=16)
#
# # smoothing factor and number of data points
# ALPHA = 0.3
# N = 15
#
# # weights - simple moving average
# w_sma = np.repeat(1/N, N)
#
# # weights - exponential moving average alpha=0.3 adjust=False
# w_ema = [(1-ALPHA)**i if i==N-1 else ALPHA*(1-ALPHA)**i for i in range(N)]
#
# # store the values in a data frame
# pd.DataFrame({'w_sma': w_sma, 'w_ema': w_ema}).plot(kind='bar', figsize=(10,6))
#
# # modify ticks size and labels
# plt.xticks([])
# plt.yticks(fontsize=14)
# plt.legend(labels=['Simple moving average', 'Exponential moving average (α=0.3)'], fontsize=14)
#
# # title and labels
# plt.title('Moving Average Weights', fontsize=20)
# plt.ylabel('Weights', fontsize=16)

