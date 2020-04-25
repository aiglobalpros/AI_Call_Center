import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import datetime as dt

from dateutil.relativedelta import relativedelta
from scipy import stats

def plot_train_val_test(X, y, idx_train, idx_val, idx_test, idx_out):
    x_train = X.loc[idx_train].values
    y_train = y.loc[idx_train].values
    x_val   = X.loc[idx_val].values
    y_val   = y.loc[idx_val].values
    x_test  = X.loc[idx_test].values
    y_test  = y.loc[idx_test].values
    x_out  = X.loc[idx_out].values
    y_out  = y.loc[idx_out].values

    fig, ax = plt.subplots(figsize=(50,10))
    ax.plot(x_train, y_train, label='Train', color='green')
    ax.plot(x_val, y_val, label='Validate', color='yellowgreen', marker='+')
    ax.plot(x_test, y_test, label='Test', color='red', marker='.')    
    ax.plot(x_out, y_out, label='Out of time', color='grey', linestyle='--')  

    ax.xaxis.set_major_locator(matplotlib.dates.YearLocator())
    ax.xaxis.set_minor_locator(matplotlib.dates.MonthLocator(list(range(1,13))))

    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b\n%Y"))
    ax.xaxis.set_minor_formatter(matplotlib.dates.DateFormatter("%b"))
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
    plt.xlabel('Months')
    plt.ylabel('Number of CCT tickets')
    plt.legend(loc='upper left')
    plt.tight_layout()

    plt.show()
    
    
def load_dataset(dataset_path):
    perfo_df = pd.read_csv(dataset_path + 'performance_centre_appels_sept2017_mars2020.csv', encoding="ISO-8859-1", low_memory=False)
    CCT_df = pd.read_csv(dataset_path + 'IncidentsCTT20170930_2.csv', encoding="ISO-8859-1", low_memory=False)

    print("Dataframe memory usage: %.2f MB" % (CCT_df.memory_usage().sum()/(1024*1024)))
    
    return CCT_df, perfo_df


def cast_CCT_features(df):
    dates_list = ['Submit_Date', 'closed_date'] #['Submit_Date','Reported_Date','Required_Resolution_DateTime','Responded_Date','Last_Resolved_Date','closed_date']

    for col in dates_list:
        print('Converting %s' % col)
        df[col] = pd.to_datetime(df[col].fillna('1900-01-01 00:00:00.000'))
        df[col + '_day'] = df[col].apply(lambda x: x.strftime('%Y-%m-%d')) # To group by day
        df[col + '_with_hour'] =df[col].apply(lambda x: x.strftime('%Y-%m-%d %H:00')) # To group by day

    for col in df.select_dtypes('object').columns:
        df[col] = df[col].astype('category')
        
    print("Dataframe memory usage: %.2f MB" % (df.memory_usage().sum()/(1024*1024)))
    
    return df


def cast_mep_dates(df):
    col_list = ['Submit_Date', 'ScheduledStartDate', 'ScheduledEndDate',
                'ActualStartDate', 'ActualEndDate','mep_date']
    
    for col in col_list:
            df[col] = pd.to_datetime(df[col].fillna('1900-01-01 00:00:00.000'))
            df[col + '_day'] = df[col].apply(lambda x: x.strftime('%Y-%m-%d')) # To group by day
            df[col + '_day'] = pd.to_datetime(df[col + '_day'], format=('%Y-%m-%d'))
        
    return df


def plot_label(x, y):
    fig, ax = plt.subplots(figsize=(100,10))
    ax.plot(x, y)

    ax.xaxis.set_major_locator(matplotlib.dates.YearLocator())
    ax.xaxis.set_minor_locator(matplotlib.dates.MonthLocator(list(range(1,13))))

    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b\n%Y"))
    ax.xaxis.set_minor_formatter(matplotlib.dates.DateFormatter("%b"))
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

    plt.show()
    

def count_mep(x, level):
    """ 
        Conditional aggregation function
    """
    return (x==level).sum()

def build_mep_features(dataset_path):
    mep_df = pd.read_csv(dataset_path + 'MEP_Videotron.csv', encoding="ISO-8859-1", low_memory=False)

    mep_df = mep_df.dropna(subset=['ScheduledStartDate'])
    mep_df['mep_date'] = mep_df['ActualStartDate']
    mep_df['mep_started'] = (~mep_df['ActualStartDate'].isna())
    mep_df.loc[mep_df['ActualStartDate'].isna(), 'mep_date'] = mep_df.loc[mep_df['ActualStartDate'].isna(), 'ScheduledStartDate']

    mep_df = cast_mep_dates(mep_df)

    start_dt = dt.datetime.strptime('2017-10-01', '%Y-%m-%d')
    cond = (mep_df['mep_date'] >= start_dt)

    agg_df = mep_df.loc[cond].groupby('mep_date_day')['mep_date_day'].count()
    agg_df = pd.DataFrame(agg_df.values, index=agg_df.index, columns=['total_mep_cnt'])
    agg_df = build_lag_features(agg_df, 'total_mep_cnt')

    for col in ['Risk_Level','ChangeType', 'Urgency', 'Impact',	'Priority']:
        for val in mep_df[col].value_counts().index:    
            col_agg = f'{col}_{val}_cnt'
            cond_risk = (cond & (mep_df[col] == val))
            agg_df[col_agg] = mep_df.loc[cond].groupby('mep_date_day')[col].agg(lambda x: count_mep(x, val))

            # build lag features
            agg_df = build_lag_features(agg_df, col_agg)
            
    return agg_df.reset_index(drop=False)
    
    
def build_date_related_features(df):
    df['month']            = df['Submit_Date_day'].apply(lambda x : x.month)
    df['week']             = df['Submit_Date_day'].apply(lambda x : x.week)
    df['is_month_start']   = df['Submit_Date_day'].apply(lambda x : x.is_month_start).astype('int')
    df['is_month_end']     = df['Submit_Date_day'].apply(lambda x : x.is_month_end).astype('int')
    df['is_quarter_start'] = df['Submit_Date_day'].apply(lambda x : x.is_quarter_start).astype('int')
    df['is_quarter_end']   = df['Submit_Date_day'].apply(lambda x : x.is_quarter_end).astype('int')
    df['is_year_start']    = df['Submit_Date_day'].apply(lambda x : x.is_year_start).astype('int')
    df['is_year_end']      = df['Submit_Date_day'].apply(lambda x : x.is_year_end).astype('int')

    day_name_df = df['Submit_Date_day'].apply(lambda x : x.day_name()) # One hot encode this feature
    df = pd.concat([df, pd.get_dummies(day_name_df)], axis=1)
    
    return df


def days_to_event(date, event_dates):
    delta_list = []
    for event_date in event_dates:
        delta = event_date - date        
        # Ignore negative delta days, the event is past
        if delta.days >= 0:
            delta_list.append(delta.days)
           
    if len(delta_list) != 0:
        return_value = np.min(delta_list)
    else:
        return_value = np.nan
    
    # Return the number of days until the next event
    return return_value


def build_event_related_features(df):
    black_friday_dates_str = ["2017-11-24", "2018-11-23", "2019-11-29", "2020-11-27"]
    back_to_school_dates_str = ["2017-08-28", "2018-08-27", "2019-08-26", "2020-08-24"]
    moving_dates_str = ["2017-07-01", "2018-07-01", "2019-07-01", "2020-07-01"]
    christmas_dates_str = ["2017-12-25", "2018-12-25", "2019-12-25", "2020-12-25"]
    new_year_dates_str = ["2018-01-01", "2019-01-01", "2020-01-01", "2021-01-01"]
    first_worked_monday_str = ["2018-01-08", "2019-01-07", "2020-01-06"]

    black_friday_dates = []
    cyber_monday_dates = []
    back_to_school_dates = []
    moving_dates = []
    christmas_dates = []
    new_year_dates = []
    first_worked_monday_dates = []

    for date_str in black_friday_dates_str:
        black_friday_dates.append(dt.datetime.strptime(date_str, "%Y-%m-%d"))
        cyber_monday_dates.append(dt.datetime.strptime(date_str, "%Y-%m-%d") + relativedelta(days=3))

    for date_str in back_to_school_dates_str:
        back_to_school_dates.append(dt.datetime.strptime(date_str, "%Y-%m-%d"))

    for date_str in moving_dates_str:
        moving_dates.append(dt.datetime.strptime(date_str, "%Y-%m-%d"))

    for date_str in christmas_dates_str:
        christmas_dates.append(dt.datetime.strptime(date_str, "%Y-%m-%d"))

    for date_str in new_year_dates_str:
        new_year_dates.append(dt.datetime.strptime(date_str, "%Y-%m-%d"))
        
    for date_str in first_worked_monday_str:
        first_worked_monday_dates.append(dt.datetime.strptime(date_str, "%Y-%m-%d"))

    df['days_to_black_friday']    = df['Submit_Date_day'].apply(lambda x: days_to_event(x, black_friday_dates))
    df['days_to_cyber_monday']    = df['Submit_Date_day'].apply(lambda x: days_to_event(x, cyber_monday_dates))
    df['days_to_christmas']       = df['Submit_Date_day'].apply(lambda x: days_to_event(x, christmas_dates))
    df['days_to_new_year']        = df['Submit_Date_day'].apply(lambda x: days_to_event(x, new_year_dates))
    df['days_to_moving_day']      = df['Submit_Date_day'].apply(lambda x: days_to_event(x, moving_dates))
    df['days_to_back_to_school']  = df['Submit_Date_day'].apply(lambda x: days_to_event(x, back_to_school_dates))
    df['first_worked_monday']     = df['Submit_Date_day'].apply(lambda x: days_to_event(x, first_worked_monday_dates))
    df['is_first_worked_monday']  = 1*(df['first_worked_monday'] == 0)
    df = df.drop(columns=['first_worked_monday'])
    
    return df


def build_lag_features(df, col):
    df[col + ' lag_1']  = df[col].shift(1)  # use the number of tickets 1 day prior the current date
    df[col + ' lag_2']  = df[col].shift(2)  # use the number of tickets 2 days prior the current date
    df[col + ' lag_3']  = df[col].shift(3)  # use the number of tickets 3 days prior the current date
    df[col + ' lag_4']  = df[col].shift(4)  # use the number of tickets 4 days prior the current date
    df[col + ' lag_5']  = df[col].shift(5)  # use the number of tickets 5 days prior the current date
    df[col + ' lag_6']  = df[col].shift(6)  # use the number of tickets 6 days prior the current date
    df[col + ' lag_7']  = df[col].shift(7)  # use the number of tickets 7 days prior the current date
    df[col + ' lag_14'] = df[col].shift(14) # use the number of tickets 14 days prior the current date
    df[col + ' lag_21'] = df[col].shift(21) # use the number of tickets 21 days prior the current date
    df[col + ' lag_28'] = df[col].shift(28) # use the number of tickets 28 days prior the current date 
    
    
    return df


def build_perfo_features(dataset_path):
    perfo_df = pd.read_csv(dataset_path + 'performance_centre_appels_sept2017_mars2020.csv', encoding="ISO-8859-1", low_memory=False)
    perfo_lag_df = pd.DataFrame(pd.to_datetime(perfo_df['Date']), columns=['Date'])
    feature_list = perfo_df.drop(columns=['Jour','Date']).columns.values
    
    for col in feature_list:
        perfo_lag_df[col + '_lag1'] = perfo_df[col].shift(1)
        perfo_lag_df[col + '_lag2'] = perfo_df[col].shift(2)
        perfo_lag_df[col + '_lag3'] = perfo_df[col].shift(3)
        perfo_lag_df[col + '_lag4'] = perfo_df[col].shift(4)
        perfo_lag_df[col + '_lag5'] = perfo_df[col].shift(5)
        perfo_lag_df[col + '_lag6'] = perfo_df[col].shift(6)
        perfo_lag_df[col + '_lag7'] = perfo_df[col].shift(7)
        perfo_lag_df[col + '_lag14'] = perfo_df[col].shift(14)
        perfo_lag_df[col + '_lag21'] = perfo_df[col].shift(21)
        perfo_lag_df[col + '_lag28'] = perfo_df[col].shift(28)
        
    return perfo_lag_df


def build_weather_features(dataset_path, min_dt, max_dt, plot=True):
    weather_df = pd.read_csv(dataset_path + 'gov_can_daily_weather.csv', encoding="ISO-8859-1", low_memory=False)
    weather_df['date'] = pd.to_datetime(weather_df['date'].fillna('1900-01-01 00:00:00.000'))
    weather_df = weather_df.sort_values(by='date')
    weather_df = weather_df.replace('X', np.nan)
    weather_df = weather_df.replace(-999, np.nan)
    weather_df['snow_grnd'] = weather_df['snow_grnd'].fillna(0)

    weather_flags = weather_df.select_dtypes('object').columns.values

    weather_df = weather_df.drop(columns=weather_flags)

    # Build some flags
    weather_df['max_temp_above_30_flag'] = 1*(weather_df['max_temp'] > 30)

    weather_df['min_temp_below_minus_10_flag'] = 1*(weather_df['min_temp'] < -10)
    weather_df['min_temp_above_minus_20_flag'] = 1*(weather_df['min_temp'] < -20)

    weather_df['max_gust_above_80_flag'] = 1*(weather_df['spd_max_gust'] > 80)
    weather_df['max_gust_above_100_flag'] = 1*(weather_df['spd_max_gust'] > 100)

    weather_df['rainy_day_above_10_flag'] = 1*(weather_df['total_rain'] >= 10)
    weather_df['rainy_day_above_20_flag'] = 1*(weather_df['total_rain'] >= 20)
    weather_df['snowy_day_flag'] = 1*(weather_df['total_snow'] > 0)
    weather_df['snow_day_above_10_flag'] = 1*(weather_df['total_snow'] >= 10)
    weather_df['snow_day_above_15_flag'] = 1*(weather_df['total_snow'] >= 15)
    weather_df['snow_day_above_20_flag'] = 1*(weather_df['total_snow'] >= 20)
    
    if plot is True:
        plot_weather_feature(weather_df)

    weather_df = weather_df.loc[(weather_df['date'] >= min_dt) & (weather_df['date'] <= max_dt)].drop(columns=['date']) 
    weather_df = weather_df.reset_index(drop=True)
        
    return weather_df


def compute_train_val_test_dates(train_start_dt, val_length, test_length, offset_months):

    train_end_dt  = train_start_dt + relativedelta(months=offset_months) - relativedelta(days=1)
    val_start_dt  = train_end_dt   + relativedelta(days=1)
    val_end_dt    = val_start_dt   + relativedelta(months=val_length) - relativedelta(days=1)
    test_start_dt = val_end_dt     + relativedelta(days=1)
    test_end_dt   = test_start_dt  + relativedelta(months=test_length)
    
    print('Fold %d:' % offset_months)
    print('Train set: from %s to %s' % (train_start_dt, train_end_dt))
    print('Validation set: from %s to %s' % (val_start_dt, val_end_dt))
    print('Test set: from %s to %s\n' % (test_start_dt, test_end_dt))

    return train_end_dt, val_start_dt, val_end_dt, test_start_dt, test_end_dt


def get_train_val_test_dataset(df, train_start_dt, val_length, test_length, offset_months):
    train_end_dt, val_start_dt, val_end_dt, test_start_dt, test_end_dt = compute_train_val_test_dates(train_start_dt, val_length,\
                                                                                                      test_length, offset_months)

    # Apply feature selection here (if needed)
    dataset = df.drop(columns=['date', 'Ticket cnt', 'year-month'])
    labels  = df['Ticket cnt']
    
    # Create a train, validation and test dataset
    X_train = dataset.loc[(df['date'] >= train_start_dt) & (df['date'] <= train_end_dt)]
    X_val   = dataset.loc[(df['date'] >= val_start_dt)   & (df['date'] <= val_end_dt)]
    X_test  = dataset.loc[(df['date'] >= test_start_dt)  & (df['date'] <= test_end_dt)]
    X_out_of_time = dataset.loc[(df['date'] > test_end_dt)]

    y_train = labels.loc[(df['date'] >= train_start_dt) & (df['date'] <= train_end_dt)]
    y_val   = labels.loc[(df['date'] >= val_start_dt)   & (df['date'] <= val_end_dt)] 
    y_test  = labels.loc[(df['date'] >= test_start_dt)  & (df['date'] <= test_end_dt)]
    y_out_of_time = labels.loc[(df['date'] > test_end_dt)]    

    return X_train, X_val, X_test, X_out_of_time, y_train, y_val, y_test, y_out_of_time, labels


def export_results(results, path):
    # results_df = pd.DataFrame(columns=['Datetime','MSE', 'RMSE', 'R2', 'model', 'comments'])
    results_df = pd.read_csv(path + 'results_models_centre_appels.csv', encoding="ISO-8859-1", low_memory=False)
    results_df = results_df.append(results, ignore_index=True)
    results_df.to_csv(path + 'results_models_centre_appels.csv', index=False)
    
    return results_df


def plot_predictions_vs_observations(x, y_obs, y_pred, test_start_dt, test_end_dt):
    fig, ax = plt.subplots(figsize=(100,10))
    ax.plot(x, y_obs, label='observation')
    ax.plot(x, y_pred, label='prediction')

    ax.xaxis.set_major_locator(matplotlib.dates.YearLocator())
    ax.xaxis.set_minor_locator(matplotlib.dates.MonthLocator(list(range(1,13))))

    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b\n%Y"))
    ax.xaxis.set_minor_formatter(matplotlib.dates.DateFormatter("%b"))
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

    ax.set_ylabel('Number of tickets received')


    subtitle = 'Test dates: from {0:s} to {1:s}'.format(test_start_dt.strftime('%Y-%m-%d'), test_end_dt.strftime('%Y-%m-%d'))
    ax.set_title('Observation vs prediction of the number of CCT tickets received\n' + subtitle)

    plt.legend()

    plt.show()
    
    
def plot_pearson_corr(df, list_, plot=True, label='Ticket cnt', threshold=0, alpha=1):
    corr_df = pd.DataFrame([], columns=['feature', 'pearson_r', 'abs_pearson_r', 'p-value'])
    
    for col in list_:
        condition = ((~df[label].isna()) & (~df[col].isna()))

        pearson_r , p_val = stats.pearsonr(df.loc[condition, label], df.loc[condition, col])
        
        if (plot is True) & (abs(pearson_r)>threshold) & (p_val < alpha):
            # Compute rolling window synchrony
            f, ax = plt.subplots(figsize=(8,2.5))

            df.loc[condition, [col, label]].rolling(window=30,center=True).median().plot(ax=ax)
            ax.set(xlabel='Days', ylabel='Pearson r')
            ax.set(title=f"Overall Pearson r = {np.round(pearson_r, 4)}\np-value: {np.round(p_val, 4)}")
            
            plt.show()
        
        corr_df = corr_df.append({'feature':col, 'pearson_r':pearson_r, 'abs_pearson_r':abs(pearson_r), 'p-value': p_val}, ignore_index=True)
        
    return corr_df


def plot_weather_feature(df):
    x = df['date']

    for col in df.columns.values:
        if col not in 'date':
            y = df[col]
            fig, ax = plt.subplots(figsize=(10,5))

            if 'flag' not in col:
                ax.plot(x, y)
            else:
                ax.bar(x, y)

            ax.xaxis.set_major_locator(matplotlib.dates.YearLocator())
            ax.xaxis.set_minor_locator(matplotlib.dates.MonthLocator(list(range(1,13))))

            ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b\n%Y"))
            ax.xaxis.set_minor_formatter(matplotlib.dates.DateFormatter("%b"))
            plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

            plt.title(col)

            plt.show()