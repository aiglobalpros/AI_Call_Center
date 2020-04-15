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
    dates_list = ['Submit_Date'] #['Submit_Date','Reported_Date','Required_Resolution_DateTime','Responded_Date','Last_Resolved_Date','closed_date']

    for col in dates_list:
        print('Converting %s' % col)
        df[col] = pd.to_datetime(df[col].fillna('1900-01-01 00:00:00.000'))
        df[col + '_day'] = df[col].apply(lambda x: x.strftime('%Y-%m-%d')) # To group by day
        df[col + '_with_hour'] =df[col].apply(lambda x: x.strftime('%Y-%m-%d %H:00')) # To group by day

    for col in df.select_dtypes('object').columns:
        df[col] = df[col].astype('category')
        
    print("Dataframe memory usage: %.2f MB" % (df.memory_usage().sum()/(1024*1024)))
    
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

    black_friday_dates = []
    cyber_monday_dates = []
    back_to_school_dates = []
    moving_dates = []
    christmas_dates = []
    new_year_dates = []

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

    df['days_to_black_friday']   = df['Submit_Date_day'].apply(lambda x: days_to_event(x, black_friday_dates))
    df['days_to_cyber_monday']   = df['Submit_Date_day'].apply(lambda x: days_to_event(x, cyber_monday_dates))
    df['days_to_christmas']      = df['Submit_Date_day'].apply(lambda x: days_to_event(x, christmas_dates))
    df['days_to_new_year']       = df['Submit_Date_day'].apply(lambda x: days_to_event(x, new_year_dates))
    df['days_to_moving_day']     = df['Submit_Date_day'].apply(lambda x: days_to_event(x, moving_dates))
    df['days_to_back_to_school'] = df['Submit_Date_day'].apply(lambda x: days_to_event(x, back_to_school_dates))
    
    return df


def build_lag_features(df):
    df['lag_1'] = df['Ticket cnt'].shift(1) # use the number of tickets 1 day prior the current date
    df['lag_2'] = df['Ticket cnt'].shift(2) # use the number of tickets 2 days prior the current date
    df['lag_3'] = df['Ticket cnt'].shift(3) # use the number of tickets 3 days prior the current date
    df['lag_4'] = df['Ticket cnt'].shift(4) # use the number of tickets 4 days prior the current date
    df['lag_5'] = df['Ticket cnt'].shift(5) # use the number of tickets 5 days prior the current date
    df['lag_6'] = df['Ticket cnt'].shift(6) # use the number of tickets 6 days prior the current date
    df['lag_7'] = df['Ticket cnt'].shift(7) # use the number of tickets 7 days prior the current date
    df['lag_14'] = df['Ticket cnt'].shift(14) # use the number of tickets 14 days prior the current date
    df['lag_21'] = df['Ticket cnt'].shift(21) # use the number of tickets 21 days prior the current date
    df['lag_28'] = df['Ticket cnt'].shift(28) # use the number of tickets 28 days prior the current date
    
    return df


def compute_train_val_test_dates(train_start_date, val_length, test_length, offset_months):
    train_end_date  = train_start_date + relativedelta(months=offset_months) - relativedelta(days=1)
    val_start_date  = train_end_date   + relativedelta(days=1)
    val_end_date    = val_start_date   + relativedelta(months=val_length) - relativedelta(days=1)
    test_start_date = val_end_date     + relativedelta(days=1)
    test_end_date   = test_start_date  + relativedelta(months=test_length)

    return train_end_date, val_start_date, val_end_date, test_start_date, test_end_date


def get_train_val_test_dataset(df, train_start_dt, val_length, test_length, offset_months):
    train_end_dt, val_start_dt, val_end_dt, test_start_dt, test_end_dt = compute_train_val_test_dates(train_start_dt, val_length,\
                                                                                                      test_length, offset_months)

#     print('Fold %d:' % offset_months)
#     print('Train set: from %s to %s' % (train_start_date, train_end_date))
#     print('Validation set: from %s to %s' % (val_start_date, val_end_date))
#     print('Test set: from %s to %s\n' % (test_start_date, test_end_date))

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
    
    
def plot_pearson_corr(df, list_, plot=True):
    corr_df = pd.DataFrame([], columns=['feature', 'pearson_r', 'abs_pearson_r', 'p-value'])
    
    for col in list_:
        condition = ~df['Ticket cnt'].isna()

        pearson_r , p = stats.pearsonr(df.loc[condition, 'Ticket cnt'], df.loc[condition, col])
        
        if plot is True:
            # Compute rolling window synchrony
            f, ax = plt.subplots(figsize=(8,2.5))

            df.loc[condition, [col, 'Ticket cnt']].rolling(window=30,center=True).median().plot(ax=ax)
            ax.set(xlabel='Days', ylabel='Pearson r')
            ax.set(title=f"Overall Pearson r = {np.round(pearson_r, 4)}")
        
        corr_df = corr_df.append({'feature':col, 'pearson_r':pearson_r, 'abs_pearson_r':abs(pearson_r), 'p-value': p}, ignore_index=True)
        
    return corr_df