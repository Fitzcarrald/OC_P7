import numpy as np
import pandas as pd

import gc

import time
from contextlib import contextmanager


# Reduce the dataframe numeric memory
def reduce_memory_usage(df, floating=True, verbose=False):
    start_memory = df.memory_usage().sum() / 1024**2

    for col in df.select_dtypes('integer').columns:
        col_min = df[col].min()
        col_max = df[col].max()

        if str(df[col].dtype) in ['int16', 'int32', 'int64']:
            if col_min > np.iinfo(np.int8).min and col_max < np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif col_min > np.iinfo(np.int16).min and col_max < np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif col_min > np.iinfo(np.int32).min and col_max < np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
            elif col_min > np.iinfo(np.int64).min and col_max < np.iinfo(np.int64).max:
                df[col] = df[col].astype(np.int64)

    if floating:
        for col in df.select_dtypes('floating').columns:
            col_min = df[col].min()
            col_max = df[col].max()

            if col_min > np.finfo(np.float16).min and col_max < np.finfo(np.float16).max:
                df[col] = df[col].astype(np.float16)
            elif col_min > np.finfo(np.float32).min and col_max < np.finfo(np.float32).max:
                df[col] = df[col].astype(np.float32)

    end_memory = df.memory_usage().sum() / 1024**2
    if verbose:
        print(
            f'Initial memory usage of the dataframe: {start_memory:.2f} MB\n'
            f'Final memory usage, after optimization: {end_memory:.2f} MB\n'
            f'Decreased by {100 * (start_memory - end_memory) / start_memory:.2f}%'
        )

    return df


# Timer
@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print(f'{title} - done in {time.time() - t0:.0f}s')


# Aggregate the categorical features
def agg_cat(df, ID, prefix):
    # In the categorical values replace any character that is not a letter,
    # digit, or underscore with an underscore
    df[df.select_dtypes('object').columns] = df.select_dtypes('object').replace(
        regex='[\W]', value=' ').replace(regex='\s+', value='_')
    # Select the categorical columns
    categorical = pd.get_dummies(df.select_dtypes('object'))
    # Introduce the ID as a column
    categorical[ID] = df[ID]
    # Aggregate by ID and perform sum and mean
    categorical_aggregated = categorical.groupby(ID).agg(['mean'])
    # Add the prefix to the column names
    categorical_aggregated.columns = [
        prefix + '_' + '_'.join(col) for col in categorical_aggregated.columns]

    return categorical_aggregated


# Aggregate the numerical features
def agg_num(df, ID, prefix, addition=True):
    # Select the numerical columns except any SK_ID identifier
    numerical = df.select_dtypes('number').filter(regex='^(?!SK_ID)')
    # Introduce (again) the ID as a column
    numerical[ID] = df[ID]
    # Aggregate by ID and perform statistics
    aggregations = ['sum', 'max', 'min', 'mean']
    None if addition else aggregations.remove('sum')
    numerical_aggregated = numerical.groupby(ID).agg(aggregations)
    # Add the prefix to the column names
    numerical_aggregated.columns = [
        prefix + '_' + '_'.join(col) for col in numerical_aggregated.columns]

    return numerical_aggregated


# Preprocess application_train.csv and application_test.csv
def application_train_test():
    # Read data and merge
    df_train = reduce_memory_usage(
        pd.read_csv(
            'data/application_train.csv'
        )
    )
    df_test = reduce_memory_usage(
        pd.read_csv(
            'data/application_test.csv'
        )
    )
    df = df_train.append(df_test).reset_index(drop=True)
    # # Drop any flag column
    # df = df.filter(regex='^(?!FLAG)')
    # Make sure tha any flag column is a categorical one    
    for col in df.filter(like='FLAG').select_dtypes(exclude='object').columns:
        # Due to a bug of the in place argument of replace this operation had
        # to be done column-wise
        df[col].replace({0: 'Y', 1: 'N'}, inplace=True)
    # Remove 4 applications with XNA CODE_GENDER (train set)
    df = df[df['CODE_GENDER'].ne('XNA')]
    # Set to NaN the identified outliers
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
    df.loc[df.OBS_30_CNT_SOCIAL_CIRCLE > 30,
           'OBS_30_CNT_SOCIAL_CIRCLE'] = np.nan
    df.loc[df.DEF_30_CNT_SOCIAL_CIRCLE > 30,
           'DEF_30_CNT_SOCIAL_CIRCLE'] = np.nan
    df.loc[df.OBS_60_CNT_SOCIAL_CIRCLE > 60,
           'OBS_60_CNT_SOCIAL_CIRCLE'] = np.nan
    df.loc[df.DEF_60_CNT_SOCIAL_CIRCLE > 60,
           'DEF_60_CNT_SOCIAL_CIRCLE'] = np.nan
    # In the categorical values replace any character that is not a letter,
    # digit, or underscore with an underscore
    df[df.select_dtypes('object').columns] = df.select_dtypes('object').replace(
        regex='[\W]', value=' ').replace(regex='\s+', value='_')
    # # Add the APP prefix to the column names, except to TARGET and the identifier
    # df.columns = [col if col in ['SK_ID_CURR', 'TARGET']
    #               else 'APP_' + col for col in df.columns]
    del df_train, df_test
    gc.collect()

    return df


# Preprocess bureau.csv and bureau_balance.csv
def bureau_and_bureau_balance():
    bureau = reduce_memory_usage(
        pd.read_csv(
            'data/bureau.csv'
        )
    )
    bureau_balance = reduce_memory_usage(
        pd.read_csv(
            'data/bureau_balance.csv'
        )
    )
    # Set to NaN the identified outliers
    bureau['DAYS_CREDIT_ENDDATE'].mask(
        bureau.DAYS_CREDIT_ENDDATE < -36500, np.nan, inplace=True)
    bureau['DAYS_ENDDATE_FACT'].mask(
        bureau.DAYS_ENDDATE_FACT < -36500, np.nan, inplace=True)
    bureau['DAYS_CREDIT_UPDATE'].mask(
        bureau.DAYS_CREDIT_UPDATE < -36500, np.nan, inplace=True)
    # Aggregate bureau_balance
    bb_categorical = agg_cat(bureau_balance, 'SK_ID_BUREAU', 'BB')
    bb_numerical = agg_num(
        bureau_balance,
        'SK_ID_BUREAU',
        'BB',
        addition=False
    )
    # Count bureau_balance lines by loan
    bb_count = bureau_balance.groupby('SK_ID_BUREAU').size().rename('BB_COUNT')
    bureau_balance_aggregated = pd.concat(
        [
            bb_categorical,
            bb_numerical,
            bb_count
        ],
        axis=1
    )
    del bureau_balance, bb_categorical, bb_numerical, bb_count
    gc.collect()

    # Merge bureau_balance to bureau
    bureau = bureau.merge(
        bureau_balance_aggregated,
        on='SK_ID_BUREAU',
        how='left'
    )
    # Aggregate bureau
    bu_categorical = agg_cat(bureau, 'SK_ID_CURR', 'BU')
    # Aggregate the numerical features except two
    bu_numerical1 = agg_num(
        bureau.drop(columns=['DAYS_ENDDATE_FACT', 'DAYS_CREDIT_ENDDATE']),
        'SK_ID_CURR',
        'BU'
    )
    # Aggregate the missing features without their respecive sum
    # since both are calculated as infinite in each case
    bu_numerical2 = agg_num(
        bureau[['SK_ID_CURR', 'DAYS_ENDDATE_FACT', 'DAYS_CREDIT_ENDDATE']],
        'SK_ID_CURR',
        'BU',
        addition=False
    )
    # Count bureau lines by client
    bu_count = bureau.groupby('SK_ID_CURR').size().rename('BU_COUNT')
    bureau_aggregated = pd.concat(
        [
            bu_categorical,
            bu_numerical1,
            bu_numerical2,
            bu_count
        ],
        axis=1
    )
    del bu_categorical, bu_numerical1, bu_numerical2, bu_count
    gc.collect()

    return bureau_aggregated


# Preprocess previous_applications.csv
def previous_application():
    prev_app = reduce_memory_usage(
        pd.read_csv(
            'data/previous_application.csv'
        )
    )
    # # Drop any flag column
    # prev_app = prev_app.filter(regex='^(?!FLAG)')
    # Set to NaN the identified outliers
    prev_app['DAYS_FIRST_DRAWING'].replace(
        365243, np.nan, inplace=True)
    prev_app['DAYS_FIRST_DUE'].replace(
        365243, np.nan, inplace=True)
    prev_app['DAYS_LAST_DUE_1ST_VERSION'].replace(
        365243, np.nan, inplace=True)
    prev_app['DAYS_LAST_DUE'].replace(
        365243, np.nan, inplace=True)
    prev_app['DAYS_TERMINATION'].replace(
        365243, np.nan, inplace=True)
    # Aggregate
    pa_categorical = agg_cat(prev_app, 'SK_ID_CURR', 'PA')
    pa_numerical = agg_num(prev_app, 'SK_ID_CURR', 'PA')
    # Count previous_application lines by client
    pa_count = prev_app.groupby('SK_ID_CURR').size().rename('PA_COUNT')
    previous_application_aggregated = pd.concat(
        [
            pa_categorical,
            pa_numerical,
            pa_count
        ],
        axis=1
    )
    del prev_app, pa_categorical, pa_numerical, pa_count
    gc.collect()

    return previous_application_aggregated


# Preprocess POS_CASH_balance.csv
def POS_CASH_balance():
    pos_cash = reduce_memory_usage(
        pd.read_csv(
            'data/POS_CASH_balance.csv'
        )
    )
    # Aggregate
    pc_categorical = agg_cat(pos_cash, 'SK_ID_CURR', 'PC')
    pc_numerical = agg_num(pos_cash, 'SK_ID_CURR', 'PC')
    # Count POS_CASH_balance lines by client
    pc_count = pos_cash.groupby('SK_ID_CURR').size().rename('PC_COUNT')
    POS_CASH_balance_aggregated = pd.concat(
        [
            pc_categorical,
            pc_numerical,
            pc_count
        ],
        axis=1
    )
    del pos_cash, pc_categorical, pc_numerical, pc_count
    gc.collect()

    return reduce_memory_usage(POS_CASH_balance_aggregated)


# Preprocess installments_payments.csv
def installments_payments():
    ins_pay = reduce_memory_usage(
        pd.read_csv(
            'data/installments_payments.csv'
        )
    )
    # Aggregate the numerical features except two
    ins_pay_numerical1 = agg_num(
        ins_pay.drop(columns=['DAYS_INSTALMENT', 'DAYS_ENTRY_PAYMENT']),
        'SK_ID_CURR',
        'IP'
    )
    # Aggregate the missing features without their respective sum
    # since both are calculated as infinite in each case
    ins_pay_numerical2 = agg_num(
        ins_pay[['SK_ID_CURR', 'DAYS_INSTALMENT', 'DAYS_ENTRY_PAYMENT']],
        'SK_ID_CURR',
        'IP',
        addition=False
    )
    # Count installments_payments lines by client
    ins_pay_count = ins_pay.groupby('SK_ID_CURR').size().rename('IP_COUNT')
    installments_payments_aggregated = pd.concat(
        [
            ins_pay_numerical1,
            ins_pay_numerical2,
            ins_pay_count
        ],
        axis=1
    )
    del ins_pay, ins_pay_numerical1, ins_pay_numerical2, ins_pay_count
    gc.collect()

    return installments_payments_aggregated


# Preprocess credit_card_balance.csv
def credit_card_balance():
    cred_card = reduce_memory_usage(
        pd.read_csv(
            'data/credit_card_balance.csv'
        )
    )
    # Aggregate
    cc_categorical = agg_cat(cred_card, 'SK_ID_CURR', 'CC')
    cc_numerical = agg_num(cred_card, 'SK_ID_CURR', 'CC')
    # Count credit_card_balance lines by client
    cc_count = cred_card.groupby('SK_ID_CURR').size().rename('CC_COUNT')
    credit_card_balance_aggregated = pd.concat(
        [
            cc_categorical,
            cc_numerical,
            cc_count
        ],
        axis=1
    )
    del cred_card, cc_categorical, cc_numerical, cc_count
    gc.collect()

    return reduce_memory_usage(credit_card_balance_aggregated)


# Preprocess and merge all the tables
def home_credit_dataframe():
    with timer('Process train and test application'):
        df = application_train_test()

    with timer('Process bureau and bureau_balance'):
        bureau = bureau_and_bureau_balance()
        df = df.merge(
            bureau,
            on='SK_ID_CURR',
            how='left'
        )
        del bureau
        gc.collect()
    with timer('Process previous_applications'):
        prev_app = previous_application()
        df = df.merge(
            prev_app,
            on='SK_ID_CURR',
            how='left'
        )
        del prev_app
        gc.collect()
    with timer('Process POS_CASH_balance'):
        POS_CASH = POS_CASH_balance()
        df = df.merge(
            POS_CASH,
            on='SK_ID_CURR',
            how='left'
        )
        del POS_CASH
        gc.collect()
    with timer('Process installments_payments'):
        ins_pay = installments_payments()
        df = df.merge(
            ins_pay,
            on='SK_ID_CURR',
            how='left'
        )
        del ins_pay
        gc.collect()
    with timer('Process credit_card_balance'):
        cred_card = credit_card_balance()
        df = df.merge(
            cred_card,
            on='SK_ID_CURR',
            how='left'
        )
        del cred_card
        gc.collect()
    # df.drop(columns='SK_ID_CURR', inplace=True)    

    return reduce_memory_usage(df)
