import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib


def load_data() -> tuple:
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')

    train_data = train_df.drop(['index', 'FLAG_MOBIL', 'credit'], axis=1)
    test_data = test_df.drop(['index', 'FLAG_MOBIL'], axis=1)

    train_data = prep_data(train_data)
    test_data = prep_data(test_data)

    pipe = joblib.load(f'credit_pipe.pkl')
    train_data = pipe.fit_transform(train_data)
    test_data = pipe.fit_transform(test_data)

    train_label = np.array(train_df[['credit']])

    return (train_data, train_label), (test_data)


def prep_data(df) -> pd.DataFrame:

    data = df.copy()
    data = data[data['occyp_type'].notnull() | (data['DAYS_EMPLOYED'] > 0)]
    data['occyp_type'] = data['occyp_type'].fillna('Unemployeed')

    data['gender'].replace({'M':0,'F':1}, inplace=True)
    data['car'].replace({'N':0,'Y':1}, inplace=True)
    data['reality'].replace({'N':0,'Y':1}, inplace=True)

    data['child_num'] = data['child_num'].apply(lambda x: 4 if x > 4 else x)
    data['DAYS_BIRTH'] = data['DAYS_BIRTH'].apply(lambda x: (x*-1)/365)
    data['DAYS_EMPLOYED'] = data['DAYS_EMPLOYED'].apply(lambda x: (x*-1)/365)
    data['begin_month'] = data['begin_month'].apply(lambda x: (x*-1)/12)
    data['family_size'] = data['family_size'].apply(lambda x: 6 if x > 6 else x)

    name_dict = get_name_dict()
    data['income_type'].replace(name_dict['income'], inplace=True)
    data['edu_type'].replace(name_dict['edu'], inplace=True)
    data['family_type'].replace(name_dict['family'], inplace=True)
    data['house_type'].replace(name_dict['house'], inplace=True)
    data['occyp_type'].replace(name_dict['occyp'], inplace=True)

    return data


def get_name_dict() -> dict:
    name_dict = dict()

    name_dict['income'] = {'Working': 0, 'Commercial associate': 1, 'Pensioner': 2, 'State servant': 3, 'Student': 4}
    name_dict['edu'] = {'Secondary / secondary special': 0, 'Higher education': 1, 'Incomplete higher': 2,
                        'Lower secondary': 3, 'Academic degree': 4}
    name_dict['family'] = {'Married': 0, 'Single / not married': 1, 'Civil marriage': 2, 'Separated': 3, 'Widow': 4}
    name_dict['house'] = {'House / apartment': 0, 'With parents': 1, 'Municipal apartment': 2, 'Rented apartment': 3,
                            'Office apartment': 4, 'Co-op apartment': 5}
    name_dict['occyp'] = {'Unemployeed': 0, 'Laborers': 1, 'Core staff': 2, 'Sales staff': 3, 'Managers': 4, 'Drivers': 5,
                            'High skill tech staff': 6, 'Accountants': 7, 'Medicine staff': 8, 'Cooking staff': 9,
                            'Security staff': 10, 'Cleaning staff': 11, 'Private service staff': 12, 'Low-skill Laborers': 13,
                            'Waiters/barmen staff': 14, 'Secretaries': 15, 'Realty agents': 16, 'HR staff': 17, 'IT staff': 18}

    return name_dict
