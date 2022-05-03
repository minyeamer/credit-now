import numpy as np
import pandas as pd
from sklearn import model_selection
import joblib


def load_data(name='train', test_size=0.3, encoding=True) -> tuple:
    if not name:
        name = 'train'
        train_data = pd.read_csv(f'original_data/{name}.csv')
        train_data = preprocess_data(train_data)
    else:
        train_data = pd.read_csv(f'credit_data/{name}_data.csv')
    train_label = np.array(train_data[['credit']])

    if test_size:
        train_data = train_data.drop(['index', 'credit'], axis=1)
        train_data, test_data, train_label, test_label = \
            model_selection.train_test_split(train_data, train_label, test_size=test_size,
                                            random_state=0, stratify=train_label)
    else:
        test_data, test_label = train_data.copy(), train_label.copy()

    if encoding:
        pipe = joblib.load(f'credit_data/{name}_pipe.pkl')
        train_data = pipe.fit_transform(train_data)
        test_data = pipe.transform(test_data)

    data = ((train_data, test_data, train_label, test_label)
            if test_size else (train_data, train_label))

    return data


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    data = df.drop(['FLAG_MOBIL'], axis=1).copy()
    data['credit'] = data['credit'].astype(int)

    data = data[data['occyp_type'].notnull() | (data['DAYS_EMPLOYED'] > 0)]
    data['occyp_type'] = data['occyp_type'].fillna('Unemployeed')

    data['child_num'] = data['child_num'].apply(lambda x: 4 if x > 4 else x)
    data['family_size'] = data['family_size'].apply(lambda x: 6 if x > 6 else x)
    data['family_size'] = data['family_size'].astype(int)

    data['DAYS_BIRTH'] = data['DAYS_BIRTH'].apply(lambda x: (x*-1)/365 if x < 0 else 0)
    data['DAYS_EMPLOYED'] = data['DAYS_EMPLOYED'].apply(lambda x: (x*-1)/365 if x < 0 else 0)
    data['begin_month'] = data['begin_month'].apply(lambda x: (x*-1)/12 if x < 0 else 0)
    data.rename(columns={'DAYS_BIRTH':'age','DAYS_EMPLOYED':'employed_year',
                        'begin_month':'begin_year'}, inplace=True)

    category_dict = get_category_dict()
    for column, cat_dict in category_dict.items():
        data[column].replace(cat_dict, inplace=True)

    return data


def get_category_dict() -> dict:
    category_dict = dict()

    category_dict['gender'] = {'M':0,'F':1}
    category_dict['car'] = {'N':0,'Y':1}
    category_dict['reality'] = {'N':0,'Y':1}
    category_dict['income_type'] = {'Working': 0, 'Commercial associate': 1, 'Pensioner': 2, 'State servant': 3, 'Student': 4}
    category_dict['edu_type'] = {'Secondary / secondary special': 0, 'Higher education': 1, 'Incomplete higher': 2,
                        'Lower secondary': 3, 'Academic degree': 4}
    category_dict['family_type'] = {'Married': 0, 'Single / not married': 1, 'Civil marriage': 2, 'Separated': 3, 'Widow': 4}
    category_dict['house_type'] = {'House / apartment': 0, 'With parents': 1, 'Municipal apartment': 2, 'Rented apartment': 3,
                            'Office apartment': 4, 'Co-op apartment': 5}
    category_dict['occyp_type'] = {'Unemployeed': 0, 'Laborers': 1, 'Core staff': 2, 'Sales staff': 3, 'Managers': 4, 'Drivers': 5,
                            'High skill tech staff': 6, 'Accountants': 7, 'Medicine staff': 8, 'Cooking staff': 9,
                            'Security staff': 10, 'Cleaning staff': 11, 'Private service staff': 12, 'Low-skill Laborers': 13,
                            'Waiters/barmen staff': 14, 'Secretaries': 15, 'Realty agents': 16, 'HR staff': 17, 'IT staff': 18}

    return category_dict
