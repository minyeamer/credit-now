import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from category_encoders.ordinal import OrdinalEncoder
import joblib


def load_train_data(path='credit_data', name='train', test_size=0.3, encoding=True) -> tuple:
    train_data = pd.read_csv(f'{path}/{name}.csv')
    train_data = preprocess_data(train_data) if path == 'original_data' else train_data
    train_label = train_data[['index','credit']]

    if test_size:
        train_data = train_data.drop(['index', 'credit'], axis=1)
        train_label = np.array(train_label[['credit']])
        train_data, test_data, train_label, test_label = \
            model_selection.train_test_split(train_data, train_label, test_size=test_size,
                                            random_state=0, stratify=train_label)
    else:
        test_data, test_label = train_data.copy(), train_label.copy()

    if encoding:
        pipe = joblib.load(f'{path}/{name}_pipe.pkl')
        train_data = pipe.fit_transform(train_data)
        test_data = pipe.transform(test_data)

    data = ((train_data, test_data, train_label, test_label)
            if test_size else (train_data, train_label))

    return data


def load_test_data(encoding=True) -> any:
    test_data = pd.read_csv(f'original_data/test.csv')
    test_data = preprocess_data(test_data, name='test')

    if encoding:
        pipe = joblib.load(f'credit_data/train_pipe.pkl')
        test_data = pipe.transform(test_data)
    
    return test_data


def preprocess_data(df: pd.DataFrame, name='train') -> pd.DataFrame:
    data = df.fillna('NaN').copy()
    data = data.drop(['FLAG_MOBIL'], axis=1)
    client_input = data.columns.tolist()

    data['DAYS_EMPLOYED'] = data['DAYS_EMPLOYED'].apply(lambda x: 0 if x > 0 else x)
    data['occyp_type'] = ['Unemployed' if emp == 0 else occ
                            for emp, occ in zip(data['DAYS_EMPLOYED'],data['occyp_type'])]

    data['family_size'] = data['family_size'].apply(lambda x: 6 if x > 6 else x)
    data[['work_phone','phone','email']] = data[['work_phone','phone','email']].astype('object')

    date_columns = ['DAYS_BIRTH', 'DAYS_EMPLOYED', 'begin_month']
    data[date_columns] = data[date_columns].apply(lambda x: abs(x))

    data = set_extra_features(data, client_input)
    data.drop(['child_num','DAYS_BIRTH','DAYS_EMPLOYED'], axis=1, inplace=True)
    data = set_ordinal_encoding(data, name)
    data = set_clustering(data, name)

    columns = data.drop(['index','credit'], axis=1).columns.tolist()
    data = data.reindex(columns=['index']+sorted(columns)+['credit'])
    data.set_index('index').to_csv(f'credit_data/{name}.csv')

    return data


def set_extra_features(df: pd.DataFrame, client_input: list) -> pd.DataFrame:
    data = df.copy()

    data['age'] = data['DAYS_BIRTH'] // 365
    data['month_birth'] = np.floor(data['DAYS_BIRTH']/30) - ((np.floor(data['DAYS_BIRTH']/30)/12).astype(int)*12)
    data['week_birth'] = np.floor(data['DAYS_BIRTH']/7) - ((np.floor(data['DAYS_BIRTH']/7)/4).astype(int)*4)

    data['career'] = data['DAYS_EMPLOYED'] // 365
    data['days_unemployed'] = data['DAYS_BIRTH'] - data['DAYS_EMPLOYED']
    data['month_unemployed'] = np.floor(data['days_unemployed']/30) - ((np.floor(data['days_unemployed']/30)/12).astype(int)*12)
    data['week_unemployed'] = np.floor(data['days_unemployed']/7) - ((np.floor(data['days_unemployed']/7)/4).astype(int)*4)

    data['days_income'] = data['income_total'] / (data['DAYS_BIRTH']+data['DAYS_EMPLOYED'])
    data['income_per'] = data['income_total'] / data['family_size']

    # id 열은 새로운 데이터에 적용하기 적절하지 않기 때문에 제거, 향후 중복 여부를 판단하는 열로 변경해서 시도
    # data['id'] = str()
    # for column in client_input:
    #     data['id'] += data[column].astype(str) + '_'

    return data


def set_ordinal_encoding(df: pd.DataFrame, name: str) -> pd.DataFrame:
    data = df.copy()

    num_features = data.dtypes[data.dtypes != 'object'].index.tolist()
    cat_features = data.dtypes[data.dtypes == 'object'].index.tolist()
    if name == 'train':
        ordianl_encoder = OrdinalEncoder(cat_features)
        data[cat_features] = ordianl_encoder.fit_transform(data[cat_features],data['credit'])
        joblib.dump(ordianl_encoder, 'credit_data/train_ord_pipe.pkl')
    else:
        ordianl_encoder = joblib.load('credit_data/train_ord_pipe.pkl')
        data[cat_features] = ordianl_encoder.transform(data[cat_features])
    # data['id'] = data['id'].astype('int64')

    if name == 'train':
        make_pipeline(data, num_features, cat_features)

    return data


def set_clustering(df: pd.DataFrame, name: str) -> pd.DataFrame:
    data = df.copy()

    train = data
    if name == 'train':
        train = data.drop(['credit'], axis=1)
        kmeans = KMeans(n_clusters=36, random_state=42).fit(train)
        joblib.dump(kmeans, 'models/model_train_kmeans.pkl')
    
    kmeans = joblib.load('models/model_train_kmeans.pkl')
    data['cluster'] = kmeans.predict(train)

    return data


def make_pipeline(df: pd.DataFrame, num_features: list, cat_features: list):
    data = df.drop(['index','credit'], axis=1).copy()
    num_features = [feature for feature in num_features if feature not in ['index','credit']]
    cat_features = [feature for feature in cat_features if feature not in ['index','credit']]

    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(categories='auto', handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, num_features),
            ('cat', categorical_transformer, cat_features)])

    pipe = Pipeline(steps=[('preprocessor', preprocessor)])
    pipe.fit(data)
    joblib.dump(pipe, 'credit_data/train_pipe.pkl')


###########################################################################
############################## Old Functions ##############################
###########################################################################


def load_data(name='train_old', test_size=0.3, encoding=True) -> tuple:
    if not name:
        name = 'train'
        train_data = pd.read_csv(f'original_data/{name}.csv')
        train_data = old_preprocess_data(train_data)
    else:
        train_data = pd.read_csv(f'credit_data/{name}.csv')
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


def old_preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
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

    category_dict = old_get_category_dict()
    for column, cat_dict in category_dict.items():
        data[column].replace(cat_dict, inplace=True)

    return data


def old_get_category_dict() -> dict:
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
