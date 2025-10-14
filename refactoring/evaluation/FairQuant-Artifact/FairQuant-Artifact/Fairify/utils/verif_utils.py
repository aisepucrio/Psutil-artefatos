# -*- coding: utf-8 -*-

from codecarbon import EmissionsTracker
import psutil
import csv
import os
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from z3 import *
import math
from random import randrange
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score


warnings.filterwarnings('ignore')
sys.path.append('../')


def load_data_from_csv(train_path, test_path, column_names, na_values=None, header=None, skipinitialspace=True):
    train = pd.read_csv(train_path, header=header, names=column_names, skipinitialspace=skipinitialspace, na_values=na_values)
    test = pd.read_csv(test_path, header=0, names=column_names, skipinitialspace=skipinitialspace, na_values=na_values)
    return pd.concat([test, train], ignore_index=True)


def preprocess_adult_data(df):
    imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    df['workclass'] = imputer.fit_transform(df[['workclass']]).ravel()
    df['occupation'] = imputer.fit_transform(df[['occupation']]).ravel()
    df['native-country'] = imputer.fit_transform(df[['native-country']]).ravel()

    hs_grad = ['HS-grad', '11th', '10th', '9th', '12th']
    elementary = ['1st-4th', '5th-6th', '7th-8th']
    df['education'].replace(to_replace=hs_grad, value='HS-grad', inplace=True)
    df['education'].replace(to_replace=elementary, value='elementary_school', inplace=True)

    married = ['Married-spouse-absent', 'Married-civ-spouse', 'Married-AF-spouse']
    separated = ['Separated', 'Divorced']
    df['marital-status'].replace(to_replace=married, value='Married', inplace=True)
    df['marital-status'].replace(to_replace=separated, value='Separated', inplace=True)

    self_employed = ['Self-emp-not-inc', 'Self-emp-inc']
    govt_employees = ['Local-gov', 'State-gov', 'Federal-gov']
    df['workclass'].replace(to_replace=self_employed, value='Self_employed', inplace=True)
    df['workclass'].replace(to_replace=govt_employees, value='Govt_employees', inplace=True)

    return df


def encode_categorical_features(df, cat_feat):
    for feature in cat_feat:
        le = LabelEncoder()
        df[feature] = le.fit_transform(df[feature])
    return df


def prepare_adult_data(df, label_name, favorable_classes):
    df = df.dropna()
    X = df.drop(labels=[label_name], axis=1, inplace=False)
    y = df[label_name]
    pos = np.logical_or.reduce(np.equal.outer(favorable_classes, df[label_name].to_numpy()))
    df.loc[pos, label_name] = 1
    df.loc[~pos, label_name] = 0
    return X, y


def split_data(X, y, test_size=0.15, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test


def load_adult_data(train_path, test_path, column_names, label_name, favorable_classes, preprocess_func=None, cat_feat=None, seed=42, drop_cols=None):
    df = load_data_from_csv(train_path, test_path, column_names, na_values=['?'])

    if preprocess_func:
        df = preprocess_func(df)

    if drop_cols:
        df.drop(labels=drop_cols, axis=1, inplace=True)

    if cat_feat:
        df = encode_categorical_features(df, cat_feat)

    X, y = prepare_adult_data(df, label_name, favorable_classes)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.15, random_state=seed)

    return df, X_train.to_numpy(), y_train.to_numpy().astype('int'), X_test.to_numpy(), y_test.to_numpy().astype('int')


def load_adult_adf():
    train_path = '../../data/adult/adult.data'
    test_path = '../../data/adult/adult.test'
    column_names = ['age', 'workclass', 'fnlwgt', 'education',
                    'education-num', 'marital-status', 'occupation', 'relationship',
                    'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
                    'native-country', 'income-per-year']
    label_name = 'income-per-year'
    favorable_classes = ['>50K', '>50K.']
    drop_cols = ['fnlwgt']
    cat_feat = ['sex', 'workclass', 'education', 'marital-status', 'occupation', 'relationship', 'native-country', 'race']
    df, X_train, y_train, X_test, y_test = load_adult_data(
        train_path, test_path, column_names, label_name, favorable_classes,
        preprocess_func=None, cat_feat=cat_feat, drop_cols=drop_cols
    )
    return df, X_train, y_train, X_test, y_test


def load_adult_ac1():
    train_path = '../../../data/adult/adult.data'
    test_path = '../../../data/adult/adult.test'
    column_names = ['age', 'workclass', 'fnlwgt', 'education',
                    'education-num', 'marital-status', 'occupation', 'relationship',
                    'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
                    'native-country', 'income-per-year']
    label_name = 'income-per-year'
    favorable_classes = ['>50K', '>50K.']
    drop_cols = ['fnlwgt']
    cat_feat = ['sex', 'workclass', 'education', 'marital-status', 'occupation', 'relationship', 'native-country', 'race']
    df, X_train, y_train, X_test, y_test = load_adult_data(
        train_path, test_path, column_names, label_name, favorable_classes,
        preprocess_func=None, cat_feat=cat_feat, drop_cols=drop_cols
    )
    return df, X_train, y_train, X_test, y_test


def load_german():
    filepath = '../../../data/german/german.data'
    column_names = ['status', 'month', 'credit_history',
                    'purpose', 'credit_amount', 'savings', 'employment',
                    'investment_as_income_percentage', 'personal_status',
                    'other_debtors', 'residence_since', 'property', 'age',
                    'installment_plans', 'housing', 'number_of_credits',
                    'skill_level', 'people_liable_for', 'telephone',
                    'foreign_worker', 'credit']
    df = pd.read_csv(filepath, sep=' ', header=None, names=column_names)
    df['age'] = df['age'].apply(lambda x: np.float(x >= 26))
    df = german_custom_preprocessing(df)
    feat_to_drop = ['personal_status']
    df = df.drop(feat_to_drop, axis=1)
    cat_feat = ['status', 'credit_history', 'purpose', 'savings', 'employment', 'other_debtors', 'property',
                'installment_plans', 'housing', 'skill_level', 'telephone', 'foreign_worker']
    for f in cat_feat:
        label = LabelEncoder()
        df[f] = label.fit_transform(df[f])
    label_name = 'credit'
    X = df.drop(labels=[label_name], axis=1, inplace=False)
    y = df[label_name]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    return df, X_train.to_numpy(), y_train.to_numpy().astype('int'), X_test.to_numpy(), y_test.to_numpy().astype('int')


def load_bank():
    file_path = '../../../data/bank/bank-additional-full.csv'
    column_names = ['age', 'job', 'marital', 'education', 'default', 'housing', 'loan', 'contact',
                    'month', 'day_of_week', 'duration', 'emp.var.rate',
                    'campaign', 'pdays', 'previous', 'poutcome', 'y']
    label_name = 'y'
    favorable_classes = ['yes']
    df = pd.read_csv(file_path, sep=';', na_values=['unknown'])
    df['age'] = df['age'].apply(lambda x: np.float(x >= 25))
    cat_feat = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']
    for f in cat_feat:
        label = LabelEncoder()
        df[f] = label.fit_transform(df[f])
    df = df[column_names]
    pos = np.logical_or.reduce(np.equal.outer(favorable_classes, df[label_name].to_numpy()))
    df.loc[pos, label_name] = 1
    df.loc[~pos, label_name] = 0
    df = df.round(0).astype(int)
    X = df.drop(labels=[label_name], axis=1, inplace=False)
    y = df[label_name]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    return df, X_train.to_numpy(), y_train.to_numpy().astype('int'), X_test.to_numpy(), y_test.to_numpy().astype('int')


def load_adult():
    train_path = '../../data/adult/adult.data'
    test_path = '../../data/adult/adult.test'
    column_names = ['age', 'workclass', 'fnlwgt', 'education',
                    'education-num', 'marital-status', 'occupation', 'relationship',
                    'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
                    'native-country', 'income-per-year']
    label_name = 'income-per-year'
    favorable_classes = ['>50K', '>50K.']
    drop_cols = ['education-num', 'fnlwgt']
    cat_feat = ['sex', 'workclass', 'education', 'marital-status', 'occupation', 'relationship', 'native-country']
    df, X_train, y_train, X_test, y_test = load_adult_data(
        train_path, test_path, column_names, label_name, favorable_classes,
        preprocess_func=preprocess_adult_data, cat_feat=cat_feat, drop_cols=drop_cols
    )
    return df, X_train, y_train, X_test, y_test


def load_compas():
    filepath = '../../../data/compas/propublica_data_for_fairml.csv'
    column_names = ['Two_yr_Recidivism', 'Number_of_Priors', 'score_factor',
                    'Age_Above_FourtyFive', 'Age_Below_TwentyFive',
                    'African_American', 'Asian', 'Hispanic', 'Native_American', 'Other',
                    'Female', 'Misdemeanor']

    df = pd.read_csv(filepath, header=0, names=column_names)

    age_column = [0 for _ in range(len(df))]
    race_column = [0 for _ in range(len(df))]

    for index, row in df.iterrows():
        if row['Age_Below_TwentyFive'] == 1:
            age_column[index] = 0
        elif row['Age_Above_FourtyFive'] == 1:
            age_column[index] = 1
        else:
            age_column[index] = 1

        if row['African_American'] == 1:
            race_column[index] = 1
        elif row['Asian'] == 1:
            race_column[index] = 1
        elif row['Hispanic'] == 1:
            race_column[index] = 1
        elif row['Native_American'] == 1:
            race_column[index] = 1
        elif row['Other'] == 1:
            race_column[index] = 1
        else:
            race_column[index] = 0

    df.insert(loc=3, column='Age', value=age_column)
    df.insert(loc=4, column='Race', value=race_column)

    feat_to_drop = ['Age_Above_FourtyFive', 'Age_Below_TwentyFive',
                    'African_American', 'Asian', 'Hispanic', 'Native_American', 'Other']
    df = df.drop(feat_to_drop, axis=1)

    label_name = 'score_factor'
    X = df.drop(labels=[label_name], axis=1, inplace=False)
    y = df[label_name]

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    return (df, X_train.to_numpy().astype('int'), y_train.to_numpy().astype('int'),
            X_val.to_numpy().astype('int'), y_val.to_numpy().astype('int'),
            X_test.to_numpy().astype('int'), y_test.to_numpy().astype('int'))


def load__trained_model(path):
    model = load_model(path)
    return model


def get_layer_weights(model):
    names = []
    weights = []
    biases = []
    for layer in model.layers:
        names.append(layer.get_config().get('name'))
        weights.append(layer.get_weights()[0])
        biases.append(layer.get_weights()[1])
    return names, weights, biases


def get_layer_outputs(model, single_input):
    inp = model.input
    outputs = [layer.output for layer in model.layers]
    functors = [K.function([inp], [out]) for out in outputs]
    X = single_input.reshape(1, 42)
    layer_outs = [func([X]) for func in functors]
    return layer_outs


def single_predict(model, X):
    return model.predict(X) > 0.5


def print_cols(dataframe):
    for col in dataframe:
        index = dataframe.columns.get_loc(col)
        print(index, col)


def print_uniques(dataframe):
    for col in dataframe:
        print(dataframe[col].unique())


def relu(x):
    return np.maximum(0, x)


def z3Relu(x):
    return np.vectorize(lambda y: If(y >= 0, y, RealVal(0)))(x)


def z3Abs(x):
    return If(x <= 0, -x, x)


def bigAnd(l):
    if not l:
        return True
    if len(l) == 1:
        return l[0]
    return And(*l)


def bigOr(l):
    if not l:
        return False
    if len(l) == 1:
        return l[0]
    return Or(*l)


def z3ToFrac(r):
    assert (is_rational_value(r))
    return r.as_fraction()


def z3ToFloat(r):
    return float(r.as_decimal(100).strip('?'))


def z3ToMath(f):
    s = str(f)
    s = s.replace("(", "[")
    s = s.replace(")", "]")
    return s


def compute_time(start_time, current_time):
    runtime = current_time - start_time
    return round(runtime, 2)


def print_metadata(dataframe):
    print(dataframe.columns)
    max_values = dataframe.max()
    min_values = dataframe.min()
    print('@@@@@ Min: ')
    print(min_values)
    print('@@@@@ Max: ')
    print(max_values)


def in_const_domain_ac1(df, x, x_, ranges, default):
    dataframe = df.drop(df.columns[len(df.columns) - 1], axis=1, inplace=False)
    props = []
    for col in dataframe:
        var = ''
        for var_name in ranges.keys():
            if col.startswith(var_name):
                var = col
                break
        index = dataframe.columns.get_loc(col)
        if var:
            props.append(And(x[index] >= ranges[var_name][0], x[index] <= ranges[var_name][1]))
            props.append(And(x_[index] >= ranges[var_name][0], x_[index] <= ranges[var_name][1]))
        else:
            props.append(And(x[index] >= default[0], x[index] <= default[1]))
            props.append(And(x_[index] >= default[0], x_[index] <= default[1]))
    return props


def in_const_range(df, x, x_, var_name, lb, ub):
    dataframe = df.drop(df.columns[len(df.columns) - 1], axis=1, inplace=False)
    props = []
    for col in dataframe:
        if col == var_name:
            index = dataframe.columns.get_loc(col)
            props.append(And(x[index] <= lb, x[index] >= ub))
            props.append(And(x_[index] <= lb, x_[index] >= ub))
    return props


def in_const_equality_domain(df, x, x_, ranges, PA):
    dataframe = df.drop(df.columns[len(df.columns) - 1], axis=1, inplace=False)
    props = []
    for col in dataframe:
        index = dataframe.columns.get_loc(col)
        if col in PA:
            p1 = []
            p2 = []
            val = ranges[col][0]
            while True:
                if val > ranges[col][1]:
                    break
                p1.append(x[index] == val)
                p2.append(x_[index] == val)
                val += 1
            props.append(bigOr(p1))
            props.append(bigOr(p2))
        else:
            p = []
            val = ranges[col][0]
            while True:
                if val > ranges[col][1]:
                    break
                p.append(x[index] == val)
                val += 1
            props.append(bigOr(p))
    return props


def in_const_domain_bank(df, x, x_, ranges, PA):
    label_name = 'y'
    dataframe = df.drop(labels=[label_name], axis=1, inplace=False)
    props = []
    for col in dataframe:
        index = dataframe.columns.get_loc(col)
        if col in PA:
            props.append(And(x[index] >= ranges[col][0], x[index] <= ranges[col][1]))
            props.append(And(x_[index] >= ranges[col][0], x_[index] <= ranges[col][1]))
        else:
            props.append(And(x[index] >= ranges[col][0], x[index] <= ranges[col][1]))
    return props


def in_const_bank(df, x, var_name, op, rhs):
    label_name = 'y'
    dataframe = df.drop(labels=[label_name], axis=1, inplace=False)
    props = []
    for col in dataframe:
        if col == var_name:
            index = dataframe.columns.get_loc(col)
            right = rhs if isinstance(rhs, (int, float)) else rhs[index]
            if op == 'gt':
                props.append(x[index] > right)
            elif op == 'lt':
                props.append(x[index] < right)
            elif op == 'gte':
                props.append(x[index] >= right)
            elif op == 'lte':
                props.append(x[index] <= right)
            elif op == 'eq':
                props.append(x[index] == right)
            elif op == 'neq':
                props.append(x[index] != right)
            else:
                raise ValueError('The operand is not defined!')
    return props


def in_const_german(df, x, var_name, op, rhs):
    label_name = 'credit'
    dataframe = df.drop(labels=[label_name], axis=1, inplace=False)
    props = []
    for col in dataframe:
        if col == var_name:
            index = dataframe.columns.get_loc(col)
            right = rhs if isinstance(rhs, (int, float)) else rhs[index]
            if op == 'gt':
                props.append(x[index] > right)
            elif op == 'lt':
                props.append(x[index] < right)
            elif op == 'gte':
                props.append(x[index] >= right)
            elif op == 'lte':
                props.append(x[index] <= right)
            elif op == 'eq':
                props.append(x[index] == right)
            elif op == 'neq':
                props.append(x[index] != right)
            else:
                raise ValueError('The operand is not defined!')
    return props


def in_const_domain_german(df, x, x_, ranges, PA):
    label_name = 'credit'
    dataframe = df.drop(labels=[label_name], axis=1, inplace=False)
    props = []
    for col in dataframe:
        index = dataframe.columns.get_loc(col)
        if col in PA:
            props.append(And(x[index] >= ranges[col][0], x[index] <= ranges[col][1]))
            props.append(And(x_[index] >= ranges[col][0], x_[index] <= ranges[col][1]))
        else:
            props.append(And(x[index] >= ranges[col][0], x[index] <= ranges[col][1]))
    return props


def in_const_adult(df, x, var_name, op, rhs):
    label_name = 'income-per-year'
    dataframe = df.drop(labels=[label_name], axis=1, inplace=False)
    props = []
    for col in dataframe:
        if col == var_name:
            index = dataframe.columns.get_loc(col)
            right = rhs if isinstance(rhs, (int, float)) else rhs[index]
            if op == 'gt':
                props.append(x[index] > right)
            elif op == 'lt':
                props.append(x[index] < right)
            elif op == 'gte':
                props.append(x[index] >= right)
            elif op == 'lte':
                props.append(x[index] <= right)
            elif op == 'eq':
                props.append(x[index] == right)
            elif op == 'neq':
                props.append(x[index] != right)
            else:
                raise ValueError('The operand is not defined!')
    return props


def in_const_domain_adult(df, x, x_, ranges, PA):
    label_name = 'income-per-year'
    dataframe = df.drop(labels=[label_name], axis=1, inplace=False)
    props = []
    for col in dataframe:
        index = dataframe.columns.get_loc(col)
        if col in PA:
            props.append(And(x[index] >= ranges[col][0], x[index] <= ranges[col][1]))
            props.append(And(x_[index] >= ranges[col][0], x_[index] <= ranges[col][1]))
        else:
            props.append(And(x[index] >= ranges[col][0], x[index] <= ranges[col][1]))
    return props


def in_const_compas(df, x, var_name, op, rhs):
    label_name = 'score_factor'
    dataframe = df.drop(labels=[label_name], axis=1, inplace=False)
    props = []
    for col in dataframe:
        if col == var_name:
            index = dataframe.columns.get_loc(col)
            right = rhs if isinstance(rhs, (int, float)) else rhs[index]
            if op == 'gt':
                props.append(x[index] > right)
            elif op == 'lt':
                props.append(x[index] < right)
            elif op == 'gte':
                props.append(x[index] >= right)
            elif op == 'lte':
                props.append(x[index] <= right)
            elif op == 'eq':
                props.append(x[index] == right)
            elif op == 'neq':
                props.append(x[index] != right)
            else:
                raise ValueError('The operand is not defined!')
    return props


def in_const_domain_compas(df, x, x_, ranges, PA):
    label_name = 'score_factor'
    dataframe = df.drop(labels=[label_name], axis=1, inplace=False)
    props = []
    for col in dataframe:
        index = dataframe.columns.get_loc(col)
        if col in PA:
            props.append(And(x[index] >= ranges[col][0], x[index] <= ranges[col][1]))
            props.append(And(x_[index] >= ranges[col][0], x_[index] <= ranges[col][1]))
        else:
            props.append(And(x[index] >= ranges[col][0], x[index] <= ranges[col][1]))
    return props


def in_const_diff_adult(df, x, x_, var_name, threshold):
    label_name = 'income-per-year'
    dataframe = df.drop(labels=[label_name], axis=1, inplace=False)
    props = []
    for col in dataframe:
        if col == var_name:
            index = dataframe.columns.get_loc(col)
            props.append(z3Abs(x[index] - x_[index]) <= threshold)
    return props


def in_const_diff_german(df, x, x_, var_name, threshold):
    label_name = 'credit'
    dataframe = df.drop(labels=[label_name], axis=1, inplace=False)
    props = []
    for col in dataframe:
        if col == var_name:
            index = dataframe.columns.get_loc(col)
            props.append(z3Abs(x[index] - x_[index]) <= threshold)
    return props


def in_const_diff_bank(df, x, x_, var_name, threshold):
    label_name = 'y'
    dataframe = df.drop(labels=[label_name], axis=1, inplace=False)
    props = []
    for col in dataframe:
        if col == var_name:
            index = dataframe.columns.get_loc(col)
            props.append(z3Abs(x[index] - x_[index]) <= threshold)
    return props


def in_const_diff(df, x, x_, var_name, op, threshold):
    dataframe = df.drop(df.columns[len(df.columns) - 1], axis=1, inplace=False)
    props = []
    for col in dataframe:
        if col == var_name:
            index = dataframe.columns.get_loc(col)
            if op == 'gt':
                props.append(z3Abs(x[index] - x_[index]) > threshold)
            elif op == 'lt':
                props.append(z3Abs(x[index] - x_[index]) < threshold)
            elif op == 'gte':
                props.append(z3Abs(x[index] - x_[index]) >= threshold)
            elif op == 'lte':
                props.append(z3Abs(x[index] - x_[index]) <= threshold)
            elif op == 'eq':
                props.append(z3Abs(x[index] - x_[index]) == threshold)
            elif op == 'neq':
                props.append(z3Abs(x[index] - x_[index]) != threshold)
            else:
                raise ValueError('The operand is not defined!')
    return props


def in_const_equals(df, x, x_, inequality):
    dataframe = df.drop(df.columns[len(df.columns) - 1], axis=1, inplace=False)
    ignore_indexes = []
    for ineq in inequality:
        for col in dataframe:
            if col.startswith(ineq):
                ignore_indexes.append(dataframe.columns.get_loc(col))
    props = []
    for col in dataframe:
        index = dataframe.columns.get_loc(col)
        if index not in ignore_indexes:
            props.append(x[index] == x_[index])
    return props


def in_const_single(df, x, var_name, op, rhs):
    rhs = rhs.item()
    dataframe = df.drop(df.columns[len(df.columns) - 1], axis=1, inplace=False)
    props = []
    for col in dataframe:
        if col == var_name:
            index = dataframe.columns.get_loc(col)
            right = rhs if isinstance(rhs, (int, float)) else rhs
            if op == 'gt':
                props.append(x[index] > right)
            elif op == 'lt':
                props.append(x[index] < right)
            elif op == 'gte':
                props.append(x[index] >= right)
            elif op == 'lte':
                props.append(x[index] <= right)
            elif op == 'eq':
                props.append(x[index] == right)
            elif op == 'neq':
                props.append(x[index] != right)
            else:
                raise ValueError('The operand is not defined!')
    return props


def cols_starts_with(df, col_name):
    dataframe = df.drop(df.columns[len(df.columns) - 1], axis=1, inplace=False)
    cols = []
    for col in dataframe:
        if col.startswith(col_name):
            cols.append(col)
    return cols


def unique_vals(df, col_name):
    dataframe = df.drop(df.columns[len(df.columns) - 1], axis=1, inplace=False)
    for col in dataframe:
        if col == col_name:
            return dataframe[col].unique()


def parse_z3Model(m):
    ce_x = {}
    ce_x_ = {}
    for d in m:
        variable = str(d)
        val = str(m[d])
        if str(d).startswith('x_'):
            ce_x_[int(variable[2:])] = val
        else:
            ce_x[int(variable[1:])] = val
    ce_x = dict(sorted(ce_x.items()))
    ce_x_ = dict(sorted(ce_x_.items()))
    inp1 = list(ce_x.values())
    inp2 = list(ce_x_.values())
    return inp1, inp2


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def get_y_pred(net, w, b, X_test):
    y_all = []
    for x in X_test:
        y = net(x, w, b)
        res = sigmoid(y)
        y_pred = res > 0.5
