import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


def ReadInData(file):
    dataset = pd.read_csv(file)
    return dataset


def DropHeadings(df):
    df = df.drop("Gender", axis=1)
    df = df.drop("University Degree", axis=1)
    df = df.drop("Hair Color", axis=1)
    df = df.drop("Instance", axis=1)
    return df

def FillInMissingData(df):
    averageAge = int(df['Age'].mean())
    averageRecord = (df['Year of Record'].mean())

    print(str(averageAge))
    print(str(averageRecord))

    ## Fill in Missing Data with average
    df['Age'] = df['Age'].fillna(averageAge)
    df['Year of Record'] = df['Year of Record'].fillna(round(averageRecord))
    df['Profession'] = df['Profession'].fillna('No')

    return df


def HandleMissingData(df):
    df = FillInMissingData(df)
    df = DropHeadings(df)
    df.describe()
    return df

def PrepTestForModel(training_df, test_df):
    countries = training_df['Country'].unique()
    professions = training_df['Profession'].unique()

    for i in test_df['Country']:
        if i in countries:
            # Write output
            print(i)


def main():
    print('Starting Assignment 1')
    trainingData = ReadInData("../data/tcd ml 2019-20 income prediction training (with labels).csv")
    trainingData = DropHeadings(trainingData)
    print(trainingData.shape)
    print(trainingData.describe())

## Might not need this line here. Just a precaution for NaN.
    if trainingData.isnull().any:
        trainingData = trainingData.fillna(method='ffill')

    print(trainingData.shape)
    print(trainingData.iloc[820])


if __name__ == '__main__':
    main()
