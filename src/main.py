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
    # df = df.drop("Gender", axis=1)
    # df = df.drop("University Degree", axis=1)
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
    df['Gender'] = df['Gender'].fillna('unknown')
    df['University Degree'] = df['University Degree'].fillna('No')

    return df


def HandleMissingData(df):
    df = FillInMissingData(df)
    df = DropHeadings(df)
    df.describe()
    return df


def MapColVarToModelInputs(training_df, test_df, colVar):
    colVar_columns = training_df[colVar].unique()
    colVar_columns.sort()

    count = 0
    array = [0] * len(colVar_columns)

    zero_array = np.zeros((len(test_df[colVar]), len(colVar_columns)))
    outputDf = pd.DataFrame(data=zero_array, columns=colVar_columns)

    for i in test_df[colVar]:
        outputDf.loc[count] = array
        if i in colVar_columns:
            # Write output
            outputDf.loc[count, i] = 1

        count = count + 1

    return outputDf


def PrintOutNaN(df):
    null_counts = df.isnull().sum()
    print("Number of null values in each column:\n{}".format(null_counts))


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
