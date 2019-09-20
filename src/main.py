import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


def ReadInData(file):
    dataset = pd.read_csv(file)
    return dataset


def main():
    print('Starting Assignment 1')
    trainingData = ReadInData("../data/tcd ml 2019-20 income prediction training (with labels).csv")
    print(trainingData.shape)
    print(trainingData.describe())

## Might not need this line here. Just a precaution for NaN.
    if trainingData.isnull().any:
        trainingData = trainingData.fillna(method='ffill')

    print(trainingData.shape)
    print(trainingData.iloc[820])


if __name__ == '__main__':
    main()
