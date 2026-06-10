import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def load_data(file_path):
    """
    Load dataset from a csv file.
    
    Args:
        file_path (str): Path to the csv file.
    
    Returns:
        pandas.DataFrame: Loaded dataset.
    """
    return pd.read_csv(file_path)

def split_data(data, target_variable):
    """
    Split dataset into features and target.
    
    Args:
        data (pandas.DataFrame): Loaded dataset.
        target_variable (str): Name of the target variable.
    
    Returns:
        tuple: Features and target.
    """
    X = data.drop(target_variable, axis=1)
    y = data[target_variable]
    return X, y

def split_train_test(X, y, test_size=0.2):
    """
    Split data into training and testing sets.
    
    Args:
        X (pandas.DataFrame): Features.
        y (pandas.Series): Target.
        test_size (float, optional): Proportion of data for testing. Defaults to 0.2.
    
    Returns:
        tuple: Training and testing sets.
    """
    return train_test_split(X, y, test_size=test_size, random_state=42)

def main():
    """
    Main function of the auto-ml-explorer tool.
    """
    file_path = 'data.csv'
    target_variable = 'target'
    data = load_data(file_path)
    X, y = split_data(data, target_variable)
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    print('Data loaded and split successfully')

if __name__ == '__main__':
    main()