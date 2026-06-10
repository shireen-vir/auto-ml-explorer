class AutoMlExplorer:
    """
    A class used to explore and automate machine learning workflows.

    Attributes:
    ----------
    data : DataFrame
        Input data for machine learning models

    Methods:
    -------
    load_data(file_path)
        Loads data from a specified file path
    train_model()
        Trains a machine learning model using the loaded data
    evaluate_model()
        Evaluates the performance of the trained model
    """

    def __init__(self, data=None):
        self.data = data

    def load_data(self, file_path):
        import pandas as pd
        self.data = pd.read_csv(file_path)

    def train_model(self):
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score
        X = self.data.drop('target', axis=1)
        y = self.data['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return accuracy_score(y_test, y_pred)

    def evaluate_model(self, accuracy):
        print(f"Model accuracy: {accuracy}")

def main():
    explorer = AutoMlExplorer()
    explorer.load_data('data.csv')
    accuracy = explorer.train_model()
    explorer.evaluate_model(accuracy)

if __name__ == "__main__":
    main()