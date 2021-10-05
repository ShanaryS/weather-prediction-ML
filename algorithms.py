from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from typing import Callable


@dataclass
class Models:
    """Different models for analyzing data"""

    # Defines the model with the data being used
    model: any = None
    model_name: str = ''
    X: pd.DataFrame = None
    y: pd.DataFrame = None

    # Test data used to compare accuracy with
    X_test: any = None
    y_test: any = None

    # Stores last used model to refresh. Each model calls after initializing.
    refresh_model_data: Callable = None

    def get_model_and_test_data(self) -> tuple:
        """Gets the model and test data to be used for analysis"""
        return self.model, self.X_test, self.y_test

    def decision_tree_classifier(self) -> None:
        """ML model of decision tree classifier"""

        self.model_name = 'Decision Tree Classifier'
        self.refresh_model_data = self.decision_tree_classifier

        X_train, self.X_test, y_train, self.y_test = split_data(self.X, self.y)
        self.model = DecisionTreeClassifier().fit(X_train, y_train)


def data() -> tuple:
    """Gets the dataset and returns 'X' and 'y' for analysis"""

    music_data = pd.read_csv('music.csv')
    X = music_data.drop(columns=['genre'])
    y = music_data['genre']

    return X, y


def split_data(x, y) -> tuple:
    """Dynamically splits data into training and testing"""
    return train_test_split(x, y, test_size=0.2)


def predict(model, x_test) -> list:
    """Returns predictions from the test data compared against model"""
    return model.predict(x_test)


def get_accuracy_score(y_test, predictions) -> float:
    """Gets the accuracy score of the predictions from a model"""
    return accuracy_score(y_test, predictions)


def get_final_accuracy(models: Models, num_loops: int, show=False) -> float:
    """Runs a model multiple times to get it's final accuracy"""

    scores = []
    for _ in range(num_loops):
        models.refresh_model_data()
        model, X_test, y_test = models.get_model_and_test_data()
        predictions = predict(model, X_test)
        score = get_accuracy_score(y_test, predictions)
        scores.append(score)

    if show:
        final_accuracy = float(np.mean(scores)) * 100
        print(f"{models.model_name}: {final_accuracy}%")

    return float(np.mean(scores)) * 100


def main(num_loops=1000) -> None:
    """Main loop"""
    models = Models()
    models.X, models.y = data()

    # Set ML model and get final accuracy with option to print
    models.decision_tree_classifier()
    _ = get_final_accuracy(models, num_loops=num_loops, show=True)


if __name__ == '__main__':
    main()
