from dataclasses import dataclass
from typing import Callable
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from sklearn.tree import DecisionTreeRegressor


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

    def set_X_y(self, x, y) -> None:
        """Sets the data for X and y"""
        self.X, self.y = x, y

    def get_model_and_test_data(self) -> tuple:
        """Gets the model and test data to be used for analysis"""
        return self.model, self.X_test

    def decision_tree_regressor(self) -> None:
        """ML model of decision tree regression"""

        self.model_name = 'Decision Tree Regression'
        self.refresh_model_data = self.decision_tree_regressor

        X_train, self.X_test, y_train, self.y_test = split_data(self.X, self.y)
        self.model = DecisionTreeRegressor().fit(X_train, y_train)


def get_data() -> tuple:
    """Gets the dataset and returns 'X' and 'y' for analysis"""

    weather = pd.read_csv('weather_nyc.csv')

    # Cleaning data
    weather['date'].replace('-', '/', regex=True, inplace=True)
    weather['date'] = weather['date'].apply(
        lambda x: datetime.strptime(x, '%d/%m/%Y').strftime('%j'))
    weather['precipitation'].replace('T', '0.0025', inplace=True)
    weather['snow_fall'].replace('T', '0.025', inplace=True)
    weather['snow_depth'].replace('T', '0.25', inplace=True)

    # Defining independent and dependent variables
    X = weather.drop(columns=['average_temperature'])
    y = weather['average_temperature']

    return X, y


def split_data(x, y) -> tuple:
    """Dynamically splits data into training and testing"""
    return train_test_split(x, y, test_size=0.2)


def predict(model: any, x_test) -> list:
    """Returns predictions from the test data compared against model"""
    return model.predict(x_test)


def get_accuracy(models: Models, show=False) -> tuple:
    """Returns the metrics of the model"""

    predictions = predict(models.model, models.X_test)
    r2 = r2_score(models.y_test, predictions)
    mean_squared = mean_squared_error(models.y_test, predictions)

    if show:
        print(f"{models.model_name} ~ "
              f"R^2: {round(r2, 3)} - "
              f"Mean Squared Error: {round(mean_squared, 3)}")

    return r2, mean_squared


def main() -> None:
    """Main function"""

    models = Models()
    X, y = get_data()
    models.set_X_y(X, y)

    # Set ML model and get accuracy with option to print
    models.decision_tree_regressor()
    _ = get_accuracy(models, show=True)
    print('...')


if __name__ == '__main__':
    main()
