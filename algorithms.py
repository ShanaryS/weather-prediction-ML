import os.path
import joblib
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier


@dataclass
class Models:
    """Different models for analyzing data"""

    # Defines the model with the data being used
    model: any = None
    model_name: str = ''
    X: pd.DataFrame = None
    y: pd.DataFrame = None
    train: bool = False

    # Test data used to compare accuracy with
    X_test: any = None
    y_test: any = None

    # Stores user input
    user_datetime: str = None
    user_temperature: str = None
    user_pressure: str = None
    user_humidity: str = None
    user_wind_speed: str = None
    user_wind_direction: str = None

    def set_X_y(self, x, y) -> None:
        """Sets the data for X and y"""
        self.X, self.y = x, y

    def get_model_and_test_data(self) -> tuple:
        """Gets the model and test data to be used for analysis"""
        return self.model, self.X_test

    def decision_tree_classifier(self) -> None:
        """ML model of Decision Tree Classifier"""

        self.model_name = 'Decision Tree Classifier'
        X_train, self.X_test, y_train, self.y_test = split_data(self.X, self.y)

        if self.train:
            self.model = DecisionTreeClassifier().fit(X_train, y_train)
        else:
            self.model = joblib.load(os.path.join(
                'models',
                f"{self.model_name.lower().replace(' ', '_')}.joblib"))

    def gaussian_nb(self) -> None:
        """ML model of GaussianNB"""

        self.model_name = 'Gaussian Naive Bayes'
        X_train, self.X_test, y_train, self.y_test = split_data(self.X, self.y)

        if self.train:
            self.model = GaussianNB().fit(X_train, y_train)
        else:
            self.model = joblib.load(os.path.join(
                'models',
                f"{self.model_name.lower().replace(' ', '_')}.joblib"))


def get_data() -> tuple:
    """Gets the dataset and returns 'X' and 'y' for analysis"""

    Numerics = LabelEncoder()

    # Independent variables
    temperature = pd.read_csv(os.path.join('dataset', 'temperature.csv'))
    pressure = pd.read_csv(os.path.join('dataset', 'pressure.csv'))
    humidity = pd.read_csv(os.path.join('dataset', 'humidity.csv'))
    wind_speed = pd.read_csv(os.path.join('dataset', 'wind_speed.csv'))
    wind_direction = pd.read_csv(os.path.join('dataset', 'wind_direction.csv'))

    # Dependent variable
    weather_description = pd.read_csv(
        os.path.join('dataset', 'weather_description.csv'))

    # Converting datetime to day of year and hour of day
    # TODO Look into this more. Maybe account for changing years? Maybe not
    #  since you want to use previous years data a data point for new
    #  datetime that's being entered.
    temperature['datetime'] = temperature['datetime'].apply(
        lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S').strftime('%j-%H'))

    # NYC data
    nyc = temperature.copy()[['datetime', 'New York']]
    nyc.rename(columns={'datetime': 'Datetime', 'New York': 'Temperature'},
               inplace=True)
    nyc['Pressure'] = pressure['New York']
    nyc['Humidity'] = humidity['New York']
    # nyc['Wind Speed'] = wind_speed['New York']
    # nyc['Wind Direction'] = wind_direction['New York']
    nyc['Weather Description'] = weather_description['New York']
    nyc.dropna(inplace=True)
    nyc['Datetime'] = Numerics.fit_transform(nyc['Datetime'])

    # TODO remove Datetime?
    X = nyc.drop(columns=['Weather Description'])
    y = nyc['Weather Description']
    # y = Numerics.fit_transform(nyc['Weather Description'])

    return X, y


def split_data(x, y) -> tuple:
    """Dynamically splits data into training and testing"""
    return train_test_split(x, y, test_size=0.2)


def get_accuracy(models: Models, show=False) -> float:
    """Returns the metrics of the model"""

    predictions = models.model.predict(models.X_test)

    score = accuracy_score(models.y_test, predictions)

    if show:
        print(f"{models.model_name} ~ Accuracy: {round(score*100, 2)}")

    return score


def prediction(models: Models, show=False) -> float:
    """Returns a number from the input date"""

    Numerics = LabelEncoder()

    user_datetime = datetime.strptime(
        models.user_datetime, '%m/%d/%Y %H').strftime('%j-%H')
    user_datetime = Numerics.fit_transform(user_datetime)

    expected_condition = models.model.predict([
        user_datetime, models.user_temperature, models.user_pressure,
        models.user_humidity
    ])[0]

    if show:
        print(f"Expected condition in NYC on {models.user_datetime}: "
              f"{expected_condition.capitalize()} (Using {models.model_name})")

    return expected_condition


def export_model(models: Models) -> None:
    """Saves model to disk"""
    joblib.dump(models.model, os.path.join(
        'models',
        f"{models.model_name.lower().replace(' ', '_')}.joblib")
    )


def export_graphviz_dot(models: Models) -> None:
    """Saves logic visualizer to disk"""
    tree.export_graphviz(
        models.model,
        out_file=f"{models.model_name.lower().replace(' ', '_')}.dot",
        feature_names=['date'],
        class_names=sorted(models.y.unique()),
        label='all',
        rounded=True,
        filled=True
    )


def main() -> None:
    """Main function"""

    models = Models()
    X, y = get_data()
    models.set_X_y(X, y)

    '''Controls training and saving models'''
    print_accuracy_or_prediction = True
    models.train = True
    model_export = True  # Requires model.train = True
    model_dot_export = False

    # Set ML model and get accuracy with option to print
    models.decision_tree_classifier()
    if models.train:
        _ = get_accuracy(models, show=print_accuracy_or_prediction)
    else:
        models.user_datetime = input('Enter datetime (mm/dd/YYYY H): ')
        models.user_temperature = input('Enter temperature (\u00b0F): ')
        models.user_pressure = input('Enter pressure (hPa): ')
        models.user_humidity = input('Enter humidity (0-100): ')
        prediction(models, show=print_accuracy_or_prediction)
    if model_export:
        export_model(models)
    if model_dot_export:
        export_graphviz_dot(models)


if __name__ == '__main__':
    main()
