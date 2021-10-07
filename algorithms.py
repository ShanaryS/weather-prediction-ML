import os.path
import joblib
from dataclasses import dataclass
from enum import Enum, unique
from datetime import datetime
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


@unique
class Condition(Enum):
    """Different conditions of the weather that the models can predict"""

    CLEAR_SKIES = 'Clear Skies'
    RAIN = 'Rain'
    THUNDERSTORM = 'Thunderstorm'
    SNOW = 'Snow'
    FOG = 'Fog'


@dataclass
class Models:
    """Different models for analyzing data"""

    # Defines the model with the data being used
    condition: str
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
    user_wind_direction: str = None

    def set_X_y(self, x_y) -> None:
        """Sets the data for X and y"""
        self.X, self.y = x_y

    def get_model_and_test_data(self) -> tuple:
        """Gets the model and test data to be used for analysis"""
        return self.model, self.X_test

    def random_forest_classifier(self) -> None:
        """ML model of Random Forest Classifier"""

        self.model_name = 'Random Forest Classifier'
        X_train, self.X_test, y_train, self.y_test = split_data(self.X, self.y)

        if self.train:
            self.model = RandomForestClassifier().fit(X_train, y_train)
        else:
            name = self.model_name.lower().replace(' ', '_')
            name += '_' + self.condition.lower().replace(' ', '_')
            self.model = joblib.load(os.path.join('models', f"{name}.joblib"))

    def k_neighbours_classifier(self) -> None:
        """ML model of K Neighbours Classifier"""

        self.model_name = 'K Neighbours Classifier'
        X_train, self.X_test, y_train, self.y_test = split_data(self.X, self.y)

        if self.train:
            self.model = KNeighborsClassifier().fit(X_train, y_train)
        else:
            name = self.model_name.lower().replace(' ', '_')
            name += '_' + self.condition.lower().replace(' ', '_')
            self.model = joblib.load(os.path.join('models', f"{name}.joblib"))

    def decision_tree_classifier(self) -> None:
        """ML model of Decision Tree Classifier"""

        self.model_name = 'Decision Tree Classifier'
        X_train, self.X_test, y_train, self.y_test = split_data(self.X, self.y)

        if self.train:
            self.model = DecisionTreeClassifier().fit(X_train, y_train)
        else:
            name = self.model_name.lower().replace(' ', '_')
            name += '_' + self.condition.lower().replace(' ', '_')
            self.model = joblib.load(os.path.join('models', f"{name}.joblib"))

    def gaussian_nb(self) -> None:
        """ML model of GaussianNB"""

        self.model_name = 'Gaussian Naive Bayes'
        X_train, self.X_test, y_train, self.y_test = split_data(self.X, self.y)

        if self.train:
            self.model = GaussianNB().fit(X_train, y_train)
        else:
            name = self.model_name.lower().replace(' ', '_')
            name += '_' + self.condition.lower().replace(' ', '_')
            self.model = joblib.load(os.path.join('models', f"{name}.joblib"))


def get_data(models: Models) -> tuple:
    """Gets the dataset and returns 'X' and 'y' for analysis"""

    Numerics = LabelEncoder()

    # Independent variables
    temperature = pd.read_csv(os.path.join('dataset', 'temperature.csv'))
    pressure = pd.read_csv(os.path.join('dataset', 'pressure.csv'))
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
    nyc['Wind Direction'] = wind_direction['New York']
    nyc['Weather Description'] = weather_description['New York']
    nyc.dropna(inplace=True)
    nyc['Datetime'] = Numerics.fit_transform(nyc['Datetime'])

    X = nyc.drop(columns=['Weather Description'])
    y = nyc['Weather Description'].copy()

    if models.condition == Condition.CLEAR_SKIES.value:
        y[~y.str.contains('sky is clear')] = 'cloudy'
    elif models.condition == Condition.RAIN.value:
        y[~y.isin({'light rain', 'moderate rain', 'shower rain',
                   'heavy intensity rain', 'very heavy rain', 'freezing rain',
                   'light rain and snow', 'thunderstorm with light rain',
                   'thunderstorm with rain',
                   'thunderstorm with heavy rain',
                   'proximity thunderstorm with rain',
                   'light intensity shower rain',
                   'drizzle', 'light intensity drizzle',
                   'heavy intensity drizzle', 'thunderstorm with light drizzle',
                   'proximity thunderstorm with drizzle'
                   })] = 'no rain'
    elif models.condition == Condition.THUNDERSTORM.value:
        y[~y.isin({'thunderstorm', 'proximity thunderstorm',
                   'thunderstorm with rain', 'thunderstorm with heavy rain',
                   'thunderstorm with light rain',
                   'thunderstorm with light drizzle',
                   'proximity thunderstorm with rain',
                   'proximity thunderstorm with drizzle', 'heavy thunderstorm'}
                  )] = 'no thunderstorms'
    elif models.condition == Condition.SNOW.value:
        y[~y.isin({'snow', 'light snow', 'heavy snow', 'light rain and snow'}
                  )] = 'no snow'
    elif models.condition == Condition.FOG.value:
        y[~y.isin({'fog', 'mist', 'haze', 'smoke'})] = 'no fog'

    return X, y


def split_data(x, y) -> tuple:
    """Dynamically splits data into training and testing"""
    return train_test_split(x, y, test_size=0.2)


def get_accuracy(models: Models, show=False) -> float:
    """Returns the metrics of the model"""

    predictions = models.model.predict(models.X_test)

    score = accuracy_score(models.y_test, predictions)

    if show:
        print(f"{models.model_name} ~ {models.condition} Accuracy: "
              f"{round(score*100, 2)}")

    return score


def prediction(models: Models, show=False) -> float:
    """Returns a number from the input date"""

    Numerics = LabelEncoder()

    user_datetime = datetime.strptime(
        models.user_datetime, '%m/%d/%Y %H').strftime('%j-%H')
    user_datetime = Numerics.fit_transform(user_datetime)

    expected_condition = models.model.predict([
        user_datetime, models.user_temperature, models.user_pressure,
        models.user_wind_direction
    ])[0]

    if show:
        print(f"{models.condition} in NYC on {models.user_datetime}: "
              f"{expected_condition.capitalize()} (Using {models.model_name})")

    return expected_condition


def export_model(models: Models) -> None:
    """Saves model to disk"""

    name = models.model_name.lower().replace(' ', '_')
    name += '_' + models.condition.lower().replace(' ', '_')
    joblib.dump(models.model, os.path.join('models', f"{name}.joblib"))


def export_graphviz_dot(models: Models) -> None:
    """Saves logic visualizer to disk"""
    tree.export_graphviz(
        models.model,
        out_file=f"{models.model_name.lower().replace(' ', '_')}.dot",
        feature_names=['datetime', 'Temperature', 'Pressure', 'Wind Direction'],
        class_names=sorted(models.y.unique()),
        label='all',
        rounded=True,
        filled=True
    )


def run(models: Models, show=False, train=False, export=False) -> None:
    """Runs the inputted model"""

    # Set ML model and get accuracy with option to print
    models.train = train
    models.random_forest_classifier()
    if train:
        _ = get_accuracy(models, show=show)
    else:
        models.user_datetime = input('Enter datetime (mm/dd/YYYY H): ')
        models.user_temperature = input('Enter temperature (\u00b0F): ')
        models.user_pressure = input('Enter pressure (hPa): ')
        models.user_wind_direction = input('Enter wind direction (0-360): ')
        prediction(models, show=show)
    if export:
        export_model(models)  # Requires train = True
        # export_graphviz_dot(models)


def main() -> None:
    """Main function"""

    show = True
    train = True
    export = True

    models_clear = Models(condition=Condition.CLEAR_SKIES.value)
    models_rain = Models(condition=Condition.RAIN.value)
    models_thunderstorm = Models(condition=Condition.THUNDERSTORM.value)
    models_snow = Models(condition=Condition.SNOW.value)
    models_fog = Models(condition=Condition.FOG.value)

    print("\nTraining models...\n")

    models_clear.set_X_y(get_data(models_clear))
    models_rain.set_X_y(get_data(models_rain))
    models_thunderstorm.set_X_y(get_data(models_thunderstorm))
    models_snow.set_X_y(get_data(models_snow))
    models_fog.set_X_y(get_data(models_fog))

    print("Running models...\n")

    run(models_clear, show=show, train=train, export=export)
    run(models_rain, show=show, train=train, export=export)
    run(models_thunderstorm, show=show, train=train, export=export)
    run(models_snow, show=show, train=train, export=export)
    run(models_fog, show=show, train=train, export=export)


if __name__ == '__main__':
    main()
