# Weather Prediction using Machine Learning

Predict the weather conditions of a day in NYC or other large cities using various machine learning algorithms. Observe the accuracy of each model as well as use any to predict the weather conditions with a few input values. Prediction is split into 5 categories:

* Sky - Clear vs Cloudy
* Rain
* Thunderstorm
* Snow
* Fog - Mist and Haze included

This split allows the models to be properly fitted to a single feature rather than overextending themselves too thin.

Models were trained on data from: https://www.kaggle.com/selfishgene/historical-hourly-weather-data?select=weather_description.csv.
Model is currently trained on NYC but can be easily swapped with other cities from the dataset.

***

Accuracy of some of the models used:             |  Example of weather pridiction:
:-------------------------:|:-------------------------:
![Demo #1](https://user-images.githubusercontent.com/86130442/136312320-af8db059-1023-496f-a38e-acb7b1fb61d2.png)  |  ![Demo #2](https://user-images.githubusercontent.com/86130442/136312326-aa2d2383-c938-4b73-89df-b0ed73113e10.png)

## Installation

Clone this repo and cd into it:

```bash
git clone https://github.com/ShanaryS/weather-prediction-ML.git
cd weather-prediction-ML
```

Create and activate your virtual environment:

* Windows:
```bash
virtualenv env
.\env\Scripts\activate
```

* MacOS/Linux:
```bash
virtualenv --no-site-packages env
source env/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

# Usage

* Train Models:
```bash
python run_training
```

* Predict Weather:
```bash
python run_prediction
```

## License
[MIT](https://github.com/ShanaryS/algorithm-visualizer/blob/main/LICENSE)
