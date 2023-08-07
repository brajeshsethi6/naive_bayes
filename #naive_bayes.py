import pandas as pd
from sklearn.naive_bayes import GaussianNB

class WeatherPredictor:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        self.df_encoded = None
        self.naive_bayes = GaussianNB()

    def preprocess_data(self):
        data_dict = {
            'Weather': self.data['Weather'].tolist(),
            'Play': self.data['Play'].tolist()
        }

        df = pd.DataFrame(data_dict)
        self.df_encoded = pd.get_dummies(df, columns=['Weather'])
        #print(self.df_encoded)
    
    def train_model(self):
        X = self.df_encoded.drop(columns=['Play'])
        y = self.df_encoded['Play']
        self.naive_bayes.fit(X, y)

    def predict_play(self, outlook):
        if self.df_encoded is None:
            raise ValueError("Please preprocess the data and train the model first.")

        all_features = list(self.df_encoded.drop(columns=['Play']).columns)
        input_data = pd.DataFrame([[0] * len(all_features)], columns=all_features)

        if outlook == 'Sunny':
            input_data['Weather_Sunny'] = 1
        elif outlook == 'Rainy':
            input_data['Weather_Rainy'] = 1
        elif outlook == 'Overcast':
            input_data['Weather_Overcast'] = 1
        else:
            raise ValueError("Invalid input. Please enter 'Sunny', 'Rainy', or 'Overcast'.")

        prediction = self.naive_bayes.predict(input_data)
        probabilities = self.naive_bayes.predict_proba(input_data)
        return prediction[0], probabilities[0]

if __name__ == "__main__":
    data_path = 'data.csv'
    weather_predictor = WeatherPredictor(data_path)
    weather_predictor.preprocess_data()
    weather_predictor.train_model()

    user_input = input("Enter 'Sunny' or 'Rainy' or 'Overcast': ")
    predicted_play, probabilities = weather_predictor.predict_play(user_input)

    print(f"Predicted 'Play': {predicted_play}")
    print(f"Probability of 'Yes': {probabilities[0]}, Probability of 'No': {probabilities[1]}")
