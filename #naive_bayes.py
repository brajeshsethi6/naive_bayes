import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

data = pd.read_csv('data.csv')

data_dict = {
    'Weather': data['Weather'].tolist(),
    'Play': data['Play'].tolist()
}

df = pd.DataFrame(data_dict)
df_encoded = pd.get_dummies(df, columns=['Weather'])

X = df_encoded.drop(columns=['Play'])
y = df_encoded['Play']

naive_bayes = GaussianNB()
naive_bayes.fit(X, y)

def predict_play(outlook):
    all_features = list(X.columns)
    input_data = pd.DataFrame([[0] * len(all_features)], columns=all_features)

    if outlook == 'Sunny':
        input_data['Weather_Sunny'] = 1
    elif outlook == 'Rainy':
        input_data['Weather_Rainy'] = 1
    elif outlook == 'Overcast':
        input_data['Weather_Overcast'] = 1
    else:
        raise ValueError("Invalid input. Please enter 'Sunny', 'Rainy', or 'Overcast'.")

    prediction = naive_bayes.predict(input_data)
    probabilities = naive_bayes.predict_proba(input_data)
    return prediction[0], probabilities[0]

user_input = input("Enter 'Sunny' or 'Rainy' or 'Overcast': ")


predicted_play, probabilities = predict_play(user_input)

print(type(predicted_play))

# print result 
print(f"Predicted 'Play': {predicted_play}")
print(f"Probability of 'Yes': {probabilities[0]}, Probability of 'No': {probabilities[1]}")