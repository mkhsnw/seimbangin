import numpy as np

def predict_market(models,data,window_size):
    predictions = {}
    for indicator,model in models.items():
        indicator_data = data[data['Description'] == indicator]
        series = indicator_data['Value'].values
        
        input_data = np.array([series[-window_size:]])
        input_data = input_data.reshape((input_data.shape[0],input_data.shape[1],1))
        prediction = model.predict(input_data)
        predictions[indicator] = prediction[0][0]
    return predictions

def define_market_condition(predictions,data):
    conditions = {}
    for indicator,prediction in predictions.items():
        indicator_data = data[data['Description'] == indicator]
        series = indicator_data['Value'].values
        mean = np.mean(series)
        std = np.std(series)
        
        bullish_threshold = mean + std
        bearish_threshold = mean - std
        
        if prediction > bullish_threshold:
            conditions[indicator] = 'Bullish'
        elif prediction < bearish_threshold:
            conditions[indicator] = 'Bearish'
        else:
            conditions[indicator] = 'Neutral'
    return conditions



