from fastapi import FastAPI,HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import pandas as pd
import uvicorn
from tensorflow.keras import backend as K
from tensorflow.keras.saving import register_keras_serializable
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoConfig

# @register_keras_serializable()
# def mse(y_true, y_pred):
#     return K.mean(K.square(y_pred - y_true), axis=-1)

app = FastAPI(
    title="Seimbangin financial advisor API",
    description="API for forecasting Indonesian market conditions and providing market analysis and give financial personal recommendations to users",
    version="1.0.0"
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

dataset = pd.read_csv('./combined_dataset.csv')

def load_model_forecasting():
    models = {}
    indicators = ['Rupiah','Saham dan Modal lainnya','TRANSPORTASI','makanan','rumahtangga','perumahan']
    for indicator in indicators:
        models[indicator] = tf.keras.models.load_model(f"model_filtered/model_{indicator}.h5")
    return models

def load_model_advisor():
    model_path = "./Llama-3.2-1B-personal-finance"
    config = AutoConfig.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path,config=config)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
    return generator

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

def analyze_cashflow(user_data):
        monthly_income = user_data['income']
        savings = user_data['savings']
        monthly_expenses = user_data['monthly_expenses']
        debt = user_data['debt']
        goals = user_data['financial_goals']
        risk_tolerance = user_data['risk_tolerance']
        
        # Calculate key financial ratios
        savings_ratio = (monthly_income - monthly_expenses) / monthly_income
        debt_to_income = debt / (monthly_income * 12)
        emergency_fund_months = savings / monthly_expenses
        
        return {
            'savings_ratio': savings_ratio,
            'debt_to_income': debt_to_income,
            'emergency_fund_months': emergency_fund_months
        }

def get_recommendations(user_data, market_conditions):
    analysis = analyze_cashflow(user_data)
    recommendations = []
        
    # Emergency fund recommendations
    if analysis['emergency_fund_months'] < 6:
        recommendations.append({
            'category': 'Emergency Fund',
            'action': f'Increase emergency fund from {analysis["emergency_fund_months"]:.1f} months to 6 months',
            'priority': 'High'
        })
    
    # Debt management
    if analysis['debt_to_income'] > 0.4:
        recommendations.append({
            'category': 'Debt',
            'action': 'Focus on debt reduction - consider debt snowball/avalanche method',
            'priority': 'High'
        })
    
    # Savings optimization
    if analysis['savings_ratio'] < 0.2:
        recommendations.append({
            'category': 'Savings',
            'action': 'Increase monthly savings to at least 20% of income',
            'priority': 'Medium'
        })
    
    # Investment allocation
    # risk_profile = risk_profiles[user_data['risk_tolerance']]
    # investment_rec = {
    #     'category': 'Investment',
    #     'action': f"Recommended portfolio allocation:\n" + \
    #              f"- Stocks: {risk_profile['stocks']*100}%\n" + \
    #              f"- Bonds: {risk_profile['bonds']*100}%\n" + \
    #              f"- Cash: {risk_profile['cash']*100}%",
    #     'priority': 'Medium'
    # }
    # recommendations.append(investment_rec)
    
    # Market-based adjustments
    for indicator, condition in market_conditions.items():
        if condition == 'Bullish':
            recommendations.append({
                'category': 'Market Strategy',
                'action': f'Consider increasing exposure to {indicator} as it is currently bullish',
                'priority': 'Medium'
            })
        elif condition == 'Bearish':
            recommendations.append({
                'category': 'Market Strategy',
                'action': f'Consider reducing exposure to {indicator} as it is currently bearish',
                'priority': 'Medium'
            })
    
    # Goal-specific recommendations
    for goal in user_data['financial_goals']:
        if goal.lower().startswith('retirement'):
            recommendations.append({
                'category': 'Retirement',
                'action': 'Maximize retirement account contributions',
                'priority': 'High'
            })
    return recommendations

def generate_monthly_budget(user_data):
    monthly_income = user_data['income']
    recommended_budget = {
        'Essential expenses': 0.5 * monthly_income,
        'Financial goals': 0.3 * monthly_income,
        'Discretionary': 0.2 * monthly_income
    }
    return recommended_budget

def get_financial_advice(context):
  prompt = f"### Context:\n{context}\n\n### Response:\n"
  generator = load_model_advisor()
  generated_text = generator(prompt, max_length=512, num_return_sequences=1)
  response = generated_text[0]['generated_text'].split("### Response:\n")[1].strip()
  return response


@app.get("/get_advice")
def get_advice():
    context = "I am a 25-year-old software engineer with a monthly income of $5000. I have $2000 in savings and $1000 in debt. My monthly expenses are $3000 and I am looking to invest in the stock market. What should I do?"
    advice = get_financial_advice(context)
    return advice

if __name__ == '__main__':
    uvicorn.run(app,host='0.0.0.0',port=8000)
