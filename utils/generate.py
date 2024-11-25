from cashflow import analyze_cashflow
from model import load_model_advisor
from transformers import pipeline

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

def get_financial_advice(context):
  prompt = f"### Context:\n{context}\n\n### Response:\n"
  model,tokenizer = load_model_advisor()
  generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
  generated_text = generator(prompt, max_length=512, num_return_sequences=1)
  response = generated_text[0]['generated_text'].split("### Response:\n")[1].strip()
  return response