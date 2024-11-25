

def analyze_cashflow(user_data):
        monthly_income = user_data['income']
        savings = user_data['savings']
        monthly_expenses = user_data['monthly_expenses']
        debt = user_data['debt']
        
        # Calculate key financial ratios
        savings_ratio = (monthly_income - monthly_expenses) / monthly_income
        debt_to_income = debt / (monthly_income * 12)
        emergency_fund_months = savings / monthly_expenses
        
        return {
            'savings_ratio': savings_ratio,
            'debt_to_income': debt_to_income,
            'emergency_fund_months': emergency_fund_months
        }

def generate_monthly_budget(user_data):
    monthly_income = user_data['income']
    recommended_budget = {
        'Essential expenses': 0.5 * monthly_income,
        'Financial goals': 0.3 * monthly_income,
        'Discretionary': 0.2 * monthly_income
    }
    return recommended_budget