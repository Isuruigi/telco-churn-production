import pandas as pd

def calculate_roi_projections(estimated_revenue_saved: float, total_campaign_cost: float) -> float:
    """Calculates the projected Return on Investment (ROI)."""
    if total_campaign_cost == 0:
        return float('inf')
    return (estimated_revenue_saved - total_campaign_cost) / total_campaign_cost

def analyze_business_impact(clv: float, total_churners: int, saved_churners: float) -> dict:
    """Analyzes the financial impact of churn and retention."""
    cost_of_doing_nothing = total_churners * clv
    revenue_retained = saved_churners * clv
    return {
        'cost_of_doing_nothing': cost_of_doing_nothing,
        'revenue_retained_by_campaign': revenue_retained
    }

def generate_recommendations(best_model_name: str, key_features: list) -> str:
    """Generates strategic recommendations based on model findings."""
    recommendations = f"""
    Based on the performance of the **{best_model_name}** model, we recommend the following actions:

    1.  **Deploy the Predictive Model:** Integrate the model into the CRM system to flag customers with a high probability of churning in real-time.
    
    2.  **Targeted Retention Campaigns:** Launch targeted retention campaigns aimed at the customers identified by the model. The campaign could include special offers, discounts, or personalized communication.
    
    3.  **Focus on Key Drivers of Churn:** The model identified **{', '.join(key_features)}** as the most significant drivers of churn. Business strategies should be developed to address these areas. For example, improving services related to these features or restructuring contracts for at-risk customer segments.
    
    4.  **Continuous Monitoring:** The model's performance and the churn landscape should be continuously monitored to ensure the model remains accurate and the retention strategies are effective.
    """
    return recommendations

def create_executive_summary(project_name: str, best_model_name: str, key_metrics: dict, 
                             business_impact: dict, recommendations: str) -> str:
    """Creates a full, formatted executive summary string."""
    summary = f"""
    # EXECUTIVE SUMMARY: {project_name}

    ## 1. Problem Statement
    This project aimed to predict customer churn for a telecommunications company, enabling proactive retention efforts to reduce revenue loss.

    ## 2. Key Findings & Best Model
    The most effective model developed was the **{best_model_name}**, which achieved the following key results on the test dataset:
    - **ROC AUC Score:** {key_metrics.get('roc_auc', 0):.3f}
    - **Precision (Weighted Avg):** {key_metrics.get('precision', 0):.3f}
    - **Recall (Weighted Avg):** {key_metrics.get('recall', 0):.3f}

    ## 3. Business Impact Analysis
    A simulation based on the model's predictions reveals a significant potential for cost savings and revenue retention.

    - **Projected Cost of Unchecked Churn:** ${business_impact.get('cost_of_doing_nothing', 0):,.2f}
    - **Projected Revenue Retained by Campaign:** ${business_impact.get('revenue_retained_by_campaign', 0):,.2f}
    - **Projected ROI on Retention Campaign:** {business_impact.get('roi', 0):.2%}

    ## 4. Recommendations
    {recommendations}
    
    ## 5. Conclusion
    This project demonstrates the value of a data-driven approach to customer retention. By implementing the proposed recommendations, the business can significantly reduce churn, improve customer loyalty, and secure substantial revenue.
    """
    return summary

if __name__ == '__main__':
    # --- Dummy Data for Demonstration ---
    PROJECT = "Telco Churn Prediction"
    MODEL_NAME = "Tuned Random Forest"
    METRICS = {'roc_auc': 0.86, 'precision': 0.81, 'recall': 0.88}
    KEY_FEATURES = ['Contract_Month-to-month', 'tenure', 'InternetService_Fiber optic']
    
    # Business assumptions
    CLV = 1500.0
    TOTAL_CHURNERS_IN_TEST_SET = 100
    SAVED_CHURNERS_BY_CAMPAIGN = 40.0 # e.g., 40% success rate on 100 true positives
    CAMPAIGN_COST = 5000.0

    # --- Generate Report Components ---
    
    # 1. Analyze Business Impact
    impact_analysis = analyze_business_impact(
        clv=CLV, 
        total_churners=TOTAL_CHURNERS_IN_TEST_SET, 
        saved_churners=SAVED_CHURNERS_BY_CAMPAIGN
    )
    
    # 2. Calculate ROI
    roi = calculate_roi_projections(
        estimated_revenue_saved=impact_analysis['revenue_retained_by_campaign'],
        total_campaign_cost=CAMPAIGN_COST
    )
    impact_analysis['roi'] = roi

    # 3. Generate Recommendations
    recommendations_text = generate_recommendations(MODEL_NAME, KEY_FEATURES)

    # 4. Create the final summary
    full_summary = create_executive_summary(
        project_name=PROJECT,
        best_model_name=MODEL_NAME,
        key_metrics=METRICS,
        business_impact=impact_analysis,
        recommendations=recommendations_text
    )

    # --- Print the Final Summary ---
    print(full_summary)

    # Optionally, save to a file
    # with open("executive_summary.md", "w") as f:
    #     f.write(full_summary)
