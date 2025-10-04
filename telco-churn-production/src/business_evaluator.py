import pandas as pd
import numpy as np

class BusinessEvaluator:
    """Performs business-focused evaluation of a churn prediction model."""

    def __init__(self, avg_monthly_revenue, avg_customer_lifespan_months, 
                 customer_acquisition_cost, retention_campaign_cost_per_customer):
        """
        Initializes the BusinessEvaluator with key business assumptions.

        Args:
            avg_monthly_revenue (float): Average monthly revenue per customer.
            avg_customer_lifespan_months (int): Average customer lifespan in months.
            customer_acquisition_cost (float): Cost to acquire a new customer.
            retention_campaign_cost_per_customer (float): Cost of a retention campaign for one customer.
        """
        self.avg_monthly_revenue = avg_monthly_revenue
        self.avg_customer_lifespan_months = avg_customer_lifespan_months
        self.customer_acquisition_cost = customer_acquisition_cost
        self.retention_campaign_cost_per_customer = retention_campaign_cost_per_customer
        self.clv = self.calculate_customer_lifetime_value()

    def calculate_customer_lifetime_value(self) -> float:
        """Calculates the Customer Lifetime Value (CLV)."""
        # A simplified CLV calculation
        return (self.avg_monthly_revenue * self.avg_customer_lifespan_months) - self.customer_acquisition_cost

    def calculate_churn_cost(self, num_churned_customers: int) -> float:
        """Calculates the total cost of churn based on lost CLV."""
        return num_churned_customers * self.clv

    def calculate_retention_roi(self, num_retained_customers: int) -> float:
        """Calculates the ROI of retention campaigns."""
        total_retention_cost = num_retained_customers * self.retention_campaign_cost_per_customer
        revenue_from_retained = num_retained_customers * self.clv
        
        if total_retention_cost == 0:
            return np.inf
        
        roi = (revenue_from_retained - total_retention_cost) / total_retention_cost
        return roi

    def analyze_campaign_effectiveness(self, y_true, y_pred, campaign_success_rate=0.5) -> dict:
        """
        Analyzes the potential effectiveness of a targeted retention campaign.

        Args:
            y_true (pd.Series): True churn labels (1 for churn, 0 for no churn).
            y_pred (pd.Series): Predicted churn labels.
            campaign_success_rate (float): The assumed success rate of the retention campaign.

        Returns:
            dict: A dictionary summarizing the campaign analysis.
        """
        targeted_for_retention = (y_pred == 1)
        actually_churned = (y_true == 1)
        
        true_positives = (targeted_for_retention & actually_churned).sum()
        
        customers_to_target = targeted_for_retention.sum()
        customers_saved_by_campaign = true_positives * campaign_success_rate
        
        total_campaign_cost = customers_to_target * self.retention_campaign_cost_per_customer
        revenue_saved = customers_saved_by_campaign * self.clv
        
        roi = (revenue_saved - total_campaign_cost) / total_campaign_cost if total_campaign_cost > 0 else np.inf

        return {
            'customers_targeted': customers_to_target,
            'estimated_customers_saved': customers_saved_by_campaign,
            'total_campaign_cost': total_campaign_cost,
            'estimated_revenue_saved': revenue_saved,
            'estimated_roi': roi
        }

    def generate_business_impact_report(self, y_true, y_pred, campaign_success_rate=0.5):
        """Generates a comprehensive report on the business impact."""
        print("--- Business Impact Report ---")
        print(f"Customer Lifetime Value (CLV): ${self.clv:,.2f}")
        
        # Cost of doing nothing (all actual churners are lost)
        total_churners = y_true.sum()
        cost_of_doing_nothing = self.calculate_churn_cost(total_churners)
        print(f"\nCost of Doing Nothing (Losing all {total_churners} churners): ${cost_of_doing_nothing:,.2f}")

        # Analysis of the proposed retention campaign
        campaign_analysis = self.analyze_campaign_effectiveness(y_true, y_pred, campaign_success_rate)
        print("\nTargeted Retention Campaign Analysis:")
        print(f"  - Customers targeted for retention: {campaign_analysis['customers_targeted']}")
        print(f"  - Estimated customers saved (at {campaign_success_rate:.0%} success rate): {campaign_analysis['estimated_customers_saved']:.1f}")
        print(f"  - Total campaign cost: ${campaign_analysis['total_campaign_cost']:,.2f}")
        print(f"  - Estimated revenue saved: ${campaign_analysis['estimated_revenue_saved']:,.2f}")
        print(f"  - Estimated Campaign ROI: {campaign_analysis['estimated_roi']:.2%}")
        
        net_impact = campaign_analysis['estimated_revenue_saved'] - campaign_analysis['total_campaign_cost']
        print(f"\nNet Financial Impact of Campaign: ${net_impact:,.2f}")
        print("--- End of Report ---")

if __name__ == '__main__':
    # --- Business Assumptions ---
    AVG_MONTHLY_REVENUE = 70.0
    AVG_CUSTOMER_LIFESPAN_MONTHS = 24
    CUSTOMER_ACQUISITION_COST = 300.0
    RETENTION_CAMPAIGN_COST_PER_CUSTOMER = 50.0

    # Initialize the evaluator
    business_eval = BusinessEvaluator(
        avg_monthly_revenue=AVG_MONTHLY_REVENUE,
        avg_customer_lifespan_months=AVG_CUSTOMER_LIFESPAN_MONTHS,
        customer_acquisition_cost=CUSTOMER_ACQUISITION_COST,
        retention_campaign_cost_per_customer=RETENTION_CAMPAIGN_COST_PER_CUSTOMER
    )

    # --- Dummy Model Predictions ---
    # Imagine we have a test set of 1000 customers
    np.random.seed(42)
    y_true_dummy = pd.Series(np.random.choice([0, 1], size=1000, p=[0.8, 0.2])) # 20% churn rate
    # A reasonably good model
    y_pred_dummy = y_true_dummy.apply(lambda x: x if np.random.rand() > 0.3 else 1 - x) 

    # --- Generate Report ---
    business_eval.generate_business_impact_report(y_true_dummy, y_pred_dummy)
