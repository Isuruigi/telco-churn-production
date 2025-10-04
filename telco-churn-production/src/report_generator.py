import os
import pandas as pd
from src.config.config import REPORTS_PATH
from src.utils import create_directories

class ReportGenerator:
    """Generates comprehensive project reports in Markdown format."""

    def __init__(self, report_dir=REPORTS_PATH):
        """
        Initializes the ReportGenerator.

        Args:
            report_dir (str): The directory where reports will be saved.
        """
        self.report_dir = report_dir
        create_directories([self.report_dir])
        self.report_content = ""

    def generate_executive_summary(self, best_model_name: str, key_metrics: dict, business_impact: dict) -> str:
        """Generates a high-level executive summary."""
        summary = f"""
        ## 1. Executive Summary
        
        This report summarizes the findings of the Telco Customer Churn prediction project. 
        The best performing model identified is the **{best_model_name}**.
        
        **Key Performance Metrics:**
        - ROC AUC: {key_metrics.get('ROC_AUC', 0):.3f}
        - Precision (Weighted Avg): {key_metrics.get('precision', 0):.3f}
        - Recall (Weighted Avg): {key_metrics.get('recall', 0):.3f}
        
        **Business Impact:**
        By using the predictive model to target at-risk customers with a retention campaign, we estimate a potential **net financial impact of ${business_impact.get('net_impact', 0):,.2f}**.
        This is based on an estimated ROI of {business_impact.get('estimated_roi', 0):.2%}.
        """
        return summary

    def generate_eda_report(self) -> str:
        """Generates the EDA section of the report."""
        # In a real scenario, this would be more detailed.
        # For now, it will point to the saved figures.
        eda_report = f"""
        ## 2. Exploratory Data Analysis (EDA)
        
        The EDA revealed several key insights:
        - The dataset is imbalanced, with churners representing the minority class.
        - Features like `Contract`, `tenure`, and `InternetService` show a strong correlation with churn.
        - Numerical features like `tenure` and `MonthlyCharges` have different distributions for churners vs. non-churners.
        
        *Detailed plots can be found in the `{os.path.join(REPORTS_PATH, 'figures')}` directory.*
        """
        return eda_report

    def generate_model_performance_report(self, comparison_df: pd.DataFrame) -> str:
        """Generates the model performance section of the report."""
        report = f"""
        ## 3. Model Performance
        
        Several models were trained and evaluated. The table below summarizes their performance on the test set.
        
        {comparison_df.to_markdown()}
        
        *ROC curves and confusion matrices can be found in the `{os.path.join(REPORTS_PATH, 'figures', 'visualization')}` directory.*
        """
        return report

    def generate_business_impact_report(self, business_impact: dict) -> str:
        """Generates the business impact analysis section."""
        report = f"""
        ## 4. Business Impact Analysis
        
        A simulation was run to estimate the business impact of using the best model to drive a retention campaign.
        
        - **Customers Targeted:** {business_impact.get('customers_targeted', 0)}
        - **Estimated Customers Saved:** {business_impact.get('estimated_customers_saved', 0):.1f}
        - **Total Campaign Cost:** ${business_impact.get('total_campaign_cost', 0):,.2f}
        - **Estimated Revenue Saved:** ${business_impact.get('estimated_revenue_saved', 0):,.2f}
        - **Estimated Campaign ROI:** {business_impact.get('estimated_roi', 0):.2%}
        
        ### Net Financial Impact: ${business_impact.get('net_impact', 0):,.2f}
        """
        return report

    def generate_full_report(self, best_model_name, key_metrics, business_impact, comparison_df, file_name="project_report.md"):
        """Generates and saves a single, comprehensive project report."""
        print("--- Generating Full Project Report ---")
        
        # Start with a title
        self.report_content = "# Telco Churn Prediction Project Report\n\n"
        
        # Add sections
        self.report_content += self.generate_executive_summary(best_model_name, key_metrics, business_impact)
        self.report_content += "\n" + "-" * 20 + "\n"
        self.report_content += self.generate_eda_report()
        self.report_content += "\n" + "-" * 20 + "\n"
        self.report_content += self.generate_model_performance_report(comparison_df)
        self.report_content += "\n" + "-" * 20 + "\n"
        self.report_content += self.generate_business_impact_report(business_impact)
        
        # Save the report
        save_path = os.path.join(self.report_dir, file_name)
        with open(save_path, 'w') as f:
            f.write(self.report_content)
        
        print(f"Full report saved to {save_path}")
        print("--- Report Generation Complete ---\\n")

if __name__ == '__main__':
    # This is a demonstration. In a real run, you would get these artifacts from your pipelines.
    
    # --- Dummy Artifacts for Demonstration ---
    best_model = 'Tuned RF'
    metrics = {'ROC_AUC': 0.85, 'precision': 0.8, 'recall': 0.9}
    biz_impact = {
        'net_impact': 50000,
        'estimated_roi': 1.5,
        'customers_targeted': 200,
        'estimated_customers_saved': 80,
        'total_campaign_cost': 10000,
        'estimated_revenue_saved': 60000
    }
    comparison_data = {
        'Model': ['Logistic Regression', 'Random Forest', 'Tuned RF'],
        'ROC_AUC': [0.82, 0.84, 0.85],
        'f1-score': [0.78, 0.81, 0.82]
    }
    model_comparison = pd.DataFrame(comparison_data).set_index('Model')

    # --- Generate Report ---
    report_gen = ReportGenerator()
    report_gen.generate_full_report(
        best_model_name=best_model,
        key_metrics=metrics,
        business_impact=biz_impact,
        comparison_df=model_comparison
    )
