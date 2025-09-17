
from ML.AutoMLPipeline import AutoMLPipeline
def main():
    """Example of how to use the AutoML pipeline with report integration"""

    # Initialize with existing AI client
    
    from ai_client.client import AIClient
    from ai_client.config import OPENROUTER_API_KEY

    try:
        # Initialize AI client
        ai_client = AIClient(OPENROUTER_API_KEY)
        
        # Initialize AutoML pipeline
        automl = AutoMLPipeline(ai_client=ai_client)
        # Specify report path 
        report_path = r"D:\projects\lex machina\outputs\data_analysis_report.json"  
        
        # Run AutoML on a processed dataset with optional report
        automl.run_automl(
            r"D:\projects\lex machina\outputs\processed_dataset_classification.csv",
            report_path=report_path
        )
        
        
        
    except Exception as e:
        print(f"AutoML failed: {str(e)}")
        return None


if __name__ == "__main__":
    main()
    