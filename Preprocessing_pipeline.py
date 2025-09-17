import pandas as pd
import argparse
from Data_Cleaning_and_Preprocessing_pipeline.src.processing.pipeline import AutomatedPreprocessingPipeline
from ai_client.client import AIClient
from ai_client.config import API_KEY


def main():
    """Main function with fully automated preprocessing"""

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="AI-driven automated preprocessing pipeline"
    )
    parser.add_argument("dataset_file", help="Path to the dataset CSV file")
    args = parser.parse_args()

    try:
        # Initialize AI client and automated pipeline
        ai_client = AIClient(API_KEY)
        pipeline = AutomatedPreprocessingPipeline(ai_client)

        # Load your dataset with pandas only
        try:
            data = pd.read_csv(args.dataset_file)
            print(f"Dataset loaded successfully with Pandas: {data.shape}")
        except FileNotFoundError:
            print(f"Dataset file not found: {args.dataset_file}")
            return
        except Exception as e:
            print(f"Error loading dataset: {str(e)}")
            return

    except Exception as e:
        print(f"Error initializing pipeline: {str(e)}")
        return

    # Process the dataset with full automation
    print("\nStarting AI-driven automated preprocessing pipeline...")
    print("=" * 80)

    result = pipeline.process_any_dataset(data)

    if result["success"]:
        print("\n" + "=" * 20)
        print("PREPROCESSING COMPLETED SUCCESSFULLY!")
        print("=" * 20)
    else:
        print(f"\nPreprocessing failed: {result['error']}")
        print("Please check your data and try again.")


if __name__ == "__main__":
    main()
