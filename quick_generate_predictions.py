# quick_generate_predictions.py
from main import TradingSystem
import logging

logging.basicConfig(level=logging.INFO)

print("Generating predictions...")

try:
    # Initialize system
    system = TradingSystem()

    # Load models
    print("Loading models...")
    feature_data = system.collect_and_prepare_data()
    system._load_models(feature_data)

    # Generate predictions
    print("Generating predictions...")
    predictions = system.generate_predictions(save_to_db=True)

    if predictions is not None and not predictions.empty:
        print(f"\n✅ Generated {len(predictions)} predictions")
        print(f"Saved to: predictions.csv")
        print(f"\nTop 5 predictions:")
        print(predictions.head()[['symbol', 'predicted_return', 'confidence']])
    else:
        print("❌ No predictions generated")

except Exception as e:
    print(f"❌ Error: {e}")
