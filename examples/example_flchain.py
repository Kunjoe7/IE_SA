"""
Example script showing how to use ICDM-SA with FLCHAIN dataset
"""

from icdm_sa import FLCHAINModel
import easydict

def main():
    # Configure model parameters
    args = easydict.EasyDict({
        "batch_size": 64,
        "cuda": True,  # Use GPU if available
        "lr": 0.01,
        "epochs": 50,  # Reduced for demo
        "clip": 5.0,
        "lambda_reg": 0.01,
        "save_path": "flchain_results",
        "eg_k": 1,
        "early_stop_patience": 11,
    })
    
    # Initialize model with FLCHAIN dataset
    print("Loading FLCHAIN dataset...")
    model = FLCHAINModel(args=args)
    
    # Prepare data
    print("Preparing data...")
    train_loader, test_loader = model.prepare_data()
    
    # Train model
    print("Training model...")
    trainer = model.train()
    
    # Evaluate model
    print("Evaluating model...")
    results = model.evaluate()
    
    print(f"\nResults:")
    print(f"C-index (Training): {results['c_index_train']:.4f}")
    print(f"C-index (Test): {results['c_index_test']:.4f}")
    
    # Access trained model for predictions
    trained_model = trainer.model
    print(f"\nModel architecture: {trained_model}")

if __name__ == "__main__":
    main()