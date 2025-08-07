#!/usr/bin/env python
"""
Simple test to verify package installation and basic functionality
"""

import sys

def test_imports():
    """Test that all main components can be imported"""
    try:
        print("Testing imports...")
        
        # Test algorithm imports
        from icdm_sa import (
            MultiTaskModel,
            EGTrainer,
            MultiTaskDataset,
            Cindex
        )
        print("✓ Algorithm components imported successfully")
        
        # Test dataset imports
        from icdm_sa import FLCHAINModel
        print("✓ Dataset models imported successfully")
        
        # Test package info
        import icdm_sa
        print(f"✓ Package version: {icdm_sa.__version__}")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_model_creation():
    """Test basic model creation"""
    try:
        print("\nTesting model creation...")
        
        from icdm_sa import MultiTaskModel
        import torch
        
        # Create a simple model
        n_features = 10
        n_intervals = 5
        model = MultiTaskModel(n_features, n_intervals)
        
        # Test forward pass
        batch_size = 2
        x = torch.randn(batch_size, n_features)
        outputs = model(x)
        
        assert len(outputs) == n_intervals, f"Expected {n_intervals} outputs, got {len(outputs)}"
        print("✓ Model creation and forward pass successful")
        
        return True
        
    except Exception as e:
        print(f"✗ Model creation error: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 50)
    print("ICDM-SA Package Test")
    print("=" * 50)
    
    all_tests_passed = True
    
    # Run tests
    if not test_imports():
        all_tests_passed = False
        
    if not test_model_creation():
        all_tests_passed = False
    
    # Summary
    print("\n" + "=" * 50)
    if all_tests_passed:
        print("✓ All tests passed!")
        print("\nPackage is ready to use. Try running:")
        print("  python examples/example_flchain.py")
        return 0
    else:
        print("✗ Some tests failed. Please check the installation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())