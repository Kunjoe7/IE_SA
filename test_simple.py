#!/usr/bin/env python
"""
Simple test of ICDM-SA package without scikit-survival dependency
"""

import numpy as np
import torch
import sys

def test_basic_functionality():
    print("Testing ICDM-SA basic functionality...")
    
    try:
        # Test basic imports (avoid the problematic util module for now)
        from icdm_sa.algorithm.model_imp import MultiTaskModel
        from icdm_sa.algorithm.multi_task_dataset import MultiTaskDataset
        from icdm_sa.algorithm.cindex import Cindex
        print("✓ Core modules imported successfully")
        
        # Test model creation
        n_features = 10
        n_intervals = 5
        model = MultiTaskModel(n_features, n_intervals)
        print("✓ Model created successfully")
        
        # Test forward pass
        batch_size = 4
        x = torch.randn(batch_size, n_features)
        outputs = model(x)
        assert len(outputs) == n_intervals
        print(f"✓ Forward pass successful - got {len(outputs)} task outputs")
        
        # Test dataset creation
        X = torch.randn(100, n_features)
        Y_list = [torch.randint(0, 2, (100, 1)).float() for _ in range(n_intervals)]
        W_list = [torch.ones(100, 1) for _ in range(n_intervals)]
        events = torch.randint(0, 2, (100,))
        
        dataset = MultiTaskDataset(X, Y_list, W_list, events)
        print(f"✓ Dataset created with {len(dataset)} samples")
        
        # Test data loading
        from torch.utils.data import DataLoader
        loader = DataLoader(dataset, batch_size=16, shuffle=True)
        x_batch, y_batch, w_batch, e_batch, idx_batch = next(iter(loader))
        print(f"✓ DataLoader working - batch shapes: X={x_batch.shape}, events={e_batch.shape}")
        
        # Test C-index
        cindex = Cindex()
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5]) 
        events = np.array([1, 1, 0, 1, 1])
        c_score = cindex(y_true, y_pred, events)
        print(f"✓ C-index calculation working: {c_score:.4f}")
        
        print("\n✅ All basic tests passed!")
        print("\nNote: Some features requiring scikit-survival are not tested.")
        print("To use full functionality, install: pip install scikit-survival")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_flchain_model():
    print("\n\nTesting FLCHAIN model functionality...")
    try:
        from icdm_sa.datasets.flchain.flchain_model import FLCHAINModel
        print("✓ FLCHAINModel imported successfully")
        
        # This would require the actual data file
        print("Note: Full FLCHAIN model test requires data files")
        
        return True
    except Exception as e:
        print(f"❌ Error importing FLCHAINModel: {e}")
        return False

if __name__ == "__main__":
    print("="*60)
    print("ICDM-SA Package Simple Test")
    print("="*60)
    
    success = test_basic_functionality()
    success &= test_flchain_model()
    
    print("\n" + "="*60)
    if success:
        print("✅ Basic package functionality is working!")
        print("\nYou can now use the package for survival analysis.")
        print("Try: python examples/example_custom_data.py")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    
    sys.exit(0 if success else 1)