#!/usr/bin/env python3
"""
Quick validation script to test both frameworks
Runs minimal tests to ensure everything is working
"""

import numpy as np
import torch
import os
import warnings
warnings.filterwarnings('ignore')

# Set environment for better CUDA debugging
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'  # Suppress transformer warnings

def quick_test_medical():
    """Quick test of medical framework"""
    print("\n=== Quick Medical Test ===")
    
    try:
        from medical_framework_enhanced import (
            MedicalConfig,
            MedicalDatasetManager,
            EnhancedMedicalEnvironment
        )
        
        # Clear any GPU cache first
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Minimal config
        config = MedicalConfig(
            max_samples=10,
            device="cpu"  # Use CPU to avoid GPU issues
        )
        
        # Load minimal data
        print("  Loading dataset...")
        dm = MedicalDatasetManager(config)
        success = dm.load_dataset()
        if not success:
            print("  Warning: Dataset loading failed, using synthetic data")
        
        # Process minimal cases
        if dm.dataset:
            dm.process_cases_for_coordination(5)
        
        # Create small environment - ensure at least 4 agents for cluster_size
        print("  Creating environment...")
        env = EnhancedMedicalEnvironment(
            num_agents=5,  # Increased from 3 to avoid cluster_size issues
            dataset_manager=dm,
            config=config
        )
        
        # Force all agents to rule-based
        for agent in env.agents:
            agent.pipeline = None
            agent._use_cpu_fallback = True
        
        # Run one episode
        print("  Running episode...")
        state = env.reset()
        coord_result, feedback = env.step(state['observations'])
        
        print(f"✓ Medical test passed!")
        print(f"  Utility: {coord_result.utility_score:.3f}")
        print(f"  Privacy: {coord_result.privacy_loss:.3f}")
        print(f"  Success: {coord_result.success}")
        
        return True
        
    except Exception as e:
        print(f"✗ Medical test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def quick_test_finance():
    """Quick test of finance framework"""
    print("\n=== Quick Finance Test ===")
    
    try:
        from finance_framework_enhanced import (
            FinanceConfig,
            FinanceDatasetManager,
            EnhancedFinanceEnvironment
        )
        
        # Clear any GPU cache first
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Minimal config
        config = FinanceConfig(
            max_samples=10,
            device="cpu"  # Use CPU to avoid GPU issues
        )
        
        # Load minimal data
        print("  Loading dataset...")
        dm = FinanceDatasetManager(config)
        success = dm.load_dataset()
        if not success:
            print("  Warning: Dataset loading failed, using synthetic data")
        
        # Process minimal cases
        if dm.dataset:
            dm.process_cases_for_coordination(5)
        
        # Create small environment - ensure at least 3 agents for cluster_size
        print("  Creating environment...")
        env = EnhancedFinanceEnvironment(
            num_agents=6,  # Increased from 4 to avoid cluster_size issues
            dataset_manager=dm,
            config=config
        )
        
        # Force all agents to rule-based
        for agent in env.agents:
            agent.pipeline = None
            agent._use_cpu_fallback = True
        
        # Run one episode
        print("  Running episode...")
        state = env.reset()
        coord_result, feedback = env.step(state['observations'])
        
        print(f"✓ Finance test passed!")
        print(f"  Utility: {coord_result.utility_score:.3f}")
        print(f"  Privacy: {coord_result.privacy_loss:.3f}")
        print(f"  Market: {state['market_state']['regime']}")
        print(f"  Success: {coord_result.success}")
        
        return True
        
    except Exception as e:
        print(f"✗ Finance test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run quick validation"""
    print("PrivacyMAS Quick Validation")
    print("=" * 40)
    
    # Clear any GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"Using GPU: {torch.cuda.get_device_name()}")
        print(f"Memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    else:
        print("Using CPU")
    
    # Test both frameworks
    medical_ok = quick_test_medical()
    
    # Clear memory between tests
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    finance_ok = quick_test_finance()
    
    # Summary
    print("\n" + "=" * 40)
    print("VALIDATION SUMMARY")
    print("=" * 40)
    print(f"Medical: {'✅ PASS' if medical_ok else '❌ FAIL'}")
    print(f"Finance: {'✅ PASS' if finance_ok else '❌ FAIL'}")
    
    if medical_ok and finance_ok:
        print("\n✅ Both frameworks are working!")
        print("\nYou can now:")
        print("1. Run full tests with test_frameworks_light.py")
        print("2. Run memory-managed experiments")
        print("3. Proceed to experiment_pipeline.py")
    else:
        print("\n❌ Fix issues before proceeding")
        print("\nTroubleshooting tips:")
        print("- Ensure datasets are accessible")
        print("- Check GPU memory availability")
        print("- Try with more agents (minimum 4-6)")
        print("- Set device='cpu' in configs")
    
    return 0 if (medical_ok and finance_ok) else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())