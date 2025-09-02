#!/usr/bin/env python3
"""
Enhanced Adaptive Privacy Mechanism for Better Performance
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict
from privacymas_core import (
    PrivacyMASEnvironment, 
    AdaptivePrivacyManager,
    PrivacyFeedback,
    CoordinationResult
)

class EnhancedAdaptivePrivacyManager(AdaptivePrivacyManager):
    """Enhanced adaptive privacy with more aggressive learning"""
    
    def __init__(self, initial_epsilon: float = 1.0, learning_rate: float = 0.05):
        # Increase learning rate from 0.01 to 0.05
        super().__init__(initial_epsilon, learning_rate * 5)
        
        # Enhanced adaptation parameters
        self.momentum = 0.9
        self.velocity = 0.0
        self.adaptation_threshold = 0.1
        
        # Track performance history for better adaptation
        self.utility_history = []
        self.privacy_history = []
        self.attack_history = []
        
        # Wider epsilon range for more flexibility
        self.min_epsilon = 0.05  # Lower minimum
        self.max_epsilon = 3.0   # Higher maximum
        
    def update_epsilon(self, feedback: PrivacyFeedback) -> float:
        """Enhanced epsilon update with momentum and performance tracking"""
        
        # Track metrics
        self.utility_history.append(feedback.coordination_quality)
        self.attack_history.append(feedback.attack_detected)
        
        # Compute enhanced adaptation signal
        adaptation_signal = self._compute_enhanced_adaptation_signal(feedback)
        
        # Apply momentum for smoother updates
        self.velocity = self.momentum * self.velocity + (1 - self.momentum) * adaptation_signal
        
        # Update epsilon with enhanced learning
        epsilon_change = self.learning_rate * self.velocity
        
        # Apply adaptive learning rate based on performance
        if len(self.utility_history) > 10:
            recent_utility = np.mean(self.utility_history[-10:])
            if recent_utility < 0.3:  # Poor performance
                epsilon_change *= 2.0  # More aggressive adaptation
            elif recent_utility > 0.8:  # Good performance
                epsilon_change *= 0.5  # More conservative
        
        # Update epsilon
        new_epsilon = np.clip(
            self.epsilon + epsilon_change,
            self.min_epsilon,
            self.max_epsilon
        )
        
        # Record adaptation
        self.adaptation_history.append({
            'old_epsilon': self.epsilon,
            'new_epsilon': new_epsilon,
            'feedback': feedback,
            'adaptation_signal': adaptation_signal,
            'velocity': self.velocity
        })
        
        self.epsilon = new_epsilon
        return self.epsilon
    
    def _compute_enhanced_adaptation_signal(self, feedback: PrivacyFeedback) -> float:
        """Enhanced adaptation signal computation"""
        signal = 0.0
        
        # Stronger response to attacks
        if feedback.attack_detected:
            signal -= 0.8  # Increased from 0.5
        
        # More aggressive utility optimization
        if feedback.utility_degradation > 0.2:
            signal += 0.5 * feedback.utility_degradation  # Increased from 0.3
        
        # Coordination quality with threshold
        if feedback.coordination_quality < 0.3:
            signal += 0.4 * (0.5 - feedback.coordination_quality)  # Increased
        elif feedback.coordination_quality < 0.5:
            signal += 0.2 * (0.5 - feedback.coordination_quality)
        
        # Historical performance consideration
        if len(self.utility_history) > 5:
            utility_trend = np.mean(self.utility_history[-5:]) - np.mean(self.utility_history[-10:-5]) if len(self.utility_history) > 10 else 0
            signal += 0.3 * utility_trend  # Adapt based on trend
        
        # Attack frequency consideration
        if len(self.attack_history) > 10:
            attack_rate = sum(self.attack_history[-10:]) / 10
            if attack_rate > 0.3:  # Frequent attacks
                signal -= 0.3
        
        return signal

def patch_adaptive_privacy():
    """Patch existing environments to use enhanced adaptive privacy"""
    from privacymas_core import PrivacyMASEnvironment
    from updated_medical_framework import EnhancedMedicalEnvironment
    
    # Monkey patch the environment classes
    original_init = PrivacyMASEnvironment.__init__
    
    def new_init(self, num_agents, domain_type="medical", initial_epsilon=1.0, cluster_size=10):
        original_init(self, num_agents, domain_type, initial_epsilon, cluster_size)
        # Replace with enhanced adaptive privacy
        self.adaptive_privacy = EnhancedAdaptivePrivacyManager(initial_epsilon)
    
    PrivacyMASEnvironment.__init__ = new_init
    
    # Also patch medical environment
    original_medical_init = EnhancedMedicalEnvironment.__init__
    
    def new_medical_init(self, num_agents, dataset_manager, initial_epsilon=1.0, cluster_size=10, device=None):
        original_medical_init(self, num_agents, dataset_manager, initial_epsilon, cluster_size, device)
        # Replace with enhanced adaptive privacy
        self.adaptive_privacy = EnhancedAdaptivePrivacyManager(initial_epsilon)
    
    EnhancedMedicalEnvironment.__init__ = new_medical_init
    
    print("✓ Patched environments with enhanced adaptive privacy")

def run_enhanced_experiment():
    """Run experiment with enhanced adaptive privacy"""
    from updated_medical_framework import MedicalDatasetManager, EnhancedMedicalEnvironment
    
    # Apply enhancement
    patch_adaptive_privacy()
    
    # Setup
    dataset_manager = MedicalDatasetManager()
    dataset_manager.load_dataset(sample_size=100)
    dataset_manager.process_cases_for_coordination(num_cases=50)
    
    print("\nTesting Enhanced Adaptive Privacy")
    print("="*60)
    
    epsilon_values = [0.1, 0.5, 1.0, 2.0]
    n_agents = 20
    num_episodes = 50  # More episodes for better adaptation
    
    results = []
    
    for epsilon in epsilon_values:
        print(f"\nTesting ε={epsilon}")
        
        # Static baseline
        env = EnhancedMedicalEnvironment(n_agents, dataset_manager, epsilon)
        static_utilities = []
        
        for episode in range(num_episodes):
            state = env.reset()
            coord_result, _ = env.step(state['observations'], use_adaptive_privacy=False)
            static_utilities.append(coord_result.utility_score)
        
        # Enhanced adaptive
        env = EnhancedMedicalEnvironment(n_agents, dataset_manager, epsilon)
        adaptive_utilities = []
        epsilon_history = []
        
        for episode in range(num_episodes):
            state = env.reset()
            coord_result, _ = env.step(state['observations'], use_adaptive_privacy=True)
            adaptive_utilities.append(coord_result.utility_score)
            epsilon_history.append(env.adaptive_privacy.epsilon)
        
        # Calculate improvement
        static_mean = np.mean(static_utilities)
        adaptive_mean = np.mean(adaptive_utilities)
        improvement = (adaptive_mean - static_mean) / static_mean * 100
        
        print(f"  Static utility: {static_mean:.3f}")
        print(f"  Adaptive utility: {adaptive_mean:.3f}")
        print(f"  Improvement: {improvement:.1f}%")
        print(f"  Epsilon range: [{min(epsilon_history):.3f}, {max(epsilon_history):.3f}]")
        print(f"  Final epsilon: {epsilon_history[-1]:.3f}")
        
        results.append({
            'epsilon': epsilon,
            'static_utility': static_mean,
            'adaptive_utility': adaptive_mean,
            'improvement': improvement,
            'epsilon_variance': np.std(epsilon_history)
        })
    
    # Summary
    avg_improvement = np.mean([r['improvement'] for r in results])
    print(f"\n{'='*60}")
    print(f"Average Improvement: {avg_improvement:.1f}%")
    print(f"Target Achieved (15-20%): {'YES' if avg_improvement >= 15 else 'NO'}")
    
    return results

if __name__ == "__main__":
    results = run_enhanced_experiment()