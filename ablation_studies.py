#!/usr/bin/env python3
"""
Ablation Studies for PrivacyMAS - FIXED VERSION
Evaluates the contribution of each component to overall performance
Author: PrivacyMAS Research Team
Date: 2025-01-09
"""

import os
import sys
import json
import logging
import warnings
warnings.filterwarnings('ignore')

# Disable LLMs completely before any imports
os.environ['DISABLE_LLMS'] = '1'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torch

# Import frameworks
from privacymas_core import PrivacyMASEnvironment, Agent

# Simple rule-based agent to avoid LLM loading
class SimpleRuleBasedAgent(Agent):
    """Simple rule-based agent that doesn't use LLMs"""
    def __init__(self, agent_id: int, capabilities: Dict = None):
        super().__init__(agent_id, capabilities or {})
        self._use_cpu_fallback = True
        self.pipeline = None
        self.model = None
        
    def generate_action(self, observation: np.ndarray, private: bool = True, context: Optional[Dict] = None) -> np.ndarray:
        """Generate action using simple rules"""
        action = observation.copy()
        
        # Add some domain-specific logic
        if context and 'specialty' in context:
            if context['specialty'] == 'emergency':
                action = action * 1.2
            elif context['specialty'] == 'risk_manager':
                action = action * 0.8
        
        # Add small random noise
        action += np.random.normal(0, 0.1, action.shape)
        return action

from medical_framework_updated import (
    MedicalConfig, MedicalDatasetManager, 
    EnhancedMedicalEnvironment
)
from finance_framework_final import (
    FinanceConfig, FinanceDatasetManager,
    EnhancedFinanceEnvironment
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class AblationConfig:
    """Configuration for ablation studies"""
    output_dir: str = "ablation_results"
    num_episodes: int = 100
    num_runs: int = 5
    epsilon: float = 1.0
    num_agents: int = 8  # Use optimal agent count
    domain: str = "medical"  # or "finance"
    num_cases: int = 200  # Optimal case count
    
    def __post_init__(self):
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)


class AblationStudy:
    """Conducts ablation studies on PrivacyMAS components - FIXED"""
    
    def __init__(self, config: AblationConfig):
        self.config = config
        self.results = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize dataset
        self._initialize_dataset()
    
    def _initialize_dataset(self):
        """Initialize dataset for ablation studies - FIXED"""
        
        if self.config.domain == "medical":
            logger.info("Initializing medical dataset for ablation...")
            # MedicalConfig has use_llm parameter
            med_config = MedicalConfig(max_samples=200, use_llm=False)
            self.dataset_manager = MedicalDatasetManager(med_config)
            self.dataset_manager.load_dataset()
            self.dataset_manager.process_cases_for_coordination(num_cases=self.config.num_cases)
            self.domain_config = med_config
        else:
            logger.info("Initializing finance dataset for ablation...")
            # FinanceConfig does NOT have use_llm parameter
            fin_config = FinanceConfig(max_samples=200)
            self.dataset_manager = FinanceDatasetManager(fin_config)
            self.dataset_manager.load_dataset()
            self.dataset_manager.process_cases_for_coordination(num_cases=self.config.num_cases)
            self.domain_config = fin_config
    
    def run_complete_ablation(self) -> Dict[str, pd.DataFrame]:
        """Run all ablation studies"""
        
        logger.info("="*60)
        logger.info("STARTING ABLATION STUDIES")
        logger.info(f"Domain: {self.config.domain}")
        logger.info(f"Episodes: {self.config.num_episodes}")
        logger.info(f"Runs: {self.config.num_runs}")
        logger.info(f"Agents: {self.config.num_agents}")
        logger.info(f"Cases: {self.config.num_cases}")
        logger.info("="*60)
        
        # Clear GPU cache at start
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 1. Hierarchical Coordination Ablation
        logger.info("\n1. Ablation: Hierarchical Coordination")
        self.results['hierarchical'] = self._ablate_hierarchical_coordination()
        
        # Clear memory between studies
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 2. Adaptive Privacy Ablation
        logger.info("\n2. Ablation: Adaptive Privacy")
        self.results['adaptive_privacy'] = self._ablate_adaptive_privacy()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 3. Environmental Learning Ablation
        logger.info("\n3. Ablation: Environmental Learning")
        self.results['environmental_learning'] = self._ablate_environmental_learning()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 4. Attack Detection Ablation
        logger.info("\n4. Ablation: Attack Detection")
        self.results['attack_detection'] = self._ablate_attack_detection()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 5. Privacy Mechanism Ablation
        logger.info("\n5. Ablation: Privacy Mechanisms")
        self.results['privacy_mechanisms'] = self._ablate_privacy_mechanisms()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 6. Agent Count Sensitivity
        logger.info("\n6. Ablation: Agent Count Sensitivity")
        self.results['agent_sensitivity'] = self._ablate_agent_sensitivity()
        
        # Save results
        self._save_results()
        
        # Generate visualizations
        self._generate_ablation_plots()
        
        logger.info("\n" + "="*60)
        logger.info("ABLATION STUDIES COMPLETED")
        logger.info("="*60)
        
        return self.results
    
    def _ablate_hierarchical_coordination(self) -> pd.DataFrame:
        """Test impact of hierarchical vs flat coordination"""
        
        results = []
        
        for run in range(self.config.num_runs):
            logger.info(f"  Run {run+1}/{self.config.num_runs}")
            
            # Test different cluster sizes (flat = all agents in one cluster)
            cluster_sizes = [
                self.config.num_agents,  # Flat (no hierarchy)
                max(4, self.config.num_agents // 2),  # Medium clusters
                max(2, self.config.num_agents // 4),  # Small clusters
            ]
            
            for cluster_size in cluster_sizes:
                env = self._create_environment(cluster_size=cluster_size)
                
                utilities = []
                times = []
                rounds = []
                successes = []
                
                for episode in range(min(30, self.config.num_episodes)):
                    state = env.reset()
                    coord_result, _ = env.step(state['observations'])
                    
                    utilities.append(coord_result.utility_score)
                    times.append(coord_result.coordination_time)
                    rounds.append(coord_result.communication_rounds)
                    successes.append(coord_result.success)
                
                results.append({
                    'run': run,
                    'cluster_size': cluster_size,
                    'hierarchy_type': 'flat' if cluster_size == self.config.num_agents else 'hierarchical',
                    'avg_utility': np.mean(utilities),
                    'std_utility': np.std(utilities),
                    'avg_time': np.mean(times),
                    'avg_rounds': np.mean(rounds),
                    'success_rate': np.mean(successes)
                })
        
        df = pd.DataFrame(results)
        df.to_csv(f"{self.config.output_dir}/ablation_hierarchical_{self.timestamp}.csv", index=False)
        return df
    
    def _ablate_adaptive_privacy(self) -> pd.DataFrame:
        """Test different adaptive privacy configurations"""
        
        results = []
        
        # Test configurations
        configs = [
            {'name': 'no_adaptation', 'adaptive': False, 'learning_rate': 0.0},
            {'name': 'slow_adaptation', 'adaptive': True, 'learning_rate': 0.001},
            {'name': 'normal_adaptation', 'adaptive': True, 'learning_rate': 0.01},
            {'name': 'fast_adaptation', 'adaptive': True, 'learning_rate': 0.05},
            {'name': 'aggressive_adaptation', 'adaptive': True, 'learning_rate': 0.1}
        ]
        
        for config in configs:
            logger.info(f"  Testing {config['name']}")
            
            for run in range(self.config.num_runs):
                env = self._create_environment()
                
                # Modify adaptive privacy settings
                if config['adaptive']:
                    env.adaptive_privacy.learning_rate = config['learning_rate']
                
                utilities = []
                privacy_losses = []
                epsilon_values = []
                successes = []
                
                # Warm-up for adaptive methods
                if config['adaptive']:
                    for _ in range(10):
                        state = env.reset()
                        env.step(state['observations'], use_adaptive_privacy=True)
                
                # Measurement phase
                for episode in range(min(50, self.config.num_episodes)):
                    state = env.reset()
                    coord_result, feedback = env.step(
                        state['observations'],
                        use_adaptive_privacy=config['adaptive']
                    )
                    
                    utilities.append(coord_result.utility_score)
                    privacy_losses.append(coord_result.privacy_loss)
                    epsilon_values.append(env.adaptive_privacy.epsilon)
                    successes.append(coord_result.success)
                
                results.append({
                    'config': config['name'],
                    'run': run,
                    'adaptive': config['adaptive'],
                    'learning_rate': config['learning_rate'],
                    'avg_utility': np.mean(utilities),
                    'std_utility': np.std(utilities),
                    'avg_privacy_loss': np.mean(privacy_losses),
                    'epsilon_variance': np.var(epsilon_values),
                    'final_epsilon': epsilon_values[-1],
                    'success_rate': np.mean(successes)
                })
        
        df = pd.DataFrame(results)
        df.to_csv(f"{self.config.output_dir}/ablation_adaptive_privacy_{self.timestamp}.csv", index=False)
        return df
    
    def _ablate_environmental_learning(self) -> pd.DataFrame:
        """Test impact of environmental learning components"""
        
        results = []
        
        # Test configurations
        configs = [
            {'name': 'no_learning', 'use_nn': False, 'use_history': False},
            {'name': 'history_only', 'use_nn': False, 'use_history': True},
            {'name': 'nn_only', 'use_nn': True, 'use_history': False},
            {'name': 'full_learning', 'use_nn': True, 'use_history': True}
        ]
        
        for config in configs:
            logger.info(f"  Testing {config['name']}")
            
            for run in range(self.config.num_runs):
                env = self._create_environment()
                
                # Disable components as needed
                if not config['use_nn'] and hasattr(env, 'environment_learner'):
                    env.environment_learner = None
                
                utilities = []
                adaptations = []
                successes = []
                
                for episode in range(min(50, self.config.num_episodes)):
                    state = env.reset()
                    
                    # Disable history if needed
                    if not config['use_history']:
                        env.coordination_history = []
                    
                    coord_result, feedback = env.step(
                        state['observations'],
                        use_adaptive_privacy=True
                    )
                    
                    utilities.append(coord_result.utility_score)
                    adaptations.append(abs(feedback.suggested_epsilon_adjustment))
                    successes.append(coord_result.success)
                
                results.append({
                    'config': config['name'],
                    'run': run,
                    'use_nn': config['use_nn'],
                    'use_history': config['use_history'],
                    'avg_utility': np.mean(utilities),
                    'std_utility': np.std(utilities),
                    'avg_adaptation': np.mean(adaptations),
                    'success_rate': np.mean(successes)
                })
        
        df = pd.DataFrame(results)
        df.to_csv(f"{self.config.output_dir}/ablation_environmental_learning_{self.timestamp}.csv", index=False)
        return df
    
    def _ablate_attack_detection(self) -> pd.DataFrame:
        """Test impact of attack detection mechanisms"""
        
        results = []
        
        # Test configurations
        thresholds = [0.3, 0.5, 0.7, 0.9, 1.1]  # 1.1 effectively disables detection
        
        for threshold in thresholds:
            logger.info(f"  Testing threshold={threshold}")
            
            for run in range(self.config.num_runs):
                env = self._create_environment()
                
                # Modify attack detection threshold
                env.privacy_manager.attack_detector.attack_threshold = threshold
                
                detections = []
                utilities = []
                privacy_losses = []
                successes = []
                
                # Simulate with attack patterns
                for episode in range(min(50, self.config.num_episodes)):
                    state = env.reset()
                    
                    # Add attack-like queries periodically
                    observations = state['observations']
                    if episode % 10 < 3:  # 30% attack episodes
                        # Make queries more similar (attack pattern)
                        base = observations[0]
                        observations = [base + np.random.normal(0, 0.1, base.shape) 
                                      for _ in range(len(observations))]
                    
                    coord_result, feedback = env.step(observations, use_adaptive_privacy=True)
                    
                    detections.append(feedback.attack_detected)
                    utilities.append(coord_result.utility_score)
                    privacy_losses.append(coord_result.privacy_loss)
                    successes.append(coord_result.success)
                
                results.append({
                    'threshold': threshold,
                    'run': run,
                    'detection_enabled': threshold < 1.0,
                    'detection_rate': np.mean(detections),
                    'avg_utility': np.mean(utilities),
                    'avg_privacy_loss': np.mean(privacy_losses),
                    'success_rate': np.mean(successes)
                })
        
        df = pd.DataFrame(results)
        df.to_csv(f"{self.config.output_dir}/ablation_attack_detection_{self.timestamp}.csv", index=False)
        return df
    
    def _ablate_privacy_mechanisms(self) -> pd.DataFrame:
        """Test different privacy mechanisms - FIXED"""
        
        results = []
        
        # Test configurations
        mechanisms = [
            {'name': 'no_privacy', 'mechanism': None},
            {'name': 'laplace', 'mechanism': 'laplace'},
            {'name': 'gaussian', 'mechanism': 'gaussian'}
        ]
        
        for mech_config in mechanisms:
            logger.info(f"  Testing {mech_config['name']}")
            
            for run in range(self.config.num_runs):
                env = self._create_environment()
                
                utilities = []
                privacy_losses = []
                successes = []
                
                for episode in range(min(50, self.config.num_episodes)):
                    state = env.reset()
                    
                    # Apply privacy mechanism
                    if mech_config['mechanism'] is None:
                        # No privacy - use original observations
                        # Set epsilon to infinity to effectively disable privacy
                        old_epsilon = env.privacy_manager.epsilon
                        env.privacy_manager.epsilon = float('inf')
                        coord_result, feedback = env.step(state['observations'], use_adaptive_privacy=False)
                        env.privacy_manager.epsilon = old_epsilon
                        privacy_loss = 0.0  # No privacy loss when no privacy is used
                    else:
                        # Use specified mechanism - store original function
                        original_apply = env.privacy_manager.apply_noise
                        
                        # Create wrapper that uses specified mechanism
                        def custom_apply(data):
                            return original_apply(data, mechanism=mech_config['mechanism'])
                        
                        # Replace the apply_noise method temporarily
                        env.privacy_manager.apply_noise = custom_apply
                        coord_result, feedback = env.step(state['observations'])
                        # Restore original method
                        env.privacy_manager.apply_noise = original_apply
                        privacy_loss = coord_result.privacy_loss
                    
                    utilities.append(coord_result.utility_score)
                    privacy_losses.append(privacy_loss)
                    successes.append(coord_result.success)
                
                results.append({
                    'mechanism': mech_config['name'],
                    'run': run,
                    'avg_utility': np.mean(utilities),
                    'std_utility': np.std(utilities),
                    'avg_privacy_loss': np.mean(privacy_losses),
                    'success_rate': np.mean(successes)
                })
        
        df = pd.DataFrame(results)
        df.to_csv(f"{self.config.output_dir}/ablation_privacy_mechanisms_{self.timestamp}.csv", index=False)
        return df
    
    def _ablate_agent_sensitivity(self) -> pd.DataFrame:
        """Test sensitivity to number of agents"""
        
        results = []
        agent_counts = [5, 8, 10, 20, 50]  # Include optimal 8
        
        for n_agents in agent_counts:
            logger.info(f"  Testing {n_agents} agents")
            
            for run in range(self.config.num_runs):
                env = self._create_environment(num_agents=n_agents)
                
                utilities = []
                times = []
                successes = []
                
                for episode in range(min(30, self.config.num_episodes)):  # Fewer episodes for large agent counts
                    state = env.reset()
                    coord_result, _ = env.step(state['observations'])
                    
                    utilities.append(coord_result.utility_score)
                    times.append(coord_result.coordination_time)
                    successes.append(coord_result.success)
                
                results.append({
                    'num_agents': n_agents,
                    'run': run,
                    'avg_utility': np.mean(utilities),
                    'std_utility': np.std(utilities),
                    'avg_time': np.mean(times),
                    'success_rate': np.mean(successes)
                })
                
                # Clear memory after each configuration
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        df = pd.DataFrame(results)
        df.to_csv(f"{self.config.output_dir}/ablation_agent_sensitivity_{self.timestamp}.csv", index=False)
        return df
    
    def _create_environment(self, num_agents: Optional[int] = None, 
                           cluster_size: Optional[int] = None):
        """Create environment for ablation testing - FIXED to disable LLMs"""
        
        num_agents = num_agents or self.config.num_agents
        cluster_size = cluster_size or min(4, max(2, num_agents // 4))
        
        if self.config.domain == "medical":
            env = EnhancedMedicalEnvironment(
                num_agents, 
                self.dataset_manager,
                self.domain_config,
                initial_epsilon=self.config.epsilon
            )
            # Replace all agents with simple rule-based ones
            env.agents = [SimpleRuleBasedAgent(i, {'medical_specialty': 'general'}) 
                         for i in range(num_agents)]
        else:
            env = EnhancedFinanceEnvironment(
                num_agents,
                self.dataset_manager,
                self.domain_config,
                initial_epsilon=self.config.epsilon
            )
            # Replace all agents with simple rule-based ones
            env.agents = [SimpleRuleBasedAgent(i, {'financial_role': 'analyst'}) 
                         for i in range(num_agents)]
        
        # Override cluster size if specified
        env.coordination_module.cluster_size = cluster_size
        
        # Clear any GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return env
    
    def _save_results(self):
        """Save ablation results"""
        
        # Save summary
        summary = {
            'timestamp': self.timestamp,
            'config': asdict(self.config),
            'studies': list(self.results.keys()),
            'metrics': {}
        }
        
        for study_name, df in self.results.items():
            if not df.empty:
                summary['metrics'][study_name] = {
                    'records': len(df),
                    'columns': list(df.columns)
                }
        
        with open(f"{self.config.output_dir}/ablation_summary_{self.timestamp}.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Results saved to {self.config.output_dir}")
    
    def _generate_ablation_plots(self):
        """Generate ablation study visualizations"""
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        # Create figure with subplots for all ablations
        fig, axes = plt.subplots(3, 2, figsize=(14, 18))
        
        # 1. Hierarchical Coordination
        if 'hierarchical' in self.results:
            ax = axes[0, 0]
            df = self.results['hierarchical']
            
            grouped = df.groupby('cluster_size').agg({
                'avg_utility': ['mean', 'std'],
                'avg_time': ['mean', 'std'],
                'success_rate': 'mean'
            }).reset_index()
            
            ax.errorbar(grouped['cluster_size'], 
                       grouped[('avg_utility', 'mean')],
                       yerr=grouped[('avg_utility', 'std')],
                       marker='o', linewidth=2, capsize=5, label='Utility')
            
            # Add success rate on secondary axis
            ax2 = ax.twinx()
            ax2.plot(grouped['cluster_size'],
                    grouped[('success_rate', 'mean')],
                    'g--', marker='s', label='Success Rate')
            
            ax.set_xlabel('Cluster Size')
            ax.set_ylabel('Utility Score')
            ax2.set_ylabel('Success Rate', color='g')
            ax.set_title('Impact of Hierarchical Coordination')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper left')
            ax2.legend(loc='upper right')
        
        # 2. Adaptive Privacy
        if 'adaptive_privacy' in self.results:
            ax = axes[0, 1]
            df = self.results['adaptive_privacy']
            
            grouped = df.groupby('config').agg({
                'avg_utility': 'mean',
                'success_rate': 'mean',
                'epsilon_variance': 'mean'
            }).reset_index()
            
            x = range(len(grouped))
            width = 0.35
            
            ax.bar([i - width/2 for i in x], grouped['avg_utility'], 
                  width, label='Utility', alpha=0.7)
            ax.bar([i + width/2 for i in x], grouped['success_rate'],
                  width, label='Success Rate', alpha=0.7)
            
            ax.set_xticks(x)
            ax.set_xticklabels(grouped['config'], rotation=45, ha='right')
            ax.set_ylabel('Score')
            ax.set_title('Impact of Adaptive Privacy')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
        
        # 3. Environmental Learning
        if 'environmental_learning' in self.results:
            ax = axes[1, 0]
            df = self.results['environmental_learning']
            
            grouped = df.groupby('config').agg({
                'avg_utility': 'mean',
                'success_rate': 'mean',
                'avg_adaptation': 'mean'
            }).reset_index()
            
            x = range(len(grouped))
            width = 0.25
            
            ax.bar([i - width for i in x], grouped['avg_utility'], 
                  width, label='Utility', alpha=0.7)
            ax.bar(x, grouped['success_rate'],
                  width, label='Success Rate', alpha=0.7)
            ax.bar([i + width for i in x], grouped['avg_adaptation'] * 2,  # Scale for visibility
                  width, label='Adaptation (×2)', alpha=0.7)
            
            ax.set_xticks(x)
            ax.set_xticklabels(grouped['config'], rotation=45, ha='right')
            ax.set_ylabel('Score')
            ax.set_title('Impact of Environmental Learning')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
        
        # 4. Attack Detection
        if 'attack_detection' in self.results:
            ax = axes[1, 1]
            df = self.results['attack_detection']
            
            grouped = df.groupby('threshold').agg({
                'detection_rate': 'mean',
                'avg_utility': 'mean',
                'avg_privacy_loss': 'mean',
                'success_rate': 'mean'
            }).reset_index()
            
            ax2 = ax.twinx()
            
            line1 = ax.plot(grouped['threshold'], grouped['detection_rate'], 
                          'b-', marker='o', label='Detection Rate')
            line2 = ax.plot(grouped['threshold'], grouped['avg_utility'], 
                          'g-', marker='s', label='Utility')
            line3 = ax2.plot(grouped['threshold'], grouped['success_rate'], 
                           'r--', marker='^', label='Success Rate')
            
            ax.set_xlabel('Detection Threshold')
            ax.set_ylabel('Detection Rate / Utility')
            ax2.set_ylabel('Success Rate', color='r')
            ax.set_title('Impact of Attack Detection')
            
            lines = line1 + line2 + line3
            labels = [l.get_label() for l in lines]
            ax.legend(lines, labels, loc='best')
            ax.grid(True, alpha=0.3)
        
        # 5. Privacy Mechanisms
        if 'privacy_mechanisms' in self.results:
            ax = axes[2, 0]
            df = self.results['privacy_mechanisms']
            
            grouped = df.groupby('mechanism').agg({
                'avg_utility': ['mean', 'std'],
                'avg_privacy_loss': ['mean', 'std'],
                'success_rate': 'mean'
            }).reset_index()
            
            x = range(len(grouped))
            width = 0.25
            
            ax.bar([i - width for i in x], 
                  grouped[('avg_utility', 'mean')],
                  width, yerr=grouped[('avg_utility', 'std')],
                  label='Utility', alpha=0.7, capsize=5)
            
            ax.bar(x,
                  1 - grouped[('avg_privacy_loss', 'mean')],  # Privacy preservation
                  width, yerr=grouped[('avg_privacy_loss', 'std')],
                  label='Privacy Preservation', alpha=0.7, capsize=5)
            
            ax.bar([i + width for i in x],
                  grouped[('success_rate', 'mean')],
                  width, label='Success Rate', alpha=0.7)
            
            ax.set_xticks(x)
            ax.set_xticklabels(grouped['mechanism'])
            ax.set_ylabel('Score')
            ax.set_title('Impact of Privacy Mechanisms')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
        
        # 6. Agent Sensitivity
        if 'agent_sensitivity' in self.results:
            ax = axes[2, 1]
            df = self.results['agent_sensitivity']
            
            grouped = df.groupby('num_agents').agg({
                'avg_utility': ['mean', 'std'],
                'avg_time': ['mean', 'std'],
                'success_rate': 'mean'
            }).reset_index()
            
            ax2 = ax.twinx()
            
            ax.errorbar(grouped['num_agents'],
                       grouped[('avg_utility', 'mean')],
                       yerr=grouped[('avg_utility', 'std')],
                       marker='o', label='Utility', linewidth=2, capsize=5)
            
            ax.plot(grouped['num_agents'],
                   grouped[('success_rate', 'mean')],
                   'g--', marker='s', label='Success Rate')
            
            ax2.errorbar(grouped['num_agents'],
                        grouped[('avg_time', 'mean')],
                        yerr=grouped[('avg_time', 'std')],
                        marker='^', linewidth=2, 
                        capsize=5, color='orange', label='Time')
            
            ax.set_xlabel('Number of Agents')
            ax.set_ylabel('Utility / Success Rate')
            ax2.set_ylabel('Coordination Time (s)', color='orange')
            ax.set_title('Sensitivity to Agent Count')
            ax.legend(loc='upper left')
            ax2.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'Ablation Studies - {self.config.domain.capitalize()} Domain', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        save_path = f"{self.config.output_dir}/ablation_all_{self.timestamp}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Ablation plots saved to {save_path}")
        
        # Generate individual plots for paper
        self._generate_individual_ablation_plots()
    
    def _generate_individual_ablation_plots(self):
        """Generate individual ablation plots for the paper"""
        
        # Most important ablation: Adaptive vs Static
        if 'adaptive_privacy' in self.results:
            fig, ax = plt.subplots(figsize=(8, 6))
            
            df = self.results['adaptive_privacy']
            
            # Filter for key comparisons
            key_configs = ['no_adaptation', 'normal_adaptation', 'fast_adaptation']
            df_filtered = df[df['config'].isin(key_configs)]
            
            grouped = df_filtered.groupby('config').agg({
                'avg_utility': ['mean', 'std'],
                'avg_privacy_loss': ['mean', 'std'],
                'success_rate': 'mean'
            }).reset_index()
            
            x = range(len(grouped))
            width = 0.25
            
            bars1 = ax.bar([i - width for i in x],
                          grouped[('avg_utility', 'mean')],
                          width, yerr=grouped[('avg_utility', 'std')],
                          label='Utility', color='steelblue', alpha=0.8, capsize=5)
            
            bars2 = ax.bar(x,
                          1 - grouped[('avg_privacy_loss', 'mean')],
                          width, yerr=grouped[('avg_privacy_loss', 'std')],
                          label='Privacy Preservation', color='coral', alpha=0.8, capsize=5)
            
            bars3 = ax.bar([i + width for i in x],
                          grouped[('success_rate', 'mean')],
                          width, label='Success Rate', color='green', alpha=0.8)
            
            ax.set_xticks(x)
            ax.set_xticklabels(['Static', 'Adaptive (Normal)', 'Adaptive (Fast)'])
            ax.set_ylabel('Score', fontsize=12)
            ax.set_title('Ablation: Adaptive Privacy Mechanism', fontsize=14, fontweight='bold')
            ax.legend(loc='upper left')
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim(0, 1)
            
            # Add value labels on bars
            for bars in [bars1, bars2, bars3]:
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=10)
            
            save_path = f"{self.config.output_dir}/ablation_adaptive_privacy.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Key ablation plot saved to {save_path}")


def main():
    """Main entry point for ablation studies"""
    
    import argparse
    parser = argparse.ArgumentParser(description='PrivacyMAS Ablation Studies')
    parser.add_argument('--domain', choices=['medical', 'finance'], 
                       default='medical', help='Domain to test')
    parser.add_argument('--episodes', type=int, default=100, 
                       help='Number of episodes per experiment')
    parser.add_argument('--runs', type=int, default=5, 
                       help='Number of runs for statistical significance')
    parser.add_argument('--agents', type=int, default=8,
                       help='Number of agents (default: 8, optimal)')
    parser.add_argument('--cases', type=int, default=200,
                       help='Number of cases to process (default: 200, optimal)')
    parser.add_argument('--output-dir', type=str, default='ablation_results',
                       help='Output directory')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("PRIVACYMAS ABLATION STUDIES")
    print("="*60)
    print(f"Configuration:")
    print(f"  Domain: {args.domain}")
    print(f"  Episodes: {args.episodes}")
    print(f"  Runs: {args.runs}")
    print(f"  Agents: {args.agents}")
    print(f"  Cases: {args.cases}")
    print(f"  Output directory: {args.output_dir}")
    print(f"\nNOTE: Using rule-based agents (LLMs disabled for stability)")
    print("="*60)
    
    # Configure ablation studies
    config = AblationConfig(
        output_dir=args.output_dir,
        num_episodes=args.episodes,
        num_runs=args.runs,
        domain=args.domain,
        num_agents=args.agents,
        num_cases=args.cases
    )
    
    print("\n" + "="*60)
    print("PRIVACYMAS ABLATION STUDIES")
    print("="*60)
    print(f"Configuration:")
    print(f"  Domain: {args.domain}")
    print(f"  Episodes: {args.episodes}")
    print(f"  Runs: {args.runs}")
    print(f"  Agents: {args.agents}")
    print(f"  Cases: {args.cases}")
    print(f"  Output directory: {args.output_dir}")
    print(f"\nNOTE: Using rule-based agents (LLMs disabled for stability)")
    print("="*60)
    
    args = parser.parse_args()
    
    # Configure ablation studies
    config = AblationConfig(
        output_dir=args.output_dir,
        num_episodes=args.episodes,
        num_runs=args.runs,
        domain=args.domain,
        num_agents=args.agents,
        num_cases=args.cases
    )
    
    # Run ablation studies
    ablation = AblationStudy(config)
    results = ablation.run_complete_ablation()
    
    # Print summary
    print("\n" + "="*60)
    print("ABLATION STUDY SUMMARY")
    print("="*60)
    
    for study_name, df in results.items():
        if not df.empty:
            print(f"\n{study_name}:")
            print(f"  Records: {len(df)}")
            
            # Show key finding
            if study_name == 'adaptive_privacy':
                no_adapt = df[df['config'] == 'no_adaptation']['avg_utility'].mean()
                with_adapt = df[df['config'] == 'normal_adaptation']['avg_utility'].mean()
                improvement = (with_adapt - no_adapt) / no_adapt * 100
                print(f"  Adaptive improvement: {improvement:.1f}%")
                
                # Also show success rates
                no_adapt_success = df[df['config'] == 'no_adaptation']['success_rate'].mean()
                with_adapt_success = df[df['config'] == 'normal_adaptation']['success_rate'].mean()
                print(f"  Static success rate: {no_adapt_success:.1%}")
                print(f"  Adaptive success rate: {with_adapt_success:.1%}")
            
            elif study_name == 'hierarchical':
                flat = df[df['hierarchy_type'] == 'flat']['avg_time'].mean()
                hier = df[df['hierarchy_type'] == 'hierarchical']['avg_time'].mean()
                if hier > 0:
                    speedup = flat / hier
                    print(f"  Hierarchical speedup: {speedup:.2f}x")
                
                # Also show success rates
                flat_success = df[df['hierarchy_type'] == 'flat']['success_rate'].mean()
                hier_success = df[df['hierarchy_type'] == 'hierarchical']['success_rate'].mean()
                print(f"  Flat success rate: {flat_success:.1%}")
                print(f"  Hierarchical success rate: {hier_success:.1%}")
            
            elif study_name == 'agent_sensitivity':
                # Show optimal agent count
                optimal_df = df.groupby('num_agents')['avg_utility'].mean()
                optimal_agents = optimal_df.idxmax()
                print(f"  Optimal agent count: {optimal_agents}")
                print(f"  Optimal utility: {optimal_df.max():.3f}")
                
                # Show success rate at optimal
                optimal_success = df[df['num_agents'] == optimal_agents]['success_rate'].mean()
                print(f"  Success rate at optimal: {optimal_success:.1%}")
    
    print(f"\nResults saved to: {config.output_dir}")
    print("\n✅ Ablation studies completed successfully!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
