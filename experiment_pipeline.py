#!/usr/bin/env python3
"""
Final Experiment Pipeline for PrivacyMAS Paper
Generates all figures and tables for NeurIPS 2025 submission
Author: PrivacyMAS Research Team
Date: 2025-01-09
"""

import os
import sys
import json
import logging
import traceback
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from dataclasses import dataclass, asdict
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torch
import pickle

# Configure plotting
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11

# Import core framework
from privacymas_core import (
    PrivacyMASEnvironment,
    CoordinationResult,
    PrivacyFeedback,
    Agent
)

# Import enhanced frameworks
from medical_framework_updated import (
    MedicalConfig,
    MedicalDatasetManager,
    EnhancedMedicalEnvironment,
    MedicalPrivacyAttackSimulator,
    MedicalPrivacyFeedback
)

from finance_framework_final import (
    FinanceConfig,
    FinanceDatasetManager,
    EnhancedFinanceEnvironment,
    FinancePrivacyAttackSimulator,
    FinancePrivacyFeedback
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('experiment_pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for experiments"""
    # Experiment settings
    output_dir: str = "experiment_results"
    figures_dir: str = "figures"
    tables_dir: str = "tables"
    checkpoint_dir: str = "checkpoints"
    
    # Experiment parameters
    epsilon_values: List[float] = None
    agent_counts: List[int] = None
    num_episodes: int = 100
    num_runs: int = 3  # Multiple runs for statistical significance
    
    # Domain settings
    enable_medical: bool = True
    enable_finance: bool = True
    medical_samples: int = 500
    finance_samples: int = 500
    
    # Performance settings
    use_parallel: bool = True
    max_workers: int = 4
    save_checkpoints: bool = True
    checkpoint_frequency: int = 50
    
    # Visualization settings
    generate_figures: bool = True
    generate_tables: bool = True
    figure_format: str = "png"
    table_format: str = "latex"  # or "csv"
    
    def __post_init__(self):
        if self.epsilon_values is None:
            self.epsilon_values = [0.1, 0.5, 1.0, 2.0]
        if self.agent_counts is None:
            self.agent_counts = [10, 20, 50, 100]
        
        # Create directories
        for dir_name in [self.output_dir, self.figures_dir, self.tables_dir, self.checkpoint_dir]:
            Path(dir_name).mkdir(parents=True, exist_ok=True)


class ExperimentOrchestrator:
    """Orchestrates all experiments for the paper"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.medical_manager = None
        self.finance_manager = None
        self.results = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize datasets
        self._initialize_datasets()
        
    def _initialize_datasets(self):
        """Initialize medical and finance datasets"""
        
        if self.config.enable_medical:
            logger.info("Initializing medical dataset...")
            med_config = MedicalConfig(
                max_samples=self.config.medical_samples,
                use_llm=False  # Stability
            )
            self.medical_manager = MedicalDatasetManager(med_config)
            if self.medical_manager.load_dataset():
                self.medical_manager.process_cases_for_coordination(
                    num_cases=min(200, self.config.medical_samples // 2)
                )
                logger.info(f"✓ Medical dataset ready: {len(self.medical_manager.processed_cases)} cases")
            else:
                logger.warning("Medical dataset loading failed, using synthetic data")
        
        if self.config.enable_finance:
            logger.info("Initializing finance dataset...")
            fin_config = FinanceConfig(
                max_samples=self.config.finance_samples,
                use_llm=False  # Stability
            )
            self.finance_manager = FinanceDatasetManager(fin_config)
            if self.finance_manager.load_dataset():
                self.finance_manager.process_cases_for_coordination(
                    num_cases=min(200, self.config.finance_samples // 2)
                )
                logger.info(f"✓ Finance dataset ready: {len(self.finance_manager.processed_cases)} cases")
            else:
                logger.warning("Finance dataset loading failed, using synthetic data")
    
    def run_complete_experiments(self) -> Dict[str, pd.DataFrame]:
        """Run all experiments for the paper"""
        
        logger.info("="*60)
        logger.info("STARTING COMPLETE EXPERIMENT SUITE")
        logger.info(f"Timestamp: {self.timestamp}")
        logger.info("="*60)
        
        # 1. Privacy-Utility Trade-off (Table 1, Figure 1)
        logger.info("\n1. Running Privacy-Utility Trade-off Analysis...")
        self.results['privacy_utility'] = self._run_privacy_utility_analysis()
        
        # 2. Scalability Analysis (Figure 2)
        logger.info("\n2. Running Scalability Analysis...")
        self.results['scalability'] = self._run_scalability_analysis()
        
        # 3. Environmental Adaptation (Figure 3)
        logger.info("\n3. Running Environmental Adaptation Analysis...")
        self.results['adaptation'] = self._run_adaptation_analysis()
        
        # 4. Domain-Specific Results (Table 2)
        logger.info("\n4. Running Domain-Specific Analysis...")
        self.results['domain_specific'] = self._run_domain_specific_analysis()
        
        # 5. Attack Resistance (Table 3)
        logger.info("\n5. Running Attack Resistance Analysis...")
        self.results['attack_resistance'] = self._run_attack_resistance_analysis()
        
        # Save all results
        self._save_results()
        
        # Generate figures and tables
        if self.config.generate_figures:
            self._generate_all_figures()
        
        if self.config.generate_tables:
            self._generate_all_tables()
        
        logger.info("\n" + "="*60)
        logger.info("EXPERIMENT SUITE COMPLETED")
        logger.info("="*60)
        
        return self.results
    
    def _run_privacy_utility_analysis(self) -> pd.DataFrame:
        """Run privacy-utility trade-off experiments"""
        
        results = []
        
        for domain in ['medical', 'finance']:
            if domain == 'medical' and not self.config.enable_medical:
                continue
            if domain == 'finance' and not self.config.enable_finance:
                continue
            
            dataset_manager = self.medical_manager if domain == 'medical' else self.finance_manager
            if not dataset_manager:
                continue
            
            for epsilon in self.config.epsilon_values:
                for n_agents in self.config.agent_counts[:3]:  # Focus on smaller agent counts
                    for run in range(self.config.num_runs):
                        
                        logger.info(f"  {domain} ε={epsilon} agents={n_agents} run={run+1}/{self.config.num_runs}")
                        
                        # Static privacy
                        static_results = self._run_single_experiment(
                            domain, dataset_manager, epsilon, n_agents, 
                            self.config.num_episodes, use_adaptive=False
                        )
                        
                        # Adaptive privacy
                        adaptive_results = self._run_single_experiment(
                            domain, dataset_manager, epsilon, n_agents,
                            self.config.num_episodes, use_adaptive=True
                        )
                        
                        # Aggregate results
                        for res in static_results:
                            res.update({
                                'domain': domain,
                                'epsilon': epsilon,
                                'num_agents': n_agents,
                                'mechanism': 'static',
                                'run': run
                            })
                            results.append(res)
                        
                        for res in adaptive_results:
                            res.update({
                                'domain': domain,
                                'epsilon': epsilon,
                                'num_agents': n_agents,
                                'mechanism': 'adaptive',
                                'run': run
                            })
                            results.append(res)
        
        df = pd.DataFrame(results)
        df.to_csv(f"{self.config.tables_dir}/privacy_utility_{self.timestamp}.csv", index=False)
        return df
    
    def _run_scalability_analysis(self) -> pd.DataFrame:
        """Run scalability experiments"""
        
        results = []
        agent_ranges = [10, 20, 50, 100, 200]
        
        for domain in ['medical', 'finance']:
            if domain == 'medical' and not self.config.enable_medical:
                continue
            if domain == 'finance' and not self.config.enable_finance:
                continue
            
            dataset_manager = self.medical_manager if domain == 'medical' else self.finance_manager
            if not dataset_manager:
                continue
            
            for n_agents in agent_ranges:
                logger.info(f"  Testing {domain} with {n_agents} agents")
                
                times = []
                rounds = []
                utilities = []
                
                for episode in range(min(30, self.config.num_episodes)):
                    if domain == 'medical':
                        env = EnhancedMedicalEnvironment(
                            n_agents, dataset_manager,
                            MedicalConfig(), initial_epsilon=1.0
                        )
                    else:
                        env = EnhancedFinanceEnvironment(
                            n_agents, dataset_manager,
                            FinanceConfig(), initial_epsilon=1.0
                        )
                    
                    state = env.reset()
                    start_time = time.time()
                    coord_result, _ = env.step(state['observations'])
                    elapsed = time.time() - start_time
                    
                    times.append(elapsed)
                    rounds.append(coord_result.communication_rounds)
                    utilities.append(coord_result.utility_score)
                
                results.append({
                    'domain': domain,
                    'num_agents': n_agents,
                    'avg_time': np.mean(times),
                    'std_time': np.std(times),
                    'avg_rounds': np.mean(rounds),
                    'avg_utility': np.mean(utilities),
                    'theoretical_complexity': np.mean(times) / np.log(n_agents + 1)
                })
        
        df = pd.DataFrame(results)
        df.to_csv(f"{self.config.tables_dir}/scalability_{self.timestamp}.csv", index=False)
        return df
    
    def _run_adaptation_analysis(self) -> pd.DataFrame:
        """Run environmental adaptation experiments"""
        
        results = []
        
        for domain in ['medical', 'finance']:
            if domain == 'medical' and not self.config.enable_medical:
                continue
            if domain == 'finance' and not self.config.enable_finance:
                continue
            
            dataset_manager = self.medical_manager if domain == 'medical' else self.finance_manager
            if not dataset_manager:
                continue
            
            n_agents = 20
            initial_epsilon = 1.0
            
            logger.info(f"  Testing adaptation in {domain} domain")
            
            if domain == 'medical':
                env = EnhancedMedicalEnvironment(
                    n_agents, dataset_manager,
                    MedicalConfig(), initial_epsilon
                )
            else:
                env = EnhancedFinanceEnvironment(
                    n_agents, dataset_manager,
                    FinanceConfig(), initial_epsilon
                )
            
            epsilon_history = []
            utility_history = []
            privacy_history = []
            attack_history = []
            
            # Simulate different scenarios
            scenarios = [
                ('normal', 50),
                ('attack', 30),
                ('high_sensitivity', 30),
                ('normal', 40)
            ]
            
            episode = 0
            for scenario, duration in scenarios:
                for _ in range(duration):
                    state = env.reset()
                    
                    # Modify observations based on scenario
                    observations = state['observations']
                    if scenario == 'attack':
                        # Simulate attack queries
                        observations = [obs + np.random.normal(0, 0.3, obs.shape) 
                                      for obs in observations]
                    elif scenario == 'high_sensitivity':
                        # Increase sensitivity features
                        observations = [obs * 1.5 for obs in observations]
                    
                    coord_result, feedback = env.step(observations, use_adaptive_privacy=True)
                    
                    epsilon_history.append(env.adaptive_privacy.epsilon)
                    utility_history.append(coord_result.utility_score)
                    privacy_history.append(coord_result.privacy_loss)
                    attack_history.append(feedback.attack_detected)
                    
                    results.append({
                        'domain': domain,
                        'episode': episode,
                        'scenario': scenario,
                        'epsilon': env.adaptive_privacy.epsilon,
                        'utility': coord_result.utility_score,
                        'privacy_loss': coord_result.privacy_loss,
                        'attack_detected': feedback.attack_detected
                    })
                    
                    episode += 1
        
        df = pd.DataFrame(results)
        df.to_csv(f"{self.config.tables_dir}/adaptation_{self.timestamp}.csv", index=False)
        return df
    
    def _run_domain_specific_analysis(self) -> pd.DataFrame:
        """Run domain-specific experiments"""
        
        results = []
        n_agents = 20
        epsilon = 1.0
        
        # Medical domain metrics
        if self.config.enable_medical and self.medical_manager:
            logger.info("  Analyzing medical domain performance")
            
            env = EnhancedMedicalEnvironment(
                n_agents, self.medical_manager,
                MedicalConfig(), epsilon
            )
            
            static_metrics = {'diagnostic_accuracy': [], 'consensus': [], 'compliance': []}
            adaptive_metrics = {'diagnostic_accuracy': [], 'consensus': [], 'compliance': []}
            
            for episode in range(self.config.num_episodes):
                # Static
                state = env.reset()
                coord_result, feedback = env.step(state['observations'], use_adaptive_privacy=False)
                if isinstance(feedback, MedicalPrivacyFeedback):
                    static_metrics['diagnostic_accuracy'].append(feedback.diagnostic_accuracy)
                    static_metrics['consensus'].append(feedback.specialist_consensus)
                    static_metrics['compliance'].append(1.0 - coord_result.privacy_loss)
                
                # Adaptive
                state = env.reset()
                coord_result, feedback = env.step(state['observations'], use_adaptive_privacy=True)
                if isinstance(feedback, MedicalPrivacyFeedback):
                    adaptive_metrics['diagnostic_accuracy'].append(feedback.diagnostic_accuracy)
                    adaptive_metrics['consensus'].append(feedback.specialist_consensus)
                    adaptive_metrics['compliance'].append(1.0 - coord_result.privacy_loss)
            
            results.append({
                'domain': 'medical',
                'metric': 'Diagnostic Accuracy',
                'static': np.mean(static_metrics['diagnostic_accuracy']),
                'adaptive': np.mean(adaptive_metrics['diagnostic_accuracy']),
                'improvement': ((np.mean(adaptive_metrics['diagnostic_accuracy']) - 
                               np.mean(static_metrics['diagnostic_accuracy'])) / 
                               np.mean(static_metrics['diagnostic_accuracy']) * 100)
            })
            
            results.append({
                'domain': 'medical',
                'metric': 'Specialist Consensus',
                'static': np.mean(static_metrics['consensus']),
                'adaptive': np.mean(adaptive_metrics['consensus']),
                'improvement': ((np.mean(adaptive_metrics['consensus']) - 
                               np.mean(static_metrics['consensus'])) / 
                               np.mean(static_metrics['consensus']) * 100)
            })
            
            results.append({
                'domain': 'medical',
                'metric': 'Privacy Preservation',
                'static': np.mean(static_metrics['compliance']),
                'adaptive': np.mean(adaptive_metrics['compliance']),
                'improvement': ((np.mean(adaptive_metrics['compliance']) - 
                               np.mean(static_metrics['compliance'])) / 
                               np.mean(static_metrics['compliance']) * 100)
            })
        
        # Finance domain metrics
        if self.config.enable_finance and self.finance_manager:
            logger.info("  Analyzing finance domain performance")
            
            env = EnhancedFinanceEnvironment(
                n_agents, self.finance_manager,
                FinanceConfig(), epsilon
            )
            
            static_metrics = {'returns': [], 'sharpe': [], 'compliance': []}
            adaptive_metrics = {'returns': [], 'sharpe': [], 'compliance': []}
            
            for episode in range(self.config.num_episodes):
                # Static
                state = env.reset()
                coord_result, feedback = env.step(state['observations'], use_adaptive_privacy=False)
                if isinstance(feedback, FinancePrivacyFeedback):
                    static_metrics['returns'].append(feedback.portfolio_return)
                    static_metrics['sharpe'].append(feedback.sharpe_ratio)
                    static_metrics['compliance'].append(feedback.regulatory_compliance)
                
                # Adaptive
                state = env.reset()
                coord_result, feedback = env.step(state['observations'], use_adaptive_privacy=True)
                if isinstance(feedback, FinancePrivacyFeedback):
                    adaptive_metrics['returns'].append(feedback.portfolio_return)
                    adaptive_metrics['sharpe'].append(feedback.sharpe_ratio)
                    adaptive_metrics['compliance'].append(feedback.regulatory_compliance)
            
            results.append({
                'domain': 'finance',
                'metric': 'Portfolio Return',
                'static': np.mean(static_metrics['returns']),
                'adaptive': np.mean(adaptive_metrics['returns']),
                'improvement': ((np.mean(adaptive_metrics['returns']) - 
                               np.mean(static_metrics['returns'])) / 
                               abs(np.mean(static_metrics['returns'])) * 100)
            })
            
            results.append({
                'domain': 'finance',
                'metric': 'Sharpe Ratio',
                'static': np.mean(static_metrics['sharpe']),
                'adaptive': np.mean(adaptive_metrics['sharpe']),
                'improvement': ((np.mean(adaptive_metrics['sharpe']) - 
                               np.mean(static_metrics['sharpe'])) / 
                               np.mean(static_metrics['sharpe']) * 100)
            })
            
            results.append({
                'domain': 'finance',
                'metric': 'Regulatory Compliance',
                'static': np.mean(static_metrics['compliance']),
                'adaptive': np.mean(adaptive_metrics['compliance']),
                'improvement': ((np.mean(adaptive_metrics['compliance']) - 
                               np.mean(static_metrics['compliance'])) / 
                               np.mean(static_metrics['compliance']) * 100)
            })
        
        df = pd.DataFrame(results)
        df.to_csv(f"{self.config.tables_dir}/domain_specific_{self.timestamp}.csv", index=False)
        return df
    
    def _run_attack_resistance_analysis(self) -> pd.DataFrame:
        """Run privacy attack resistance experiments"""
        
        results = []
        n_agents = 20
        epsilon = 1.0
        attack_intensities = [0.1, 0.3, 0.5, 0.7, 0.9]
        
        for domain in ['medical', 'finance']:
            if domain == 'medical' and not self.config.enable_medical:
                continue
            if domain == 'finance' and not self.config.enable_finance:
                continue
            
            dataset_manager = self.medical_manager if domain == 'medical' else self.finance_manager
            if not dataset_manager:
                continue
            
            logger.info(f"  Testing attack resistance in {domain}")
            
            if domain == 'medical':
                env = EnhancedMedicalEnvironment(
                    n_agents, dataset_manager,
                    MedicalConfig(), epsilon
                )
                attack_sim = MedicalPrivacyAttackSimulator(dataset_manager, MedicalConfig())
            else:
                env = EnhancedFinanceEnvironment(
                    n_agents, dataset_manager,
                    FinanceConfig(), epsilon
                )
                attack_sim = FinancePrivacyAttackSimulator(dataset_manager, FinanceConfig())
            
            for attack_type in ['membership', 'attribute']:
                for intensity in attack_intensities:
                    
                    # Simulate attacks
                    num_attacks = 30
                    static_success = []
                    adaptive_success = []
                    
                    for _ in range(num_attacks):
                        if attack_type == 'membership':
                            # Static
                            attack_result = attack_sim.simulate_membership_inference(
                                env, target_case_id=0, num_queries=int(20 * intensity)
                            )
                            static_success.append(attack_result.get('attack_success_rate', 0))
                            
                            # Adaptive
                            env.adaptive_privacy.epsilon = epsilon  # Reset
                            attack_result = attack_sim.simulate_membership_inference(
                                env, target_case_id=0, num_queries=int(20 * intensity)
                            )
                            adaptive_success.append(attack_result.get('attack_success_rate', 0))
                        
                        else:  # attribute
                            attributes = ['age', 'diagnosis'] if domain == 'medical' else ['amount', 'type']
                            
                            # Static
                            attack_result = attack_sim.simulate_attribute_inference(
                                env, target_attributes=attributes, 
                                num_queries=int(15 * intensity)
                            )
                            static_success.append(attack_result.get('success_rate', 0))
                            
                            # Adaptive
                            env.adaptive_privacy.epsilon = epsilon  # Reset
                            attack_result = attack_sim.simulate_attribute_inference(
                                env, target_attributes=attributes,
                                num_queries=int(15 * intensity)
                            )
                            adaptive_success.append(attack_result.get('success_rate', 0))
                    
                    results.append({
                        'domain': domain,
                        'attack_type': f"{attack_type}_inference",
                        'intensity': intensity,
                        'static_success': np.mean(static_success),
                        'adaptive_success': np.mean(adaptive_success),
                        'static_resistance': 1.0 - np.mean(static_success),
                        'adaptive_resistance': 1.0 - np.mean(adaptive_success)
                    })
        
        df = pd.DataFrame(results)
        df.to_csv(f"{self.config.tables_dir}/attack_resistance_{self.timestamp}.csv", index=False)
        return df
    
    def _run_single_experiment(self, domain: str, dataset_manager: Any,
                              epsilon: float, n_agents: int, 
                              num_episodes: int, use_adaptive: bool) -> List[Dict]:
        """Run a single experiment configuration"""
        
        results = []
        
        if domain == 'medical':
            env = EnhancedMedicalEnvironment(
                n_agents, dataset_manager,
                MedicalConfig(), epsilon
            )
        else:
            env = EnhancedFinanceEnvironment(
                n_agents, dataset_manager,
                FinanceConfig(), epsilon
            )
        
        for episode in range(num_episodes):
            state = env.reset()
            coord_result, feedback = env.step(
                state['observations'], 
                use_adaptive_privacy=use_adaptive
            )
            
            result = {
                'episode': episode,
                'utility': coord_result.utility_score,
                'privacy_loss': coord_result.privacy_loss,
                'success': coord_result.success,
                'time': coord_result.coordination_time,
                'rounds': coord_result.communication_rounds,
                'current_epsilon': env.adaptive_privacy.epsilon if use_adaptive else epsilon
            }
            
            results.append(result)
        
        return results
    
    def _save_results(self):
        """Save all results to disk"""
        
        # Save as pickle for later analysis
        with open(f"{self.config.output_dir}/results_{self.timestamp}.pkl", 'wb') as f:
            pickle.dump(self.results, f)
        
        # Save summary JSON
        summary = {
            'timestamp': self.timestamp,
            'config': asdict(self.config),
            'datasets': {
                'medical': self.medical_manager is not None,
                'finance': self.finance_manager is not None
            },
            'experiments': list(self.results.keys()),
            'metrics': {}
        }
        
        # Calculate summary metrics
        for exp_name, df in self.results.items():
            if not df.empty:
                summary['metrics'][exp_name] = {
                    'num_records': len(df),
                    'columns': list(df.columns)
                }
        
        with open(f"{self.config.output_dir}/summary_{self.timestamp}.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Results saved to {self.config.output_dir}")
    
    def _generate_all_figures(self):
        """Generate all figures for the paper"""
        
        logger.info("Generating figures...")
        
        # Figure 1: Privacy-Utility Trade-off
        if 'privacy_utility' in self.results:
            self._plot_privacy_utility_tradeoff()
        
        # Figure 2: Scalability Analysis
        if 'scalability' in self.results:
            self._plot_scalability()
        
        # Figure 3: Environmental Adaptation
        if 'adaptation' in self.results:
            self._plot_adaptation()
        
        # Additional visualizations
        if 'attack_resistance' in self.results:
            self._plot_attack_resistance()
        
        logger.info(f"Figures saved to {self.config.figures_dir}")
    
    def _plot_privacy_utility_tradeoff(self):
        """Generate privacy-utility trade-off plot (Figure 1)"""
        
        df = self.results['privacy_utility']
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Aggregate by epsilon and mechanism
        for domain_idx, domain in enumerate(['medical', 'finance']):
            ax = axes[domain_idx]
            domain_df = df[df['domain'] == domain]
            
            if domain_df.empty:
                continue
            
            # Group by epsilon and mechanism
            grouped = domain_df.groupby(['epsilon', 'mechanism']).agg({
                'utility': ['mean', 'std'],
                'privacy_loss': ['mean', 'std']
            }).reset_index()
            
            # Plot for each mechanism
            for mechanism in ['static', 'adaptive']:
                mech_data = grouped[grouped['mechanism'] == mechanism]
                
                ax.errorbar(
                    mech_data[('epsilon', '')],
                    mech_data[('utility', 'mean')],
                    yerr=mech_data[('utility', 'std')],
                    label=mechanism.capitalize(),
                    marker='o' if mechanism == 'static' else 's',
                    linewidth=2,
                    capsize=5
                )
            
            ax.set_xlabel('Privacy Budget (ε)', fontsize=12)
            ax.set_ylabel('Utility Score', fontsize=12)
            ax.set_title(f'{domain.capitalize()} Domain', fontsize=14, fontweight='bold')
            ax.legend(loc='lower right')
            ax.grid(True, alpha=0.3)
            ax.set_xlim([0, 2.2])
            ax.set_ylim([0, 1])
        
        plt.suptitle('Privacy-Utility Trade-off', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        save_path = f"{self.config.figures_dir}/privacy_utility_tradeoff.{self.config.figure_format}"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Generate Pareto frontier
        self._plot_pareto_frontier(df)
    
    def _plot_pareto_frontier(self, df):
        """Generate Pareto frontier plot"""
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        colors = {'medical': 'blue', 'finance': 'green'}
        markers = {'static': 'o', 'adaptive': 's'}
        
        for domain in ['medical', 'finance']:
            domain_df = df[df['domain'] == domain]
            
            if domain_df.empty:
                continue
            
            for mechanism in ['static', 'adaptive']:
                mech_df = domain_df[domain_df['mechanism'] == mechanism]
                
                # Aggregate
                grouped = mech_df.groupby('epsilon').agg({
                    'utility': 'mean',
                    'privacy_loss': 'mean'
                }).reset_index()
                
                # Plot
                ax.scatter(
                    grouped['privacy_loss'],
                    grouped['utility'],
                    c=colors[domain],
                    marker=markers[mechanism],
                    s=100,
                    alpha=0.7,
                    label=f'{domain}-{mechanism}'
                )
                
                # Connect points
                ax.plot(
                    grouped['privacy_loss'],
                    grouped['utility'],
                    c=colors[domain],
                    alpha=0.3,
                    linestyle='--' if mechanism == 'static' else '-'
                )
        
        ax.set_xlabel('Privacy Loss', fontsize=12)
        ax.set_ylabel('Utility Score', fontsize=12)
        ax.set_title('Privacy-Utility Pareto Frontier', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        save_path = f"{self.config.figures_dir}/pareto_frontier.{self.config.figure_format}"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_scalability(self):
        """Generate scalability plot (Figure 2)"""
        
        df = self.results['scalability']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        for domain_idx, domain in enumerate(['medical', 'finance']):
            domain_df = df[df['domain'] == domain]
            
            if domain_df.empty:
                continue
            
            # Time complexity
            ax1 = axes[0, domain_idx]
            ax1.errorbar(
                domain_df['num_agents'],
                domain_df['avg_time'],
                yerr=domain_df['std_time'],
                marker='o',
                linewidth=2,
                capsize=5,
                label='Empirical'
            )
            
            # Theoretical O(log n)
            x = domain_df['num_agents'].values
            theoretical = domain_df['avg_time'].iloc[0] * np.log(x) / np.log(x[0])
            ax1.plot(x, theoretical, 'r--', alpha=0.5, label='O(log n)')
            
            ax1.set_xlabel('Number of Agents', fontsize=12)
            ax1.set_ylabel('Coordination Time (s)', fontsize=12)
            ax1.set_title(f'{domain.capitalize()} - Time Complexity', fontsize=14)
            ax1.set_yscale('log')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Communication rounds
            ax2 = axes[1, domain_idx]
            ax2.plot(
                domain_df['num_agents'],
                domain_df['avg_rounds'],
                marker='s',
                linewidth=2,
                markersize=8
            )
            
            ax2.set_xlabel('Number of Agents', fontsize=12)
            ax2.set_ylabel('Communication Rounds', fontsize=12)
            ax2.set_title(f'{domain.capitalize()} - Communication Overhead', fontsize=14)
            ax2.grid(True, alpha=0.3)
        
        plt.suptitle('Scalability Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        save_path = f"{self.config.figures_dir}/scalability.{self.config.figure_format}"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_adaptation(self):
        """Generate adaptation plot (Figure 3)"""
        
        df = self.results['adaptation']
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        for domain_idx, domain in enumerate(['medical', 'finance']):
            domain_df = df[df['domain'] == domain]
            
            if domain_df.empty:
                continue
            
            # Epsilon adaptation
            ax1 = axes[0, domain_idx]
            
            # Color by scenario
            scenario_colors = {
                'normal': 'blue',
                'attack': 'red',
                'high_sensitivity': 'orange'
            }
            
            for scenario in ['normal', 'attack', 'high_sensitivity']:
                scenario_df = domain_df[domain_df['scenario'] == scenario]
                if not scenario_df.empty:
                    ax1.scatter(
                        scenario_df['episode'],
                        scenario_df['epsilon'],
                        c=scenario_colors[scenario],
                        alpha=0.6,
                        s=10,
                        label=scenario
                    )
            
            ax1.set_xlabel('Episode', fontsize=12)
            ax1.set_ylabel('Privacy Budget (ε)', fontsize=12)
            ax1.set_title(f'{domain.capitalize()} - Privacy Adaptation', fontsize=14)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Utility over time
            ax2 = axes[1, domain_idx]
            
            # Rolling average
            window = 10
            rolling_utility = domain_df['utility'].rolling(window=window, min_periods=1).mean()
            
            ax2.plot(domain_df['episode'], rolling_utility, linewidth=2)
            ax2.fill_between(
                domain_df['episode'],
                domain_df['utility'].rolling(window=window, min_periods=1).quantile(0.25),
                domain_df['utility'].rolling(window=window, min_periods=1).quantile(0.75),
                alpha=0.3
            )
            
            ax2.set_xlabel('Episode', fontsize=12)
            ax2.set_ylabel('Utility Score', fontsize=12)
            ax2.set_title(f'{domain.capitalize()} - Utility Evolution', fontsize=14)
            ax2.grid(True, alpha=0.3)
        
        plt.suptitle('Environmental Adaptation', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        save_path = f"{self.config.figures_dir}/adaptation.{self.config.figure_format}"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_attack_resistance(self):
        """Generate attack resistance plot"""
        
        df = self.results['attack_resistance']
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        for attack_idx, attack_type in enumerate(['membership_inference', 'attribute_inference']):
            ax = axes[attack_idx]
            
            attack_df = df[df['attack_type'] == attack_type]
            
            if attack_df.empty:
                continue
            
            # Plot for each domain
            for domain in ['medical', 'finance']:
                domain_df = attack_df[attack_df['domain'] == domain]
                
                if domain_df.empty:
                    continue
                
                # Static vs Adaptive
                ax.plot(
                    domain_df['intensity'],
                    domain_df['static_resistance'],
                    marker='o',
                    label=f'{domain}-static',
                    linewidth=2
                )
                
                ax.plot(
                    domain_df['intensity'],
                    domain_df['adaptive_resistance'],
                    marker='s',
                    label=f'{domain}-adaptive',
                    linewidth=2
                )
            
            ax.set_xlabel('Attack Intensity', fontsize=12)
            ax.set_ylabel('Resistance Score', fontsize=12)
            ax.set_title(attack_type.replace('_', ' ').title(), fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1])
        
        plt.suptitle('Privacy Attack Resistance', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        save_path = f"{self.config.figures_dir}/attack_resistance.{self.config.figure_format}"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_all_tables(self):
        """Generate all tables for the paper"""
        
        logger.info("Generating tables...")
        
        # Table 1: Privacy-Utility Summary
        if 'privacy_utility' in self.results:
            self._generate_privacy_utility_table()
        
        # Table 2: Domain-Specific Results
        if 'domain_specific' in self.results:
            self._generate_domain_specific_table()
        
        # Table 3: Attack Resistance
        if 'attack_resistance' in self.results:
            self._generate_attack_resistance_table()
        
        logger.info(f"Tables saved to {self.config.tables_dir}")
    
    def _generate_privacy_utility_table(self):
        """Generate Table 1: Privacy-Utility Trade-off"""
        
        df = self.results['privacy_utility']
        
        # Aggregate results
        summary = df.groupby(['epsilon', 'mechanism']).agg({
            'utility': ['mean', 'std'],
            'privacy_loss': ['mean', 'std']
        }).round(3)
        
        # Format for LaTeX
        if self.config.table_format == 'latex':
            latex_table = summary.to_latex()
            
            # Add caption and label
            latex_table = f"""\\begin{{table}}[h]
\\centering
\\caption{{Privacy-utility trade-off across different mechanisms}}
\\label{{tab:privacy_utility}}
{latex_table}
\\end{{table}}"""
            
            with open(f"{self.config.tables_dir}/table1_privacy_utility.tex", 'w') as f:
                f.write(latex_table)
        
        # Also save as CSV
        summary.to_csv(f"{self.config.tables_dir}/table1_privacy_utility.csv")
    
    def _generate_domain_specific_table(self):
        """Generate Table 2: Domain-Specific Results"""
        
        df = self.results['domain_specific']
        
        # Format for display
        df['improvement'] = df['improvement'].apply(lambda x: f"{x:.1f}%")
        df['static'] = df['static'].round(3)
        df['adaptive'] = df['adaptive'].round(3)
        
        # Pivot for better display
        pivot = df.pivot(index='metric', columns='domain', 
                         values=['static', 'adaptive', 'improvement'])
        
        if self.config.table_format == 'latex':
            latex_table = pivot.to_latex()
            
            latex_table = f"""\\begin{{table}}[h]
\\centering
\\caption{{Domain-specific performance metrics}}
\\label{{tab:domain_results}}
{latex_table}
\\end{{table}}"""
            
            with open(f"{self.config.tables_dir}/table2_domain_specific.tex", 'w') as f:
                f.write(latex_table)
        
        pivot.to_csv(f"{self.config.tables_dir}/table2_domain_specific.csv")
    
    def _generate_attack_resistance_table(self):
        """Generate Table 3: Attack Resistance"""
        
        df = self.results['attack_resistance']
        
        # Aggregate by attack type
        summary = df.groupby(['attack_type', 'domain']).agg({
            'static_resistance': 'mean',
            'adaptive_resistance': 'mean'
        }).round(3)
        
        # Calculate improvement
        summary['improvement'] = ((summary['adaptive_resistance'] - 
                                  summary['static_resistance']) / 
                                  summary['static_resistance'] * 100).round(1)
        
        if self.config.table_format == 'latex':
            latex_table = summary.to_latex()
            
            latex_table = f"""\\begin{{table}}[h]
\\centering
\\caption{{Resistance to privacy attacks}}
\\label{{tab:attacks}}
{latex_table}
\\end{{table}}"""
            
            with open(f"{self.config.tables_dir}/table3_attack_resistance.tex", 'w') as f:
                f.write(latex_table)
        
        summary.to_csv(f"{self.config.tables_dir}/table3_attack_resistance.csv")


def main():
    """Main entry point"""
    
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(description='PrivacyMAS Experiment Pipeline')
    parser.add_argument('--quick', action='store_true', help='Run quick experiments')
    parser.add_argument('--parallel', action='store_true', help='Use parallel processing')
    parser.add_argument('--episodes', type=int, default=100, help='Number of episodes')
    parser.add_argument('--runs', type=int, default=3, help='Number of runs')
    parser.add_argument('--output-dir', type=str, default='experiment_results', help='Output directory')
    
    args = parser.parse_args()
    
    # Configure experiments
    if args.quick:
        config = ExperimentConfig(
            output_dir=args.output_dir,
            epsilon_values=[0.5, 1.0],
            agent_counts=[10, 20],
            num_episodes=30,
            num_runs=1,
            medical_samples=100,
            finance_samples=100
        )
    else:
        config = ExperimentConfig(
            output_dir=args.output_dir,
            epsilon_values=[0.1, 0.5, 1.0, 2.0],
            agent_counts=[10, 20, 50, 100],
            num_episodes=args.episodes,
            num_runs=args.runs,
            medical_samples=500,
            finance_samples=500,
            use_parallel=args.parallel
        )
    
    # Run experiments
    orchestrator = ExperimentOrchestrator(config)
    results = orchestrator.run_complete_experiments()
    
    # Print summary
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    
    for exp_name, df in results.items():
        if not df.empty:
            print(f"\n{exp_name}:")
            print(f"  Records: {len(df)}")
            print(f"  Columns: {list(df.columns)}")
    
    print(f"\nResults saved to: {config.output_dir}")
    print(f"Figures saved to: {config.figures_dir}")
    print(f"Tables saved to: {config.tables_dir}")
    
    print("\n✅ Experiment pipeline completed successfully!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())