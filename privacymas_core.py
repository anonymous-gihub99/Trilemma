import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging
from collections import defaultdict
import time

@dataclass
class CoordinationResult:
    """Results from a coordination episode"""
    success: bool
    utility_score: float
    privacy_loss: float
    coordination_time: float
    communication_rounds: int
    agent_contributions: Dict[int, float]

@dataclass
class PrivacyFeedback:
    """Environmental feedback for privacy adaptation"""
    attack_detected: bool
    utility_degradation: float
    coordination_quality: float
    suggested_epsilon_adjustment: float

class Agent:
    """Individual agent in the multi-agent system"""
    def __init__(self, agent_id: int, capabilities: Dict):
        self.agent_id = agent_id
        self.capabilities = capabilities
        self.local_data = None
        self.coordination_history = []
        
    def generate_action(self, observation: np.ndarray, private: bool = True) -> np.ndarray:
        """Generate action based on observation, optionally with privacy"""
        # Base action generation (replace with domain-specific logic)
        action = np.random.randn(len(observation)) * 0.1 + observation
        
        if private:
            # Add local differential privacy noise
            sensitivity = self.estimate_sensitivity(observation)
            noise = np.random.laplace(0, sensitivity, action.shape)
            action += noise
            
        return action
    
    def estimate_sensitivity(self, data: np.ndarray) -> float:
        """Estimate sensitivity for differential privacy"""
        return np.std(data) * 0.1  # Simplified sensitivity estimation

class DifferentialPrivacyManager:
    """Manages privacy mechanisms across the environment"""
    def __init__(self, initial_epsilon: float = 1.0, delta: float = 1e-5):
        self.epsilon = initial_epsilon
        self.delta = delta
        self.privacy_budget_used = 0.0
        self.attack_detector = PrivacyAttackDetector()
        
    def apply_noise(self, data: np.ndarray, mechanism: str = "laplace") -> np.ndarray:
        """Apply differential privacy noise to data"""
        if mechanism == "laplace":
            return self._apply_laplace_noise(data)
        elif mechanism == "gaussian":
            return self._apply_gaussian_noise(data)
        else:
            raise ValueError(f"Unknown privacy mechanism: {mechanism}")
    
    def _apply_laplace_noise(self, data: np.ndarray) -> np.ndarray:
        """Apply Laplace noise for ε-differential privacy"""
        sensitivity = self._compute_sensitivity(data)
        noise = np.random.laplace(0, sensitivity / self.epsilon, data.shape)
        self.privacy_budget_used += self.epsilon
        return data + noise
    
    def _apply_gaussian_noise(self, data: np.ndarray) -> np.ndarray:
        """Apply Gaussian noise for (ε,δ)-differential privacy"""
        sensitivity = self._compute_sensitivity(data)
        sigma = sensitivity * np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon
        noise = np.random.normal(0, sigma, data.shape)
        self.privacy_budget_used += self.epsilon
        return data + noise
    
    def _compute_sensitivity(self, data: np.ndarray) -> float:
        """Compute L1 sensitivity of the data"""
        return np.max(np.abs(data)) if len(data) > 0 else 1.0
    
    def detect_privacy_attack(self, queries: List[np.ndarray]) -> bool:
        """Detect potential privacy attacks from query patterns"""
        return self.attack_detector.detect_attack(queries)

class PrivacyAttackDetector:
    """Detects various privacy attacks"""
    def __init__(self):
        self.query_history = []
        self.attack_threshold = 0.7
        
    def detect_attack(self, queries: List[np.ndarray]) -> bool:
        """Detect if current queries constitute an attack"""
        self.query_history.extend(queries)
        
        # Simple attack detection based on query similarity
        if len(self.query_history) < 2:
            return False
            
        # Check for repeated similar queries (membership inference indicator)
        recent_queries = self.query_history[-10:]
        similarities = []
        
        for i in range(len(recent_queries)-1):
            for j in range(i+1, len(recent_queries)):
                sim = self._compute_similarity(recent_queries[i], recent_queries[j])
                similarities.append(sim)
        
        if similarities and np.mean(similarities) > self.attack_threshold:
            return True
            
        return False
    
    def _compute_similarity(self, query1: np.ndarray, query2: np.ndarray) -> float:
        """Compute cosine similarity between queries"""
        if len(query1) != len(query2):
            return 0.0
        norm1, norm2 = np.linalg.norm(query1), np.linalg.norm(query2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return np.dot(query1, query2) / (norm1 * norm2)

class HierarchicalCoordination:
    """Implements hierarchical coordination for scalability"""
    def __init__(self, cluster_size: int = 10):
        self.cluster_size = cluster_size
        self.coordination_rounds = 0
        
    def coordinate(self, agents: List[Agent], actions: List[np.ndarray]) -> CoordinationResult:
        """Coordinate agents using hierarchical approach"""
        start_time = time.time()
        n_agents = len(agents)
        
        if n_agents <= self.cluster_size:
            # Direct coordination for small groups
            result = self._direct_coordination(agents, actions)
        else:
            # Hierarchical coordination for large groups
            result = self._hierarchical_coordination(agents, actions)
        
        coordination_time = time.time() - start_time
        result.coordination_time = coordination_time
        result.communication_rounds = self.coordination_rounds
        
        return result
    
    def _direct_coordination(self, agents: List[Agent], actions: List[np.ndarray]) -> CoordinationResult:
        """Direct coordination for small agent groups"""
        self.coordination_rounds = 1
        
        # Simple consensus mechanism
        if not actions:
            return CoordinationResult(False, 0.0, 0.0, 0.0, 0, {})
            
        consensus = np.mean(actions, axis=0)
        
        # Compute utility based on consensus quality
        utility_score = self._compute_utility(actions, consensus)
        
        # Estimate privacy loss from coordination
        privacy_loss = self._estimate_privacy_loss(actions)
        
        agent_contributions = {i: np.linalg.norm(action - consensus) 
                             for i, action in enumerate(actions)}
        
        success = utility_score > 0.5  # Threshold for successful coordination
        
        return CoordinationResult(
            success=success,
            utility_score=utility_score,
            privacy_loss=privacy_loss,
            coordination_time=0.0,  # Will be set by caller
            communication_rounds=self.coordination_rounds,
            agent_contributions=agent_contributions
        )
    
    def _hierarchical_coordination(self, agents: List[Agent], actions: List[np.ndarray]) -> CoordinationResult:
        """Hierarchical coordination for large agent groups"""
        # Cluster agents
        clusters = self._create_clusters(agents, actions)
        cluster_representatives = []
        
        self.coordination_rounds = 2  # Intra-cluster + inter-cluster
        
        # Intra-cluster coordination
        total_utility = 0.0
        total_privacy_loss = 0.0
        all_contributions = {}
        
        for cluster_agents, cluster_actions in clusters:
            cluster_result = self._direct_coordination(cluster_agents, cluster_actions)
            
            # Representative is the cluster consensus
            representative = np.mean(cluster_actions, axis=0)
            cluster_representatives.append(representative)
            
            total_utility += cluster_result.utility_score * len(cluster_agents)
            total_privacy_loss += cluster_result.privacy_loss
            all_contributions.update(cluster_result.agent_contributions)
        
        # Inter-cluster coordination
        if len(cluster_representatives) > 1:
            global_consensus = np.mean(cluster_representatives, axis=0)
            global_utility = self._compute_utility(cluster_representatives, global_consensus)
        else:
            global_utility = total_utility / len(agents)
        
        # Normalize utility by total number of agents
        final_utility = total_utility / len(agents)
        success = final_utility > 0.5
        
        return CoordinationResult(
            success=success,
            utility_score=final_utility,
            privacy_loss=total_privacy_loss,
            coordination_time=0.0,  # Will be set by caller
            communication_rounds=self.coordination_rounds,
            agent_contributions=all_contributions
        )
    
    def _create_clusters(self, agents: List[Agent], actions: List[np.ndarray]) -> List[Tuple[List[Agent], List[np.ndarray]]]:
        """Create agent clusters for hierarchical coordination"""
        clusters = []
        n_agents = len(agents)
        
        for i in range(0, n_agents, self.cluster_size):
            end_idx = min(i + self.cluster_size, n_agents)
            cluster_agents = agents[i:end_idx]
            cluster_actions = actions[i:end_idx]
            clusters.append((cluster_agents, cluster_actions))
        
        return clusters
    
    def _compute_utility(self, actions: List[np.ndarray], consensus: np.ndarray) -> float:
        """Compute coordination utility based on consensus quality"""
        if not actions:
            return 0.0
            
        # Utility is inversely related to variance from consensus
        deviations = [np.linalg.norm(action - consensus) for action in actions]
        avg_deviation = np.mean(deviations)
        
        # Normalize to [0, 1] range
        max_possible_deviation = np.linalg.norm(consensus) * 2  # Rough estimate
        if max_possible_deviation == 0:
            return 1.0
            
        utility = max(0.0, 1.0 - (avg_deviation / max_possible_deviation))
        return utility
    
    def _estimate_privacy_loss(self, actions: List[np.ndarray]) -> float:
        """Estimate privacy loss from coordination process"""
        if not actions:
            return 0.0
            
        # Privacy loss increases with the amount of information shared
        # Simplified model: based on variance in actions
        action_matrix = np.array(actions)
        variance = np.var(action_matrix)
        
        # Normalize to [0, 1] range
        privacy_loss = min(1.0, variance / 10.0)  # Scaling factor
        return privacy_loss

class AdaptivePrivacyManager:
    """Manages adaptive privacy budget based on environmental feedback"""
    def __init__(self, initial_epsilon: float = 1.0, learning_rate: float = 0.01):
        self.epsilon = initial_epsilon
        self.initial_epsilon = initial_epsilon
        self.learning_rate = learning_rate
        self.adaptation_history = []
        self.min_epsilon = 0.1
        self.max_epsilon = 2.0
        
    def update_epsilon(self, feedback: PrivacyFeedback) -> float:
        """Update privacy budget based on environmental feedback"""
        # Compute adaptation signal
        adaptation_signal = self._compute_adaptation_signal(feedback)
        
        # Update epsilon with learning rate
        epsilon_change = self.learning_rate * adaptation_signal
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
            'adaptation_signal': adaptation_signal
        })
        
        self.epsilon = new_epsilon
        return self.epsilon
    
    def _compute_adaptation_signal(self, feedback: PrivacyFeedback) -> float:
        """Compute how much to adjust epsilon based on feedback"""
        signal = 0.0
        
        # If attack detected, decrease epsilon (more privacy)
        if feedback.attack_detected:
            signal -= 0.5
        
        # If utility degrading, increase epsilon (less privacy, more utility)
        if feedback.utility_degradation > 0.3:
            signal += 0.3 * feedback.utility_degradation
        
        # If coordination quality poor, increase epsilon
        if feedback.coordination_quality < 0.5:
            signal += 0.2 * (0.5 - feedback.coordination_quality)
        
        # Use suggested adjustment if available
        if feedback.suggested_epsilon_adjustment != 0:
            signal += 0.1 * feedback.suggested_epsilon_adjustment
        
        return signal

class PrivacyMASEnvironment:
    """Main environment for privacy-preserving multi-agent coordination"""
    def __init__(self, num_agents: int, domain_type: str = "medical", 
                 initial_epsilon: float = 1.0, cluster_size: int = 10):
        self.num_agents = num_agents
        self.domain_type = domain_type
        
        # Initialize components
        self.agents = [Agent(i, {}) for i in range(num_agents)]
        self.privacy_manager = DifferentialPrivacyManager(initial_epsilon)
        self.adaptive_privacy = AdaptivePrivacyManager(initial_epsilon)
        self.coordination_module = HierarchicalCoordination(cluster_size)
        
        # Environment state
        self.episode_count = 0
        self.total_privacy_budget_used = 0.0
        self.coordination_history = []
        
        self.logger = logging.getLogger(__name__)
        
    def reset(self) -> Dict:
        """Reset environment for new episode"""
        self.episode_count += 1
        
        # Generate initial observations for all agents
        observations = self._generate_observations()
        
        state = {
            'observations': observations,
            'episode': self.episode_count,
            'privacy_budget_remaining': 10.0 - self.total_privacy_budget_used,
            'num_agents': self.num_agents
        }
        
        return state
    
    def step(self, observations: List[np.ndarray], use_adaptive_privacy: bool = True) -> Tuple[CoordinationResult, PrivacyFeedback]:
        """Execute one coordination step"""
        # Generate actions from observations
        actions = []
        for i, obs in enumerate(observations):
            action = self.agents[i].generate_action(obs, private=False)  # Privacy applied later
            actions.append(action)
        
        # Apply privacy mechanisms
        if use_adaptive_privacy:
            current_epsilon = self.adaptive_privacy.epsilon
        else:
            current_epsilon = self.privacy_manager.epsilon
            
        private_actions = []
        for action in actions:
            private_action = self.privacy_manager.apply_noise(action)
            private_actions.append(private_action)
        
        # Coordinate agents
        coordination_result = self.coordination_module.coordinate(self.agents, private_actions)
        
        # Detect privacy attacks
        attack_detected = self.privacy_manager.detect_privacy_attack([obs for obs in observations])
        
        # Generate environmental feedback
        feedback = self._generate_feedback(coordination_result, attack_detected)
        
        # Update adaptive privacy if enabled
        if use_adaptive_privacy:
            self.adaptive_privacy.update_epsilon(feedback)
        
        # Update environment state
        self.total_privacy_budget_used += self.privacy_manager.privacy_budget_used
        self.coordination_history.append(coordination_result)
        
        return coordination_result, feedback
    
    def _generate_observations(self) -> List[np.ndarray]:
        """Generate domain-specific observations"""
        if self.domain_type == "medical":
            return self._generate_medical_observations()
        else:
            # Generic observations
            return [np.random.randn(10) for _ in range(self.num_agents)]
    
    def _generate_medical_observations(self) -> List[np.ndarray]:
        """Generate synthetic medical dialogue observations"""
        observations = []
        
        for i in range(self.num_agents):
            # Simulate medical features: symptoms, vitals, history
            symptoms = np.random.binomial(1, 0.3, 20)  # 20 possible symptoms
            vitals = np.random.normal([120, 80, 98.6, 16], [15, 10, 1, 3])  # BP, temp, resp
            risk_factors = np.random.binomial(1, 0.2, 10)  # 10 risk factors
            
            obs = np.concatenate([symptoms, vitals, risk_factors])
            observations.append(obs)
        
        return observations
    
    def _generate_feedback(self, coordination_result: CoordinationResult, 
                          attack_detected: bool) -> PrivacyFeedback:
        """Generate environmental feedback for privacy adaptation"""
        # Compute utility degradation compared to baseline
        baseline_utility = 0.8  # Expected utility without privacy constraints
        utility_degradation = max(0.0, baseline_utility - coordination_result.utility_score)
        
        # Suggest epsilon adjustment based on performance
        if coordination_result.utility_score < 0.5:
            suggested_adjustment = 0.2  # Increase epsilon for better utility
        elif attack_detected:
            suggested_adjustment = -0.3  # Decrease epsilon for better privacy
        else:
            suggested_adjustment = 0.0
        
        return PrivacyFeedback(
            attack_detected=attack_detected,
            utility_degradation=utility_degradation,
            coordination_quality=coordination_result.utility_score,
            suggested_epsilon_adjustment=suggested_adjustment
        )
    
    def get_metrics(self) -> Dict:
        """Get comprehensive environment metrics"""
        if not self.coordination_history:
            return {}
        
        recent_results = self.coordination_history[-10:]  # Last 10 episodes
        
        metrics = {
            'avg_utility': np.mean([r.utility_score for r in recent_results]),
            'avg_privacy_loss': np.mean([r.privacy_loss for r in recent_results]),
            'success_rate': np.mean([r.success for r in recent_results]),
            'avg_coordination_time': np.mean([r.coordination_time for r in recent_results]),
            'avg_communication_rounds': np.mean([r.communication_rounds for r in recent_results]),
            'total_privacy_budget_used': self.total_privacy_budget_used,
            'current_epsilon': self.adaptive_privacy.epsilon,
            'episodes_completed': self.episode_count
        }
        
        return metrics