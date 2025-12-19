# -*- coding: utf-8 -*-
"""
Created on Tue Dec 16 23:50:34 2025

@author: vishnuprashob
"""

"""
Intelligent Resource Management System for Serverless Edge Computing with PyQt5 GUI

Hybrid Intelligent Resource Manager (HIRM) Implementation:
1. Load balancer with Random Forest for workload distribution
2. PPO-inspired cost-aware scheduling with learning
3. DDPG-inspired node selection with adaptive feature weights
4. Hysteresis thresholding for deployment stability
5. Dynamic node management (add/remove/edit nodes)
6. Workload distribution across active nodes
7. Automatic node activation for heavy loads

Time Complexity: O(n*m + n*log(n)) where n=nodes, m=features
Space Complexity: O(n*m) for storing node features and scores
"""

import sys
import numpy as np
import random
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import time
from datetime import datetime
from collections import deque

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QTextEdit, QPushButton, QLabel, 
                             QGroupBox, QGridLayout, QTableWidget, QTableWidgetItem,
                             QSpinBox, QDoubleSpinBox, QComboBox, QSplitter,
                             QProgressBar, QFrame, QDialog, QLineEdit, QMessageBox,
                             QCheckBox, QScrollArea)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QFont, QColor, QTextCursor

# ============================================================================
# DATA STRUCTURES
# ============================================================================

class DeploymentLocation(Enum):
    """Deployment location options"""
    EDGE = "edge"
    CLOUD = "cloud"
    QUEUED = "queued"

@dataclass
class EdgeNode:
    """Represents a heterogeneous edge device"""
    node_id: str
    efficiency: float  # 0-1
    cost: float  # $/hour base rate
    bandwidth_limit: float  # Mbps
    deployment_time: float  # seconds
    feature_vector: List[float]  # [reliability, availability, power_efficiency, network_latency]
    current_load: float = 0.0  # 0-100
    is_active: bool = False
    deployed_workloads: List[str] = field(default_factory=list)
    total_processed: int = 0
    success_rate: float = 1.0
    
    def __post_init__(self):
        """Validate node parameters"""
        assert 0 <= self.efficiency <= 1, "Efficiency must be in [0,1]"
        assert self.cost >= 0, "Cost must be non-negative"
        assert self.bandwidth_limit > 0, "Bandwidth must be positive"
        assert self.deployment_time >= 0, "Deployment time must be non-negative"

@dataclass
class WorkloadRequest:
    """Represents an incoming API request workload"""
    request_id: str
    cpu_requirement: float
    memory_requirement: float
    bandwidth_requirement: float
    latency_sensitivity: float  # ms
    duration_estimate: float  # seconds
    data_size: float = 100.0  # MB
    priority: int = 1  # 1-5, higher is more important

@dataclass
class WorkloadAllocation:
    """Tracks how workload is allocated to nodes"""
    request_id: str
    node_id: str
    allocated_cpu: float
    allocated_memory: float
    allocated_bandwidth: float
    start_time: float
    estimated_end_time: float

@dataclass
class DeploymentDecision:
    """Results of deployment decision"""
    location: DeploymentLocation
    selected_nodes: List[str] = field(default_factory=list)
    node_allocations: List[WorkloadAllocation] = field(default_factory=list)
    total_cost: float = 0.0
    expected_latency: float = 0.0
    reason: str = ""
    queued: bool = False

# ============================================================================
# RANDOM FOREST-BASED LOAD BALANCER
# ============================================================================

class SimpleDecisionTree:
    """Simplified decision tree for load prediction and balancing"""
    def __init__(self, max_depth: int = 5, tree_id: int = 0):
        self.max_depth = max_depth
        self.tree_id = tree_id
        self.tree = self._build_tree()
    
    def _build_tree(self) -> Dict:
        """Build a simple rule-based tree with variation"""
        # Different trees have slightly different thresholds for diversity
        cpu_thresh = 50 + (self.tree_id * 5)
        mem_thresh = 1024 + (self.tree_id * 256)
        lat_thresh = 100 - (self.tree_id * 10)
        
        return {
            'feature': 'cpu_requirement',
            'threshold': cpu_thresh,
            'left': {
                'feature': 'memory_requirement', 
                'threshold': mem_thresh, 
                'left': {'value': 'low'}, 
                'right': {'value': 'medium'}
            },
            'right': {
                'feature': 'latency_sensitivity', 
                'threshold': lat_thresh,
                'left': {'value': 'high'}, 
                'right': {'value': 'medium'}
            }
        }
    
    def predict(self, features: Dict[str, float]) -> str:
        """Predict load category"""
        node = self.tree
        while 'value' not in node:
            feature = node['feature']
            threshold = node['threshold']
            if features.get(feature, 0) < threshold:
                node = node['left']
            else:
                node = node['right']
        return node['value']

class LoadBalancer:
    """
    Random Forest-based load balancer for workload distribution
    Time Complexity: O(k*d + n) where k=trees, d=depth, n=active_nodes
    """
    def __init__(self, num_trees: int = 5):
        self.num_trees = num_trees
        self.trees = [SimpleDecisionTree(tree_id=i) for i in range(num_trees)]
        self.load_history = deque(maxlen=100)
    
    def predict_load_category(self, workload: WorkloadRequest) -> str:
        """
        Predict workload category using ensemble voting
        Returns: 'low', 'medium', or 'high'
        """
        features = {
            'cpu_requirement': workload.cpu_requirement,
            'memory_requirement': workload.memory_requirement,
            'bandwidth_requirement': workload.bandwidth_requirement,
            'latency_sensitivity': workload.latency_sensitivity
        }
        
        predictions = [tree.predict(features) for tree in self.trees]
        vote_counts = {'low': 0, 'medium': 0, 'high': 0}
        for pred in predictions:
            vote_counts[pred] = vote_counts.get(pred, 0) + 1
        
        category = max(vote_counts, key=vote_counts.get)
        self.load_history.append((workload.request_id, category))
        return category
    
    def distribute_workload(self, workload: WorkloadRequest, 
                          active_nodes: List[EdgeNode]) -> List[Tuple[EdgeNode, float]]:
        """
        Distribute workload across active edge nodes based on capacity
        Time Complexity: O(n) where n=number of active nodes
        
        Returns: List of (node, workload_fraction) tuples
        """
        if not active_nodes:
            return []
        
        # Calculate available capacity for each node
        node_capacities = []
        total_available_capacity = 0.0
        
        for node in active_nodes:
            # Available capacity is inverse of current load
            available = (100 - node.current_load) / 100.0
            # Weight by efficiency
            weighted_capacity = available * node.efficiency
            node_capacities.append((node, weighted_capacity))
            total_available_capacity += weighted_capacity
        
        if total_available_capacity == 0:
            return []
        
        # Distribute workload proportionally
        distribution = []
        for node, capacity in node_capacities:
            fraction = capacity / total_available_capacity
            distribution.append((node, fraction))
        
        return distribution

# ============================================================================
# PPO-INSPIRED COST-AWARE SCHEDULING WITH LEARNING
# ============================================================================

class PPOScheduler:
    """
    Proximal Policy Optimization-inspired scheduler with cost learning
    Maintains value function and policy updates based on deployment outcomes
    """
    def __init__(self, edge_cost_weight: float = 0.3, 
                 cloud_cost_weight: float = 0.5,
                 latency_weight: float = 0.2):
        self.edge_cost_weight = edge_cost_weight
        self.cloud_cost_weight = cloud_cost_weight
        self.latency_weight = latency_weight
        
        # PPO parameters
        self.value_threshold = 0.6
        self.exploration_rate = 0.1
        self.learning_rate = 0.01
        
        # Value function history for learning
        self.value_history = deque(maxlen=100)
        self.reward_history = deque(maxlen=100)
        
        # Cost tracking for optimization
        self.total_edge_cost = 0.0
        self.total_cloud_cost = 0.0
        self.deployment_count = 0
    
    def compute_policy_value(self, workload: WorkloadRequest, 
                            load_category: str,
                            active_nodes: List[EdgeNode],
                            available_inactive_nodes: List[EdgeNode]) -> Tuple[DeploymentLocation, float]:
        """
        Compute policy value with PPO-style advantage estimation
        Time Complexity: O(1)
        """
        cpu_norm = min(workload.cpu_requirement / 100, 1.0)
        latency_norm = min(workload.latency_sensitivity / 500, 1.0)
        
        # Compute edge deployment value
        edge_value = 0.0
        edge_capacity = sum((100 - n.current_load) for n in active_nodes)
        
        if edge_capacity > 0:
            # Edge benefits: low latency, distributed processing
            latency_benefit = (1 - latency_norm) * self.latency_weight
            cost_benefit = (1 - cpu_norm) * self.edge_cost_weight
            
            # Capacity factor: how much free capacity exists
            capacity_factor = min(edge_capacity / (workload.cpu_requirement * 2), 1.0)
            
            edge_value = (latency_benefit + cost_benefit) * capacity_factor
            
            # Load category adjustment
            load_multipliers = {'low': 1.2, 'medium': 1.0, 'high': 0.7}
            edge_value *= load_multipliers.get(load_category, 1.0)
            
            # PPO advantage: compare to historical average
            if self.value_history:
                avg_value = np.mean(self.value_history)
                advantage = edge_value - avg_value
                edge_value += self.learning_rate * advantage
        
        # Compute cloud deployment value
        cloud_value = 0.0
        capacity_benefit = cpu_norm * self.cloud_cost_weight
        flexibility_benefit = latency_norm * (1 - self.latency_weight)
        cloud_value = capacity_benefit + flexibility_benefit
        
        load_multipliers = {'low': 0.8, 'medium': 1.0, 'high': 1.3}
        cloud_value *= load_multipliers.get(load_category, 1.0)
        
        # Exploration noise
        if random.random() < self.exploration_rate:
            edge_value += random.uniform(-0.1, 0.1)
            cloud_value += random.uniform(-0.1, 0.1)
        
        # Decision with option to activate new nodes
        if edge_value > cloud_value and edge_value > self.value_threshold:
            # Check if we need to activate more nodes for heavy loads
            if load_category == 'high' and edge_capacity < workload.cpu_requirement * 1.5:
                if available_inactive_nodes:
                    location = DeploymentLocation.EDGE
                    value = edge_value
                else:
                    # Queue if no nodes available
                    location = DeploymentLocation.QUEUED
                    value = edge_value * 0.5
            else:
                location = DeploymentLocation.EDGE
                value = edge_value
        else:
            location = DeploymentLocation.CLOUD
            value = cloud_value
        
        # Store value for learning
        self.value_history.append(value)
        
        return location, value
    
    def update_from_deployment(self, decision: DeploymentDecision, actual_cost: float):
        """Update policy based on deployment outcome (PPO update)"""
        # Compute reward: negative cost (we want to minimize)
        reward = -actual_cost
        
        if decision.location == DeploymentLocation.EDGE:
            self.total_edge_cost += actual_cost
        elif decision.location == DeploymentLocation.CLOUD:
            self.total_cloud_cost += actual_cost
        
        self.reward_history.append(reward)
        self.deployment_count += 1
        
        # Adjust weights based on cost efficiency
        if len(self.reward_history) >= 10:
            recent_rewards = list(self.reward_history)[-10:]
            avg_reward = np.mean(recent_rewards)
            
            # If rewards improving, increase exploration
            if len(self.reward_history) >= 20:
                older_rewards = list(self.reward_history)[-20:-10]
                if np.mean(recent_rewards) > np.mean(older_rewards):
                    self.exploration_rate = min(0.2, self.exploration_rate + 0.01)
                else:
                    self.exploration_rate = max(0.05, self.exploration_rate - 0.01)

# ============================================================================
# DDPG-INSPIRED NODE SELECTION WITH ADAPTIVE WEIGHTS
# ============================================================================

class DDPGNodeSelector:
    """
    Deep Deterministic Policy Gradient-inspired node selection
    Adaptive feature weights that learn from deployment success
    """
    def __init__(self, feature_weights: Optional[Dict[str, float]] = None):
        # Initial feature weights (DDPG "actor" network weights)
        self.feature_weights = feature_weights or {
            'efficiency': 0.35,
            'cost': -0.25,
            'bandwidth': 0.20,
            'deployment_time': -0.10,
            'reliability': 0.10,
            'availability': 0.10,
            'power_efficiency': 0.05,
            'network_latency': -0.05
        }
        
        # DDPG parameters
        self.learning_rate = 0.01
        self.weight_history = deque(maxlen=50)
        self.performance_history = deque(maxlen=50)
        
        # Track weight updates
        self.weight_updates = 0
    
    def _build_feature_matrix(self, nodes: List[EdgeNode]) -> np.ndarray:
        """
        Build feature matrix X where X[i,j] = x(v,j) for node v and feature j
        Time Complexity: O(n*m) where n=nodes, m=features
        """
        n = len(nodes)
        # Base features + heuristic features
        num_features = 4 + len(nodes[0].feature_vector)
        
        X = np.zeros((n, num_features))
        
        for i, node in enumerate(nodes):
            X[i, 0] = node.efficiency
            X[i, 1] = node.cost
            X[i, 2] = node.bandwidth_limit
            X[i, 3] = node.deployment_time
            
            # Heuristic features from feature_vector
            for j, feat in enumerate(node.feature_vector):
                X[i, 4 + j] = feat
        
        # Normalize to [0, 1]
        for j in range(num_features):
            col_max = X[:, j].max()
            col_min = X[:, j].min()
            if col_max > col_min:
                X[:, j] = (X[:, j] - col_min) / (col_max - col_min)
        
        return X
    
    def _compute_deployment_score(self, node: EdgeNode, 
                                  feature_matrix: np.ndarray,
                                  node_idx: int) -> float:
        """
        Compute weighted deployment score: S(v) = Σ(w_j * x(v,j))
        Time Complexity: O(m) where m=num_features
        """
        score = 0.0
        features = feature_matrix[node_idx]
        
        # Map features to weights
        feature_names = ['efficiency', 'cost', 'bandwidth', 'deployment_time']
        
        # Add heuristic feature names
        heuristic_names = ['reliability', 'availability', 'power_efficiency', 'network_latency']
        feature_names.extend(heuristic_names[:len(node.feature_vector)])
        
        # Compute weighted sum
        for i, feat_name in enumerate(feature_names):
            weight = self.feature_weights.get(feat_name, 0.0)
            score += weight * features[i]
        
        # Success rate factor (DDPG "critic" feedback)
        score *= node.success_rate
        
        # Load penalty
        load_penalty = node.current_load / 100.0
        score *= (1 - 0.3 * load_penalty)
        
        return score
    
    def select_nodes(self, nodes: List[EdgeNode], 
                    workload: WorkloadRequest,
                    max_nodes: int = 3,
                    only_active: bool = True) -> List[Tuple[EdgeNode, float]]:
        """
        Select best nodes using DDPG-inspired scoring
        Time Complexity: O(n*m + n*log(n))
        """
        if not nodes:
            return []
        
        # Filter nodes
        candidate_nodes = []
        for node in nodes:
            if only_active and not node.is_active:
                continue
            if (node.bandwidth_limit < workload.bandwidth_requirement or
                node.current_load > 95):
                continue
            candidate_nodes.append(node)
        
        if not candidate_nodes:
            return []
        
        # Build feature matrix
        feature_matrix = self._build_feature_matrix(candidate_nodes)
        
        # Compute scores
        node_scores = []
        for i, node in enumerate(candidate_nodes):
            score = self._compute_deployment_score(node, feature_matrix, i)
            node_scores.append((node, score))
        
        # Sort by score
        node_scores.sort(key=lambda x: x[1], reverse=True)
        
        return node_scores[:max_nodes]
    
    def update_weights_from_performance(self, node: EdgeNode, success: bool):
        """
        Update feature weights based on deployment performance (DDPG update)
        """
        # Update node success rate
        node.total_processed += 1
        alpha = 0.1  # Learning rate for success rate
        node.success_rate = (1 - alpha) * node.success_rate + alpha * (1.0 if success else 0.0)
        
        # Store performance
        self.performance_history.append((node.node_id, success))
        
        # Update weights periodically
        if len(self.performance_history) >= 10:
            recent_success_rate = sum(1 for _, s in list(self.performance_history)[-10:] if s) / 10.0
            
            # If success rate is low, adjust weights
            if recent_success_rate < 0.7:
                # Increase weight on reliability and efficiency
                self.feature_weights['reliability'] += self.learning_rate * 0.1
                self.feature_weights['efficiency'] += self.learning_rate * 0.05
                self.feature_weights['cost'] += self.learning_rate * 0.05  # Less cost-focused
                self.weight_updates += 1
            elif recent_success_rate > 0.9:
                # Can afford to be more cost-conscious
                self.feature_weights['cost'] -= self.learning_rate * 0.05
                self.weight_updates += 1
            
            # Normalize weights
            total_positive = sum(w for w in self.feature_weights.values() if w > 0)
            if total_positive > 0:
                for key in self.feature_weights:
                    if self.feature_weights[key] > 0:
                        self.feature_weights[key] /= (total_positive / 0.8)

# ============================================================================
# HYSTERESIS THRESHOLDING
# ============================================================================

class HysteresisController:
    """
    Hysteresis thresholding for stable deployment decisions
    Prevents oscillation in node activation/deactivation
    """
    def __init__(self, upper_threshold: float = 0.75, 
                 lower_threshold: float = 0.35,
                 cooldown_period: float = 30.0):
        self.upper_threshold = upper_threshold
        self.lower_threshold = lower_threshold
        self.cooldown_period = cooldown_period
        self.last_change_time = {}
        self.state_history = deque(maxlen=100)
    
    def should_activate(self, node_id: str, score: float, force: bool = False) -> bool:
        """Check if node should be activated"""
        current_time = time.time()
        last_change = self.last_change_time.get(node_id, 0)
        
        # Force activation for heavy loads
        if force:
            self.last_change_time[node_id] = current_time
            self.state_history.append((node_id, 'activate', score, current_time))
            return True
        
        # Check cooldown
        if current_time - last_change < self.cooldown_period:
            return False
        
        # Check threshold
        if score > self.upper_threshold:
            self.last_change_time[node_id] = current_time
            self.state_history.append((node_id, 'activate', score, current_time))
            return True
        
        return False
    
    def should_deactivate(self, node_id: str, score: float) -> bool:
        """Check if node should be deactivated"""
        current_time = time.time()
        last_change = self.last_change_time.get(node_id, 0)
        
        # Check cooldown
        if current_time - last_change < self.cooldown_period:
            return False
        
        # Check threshold
        if score < self.lower_threshold:
            self.last_change_time[node_id] = current_time
            self.state_history.append((node_id, 'deactivate', score, current_time))
            return True
        
        return False
    
    def get_cooldown_remaining(self, node_id: str) -> float:
        """Get remaining cooldown time for a node"""
        current_time = time.time()
        last_change = self.last_change_time.get(node_id, 0)
        remaining = self.cooldown_period - (current_time - last_change)
        return max(0, remaining)

# ============================================================================
# HYBRID INTELLIGENT RESOURCE MANAGER (HIRM)
# ============================================================================

class HIRMSystem:
    """
    Hierarchical Intelligent Resource Manager
    Integrates: Load Balancer, PPO Scheduler, DDPG Selector, Hysteresis Control
    
    Overall Time Complexity: O(n*m + n*log(n) + k*d) per request
    Overall Space Complexity: O(n*m + k*d + h) where h=history size
    """
    def __init__(self, nodes: List[EdgeNode], logger=None):
        self.nodes = nodes
        self.logger = logger or print
        
        # Core components
        self.load_balancer = LoadBalancer(num_trees=5)
        self.ppo_scheduler = PPOScheduler()
        self.ddpg_selector = DDPGNodeSelector()
        self.hysteresis = HysteresisController()
        
        # Cloud parameters
        self.cloud_base_cost = 0.10  # $/hour base
        self.cloud_cpu_cost = 0.015  # $ per CPU unit per hour
        self.cloud_bandwidth_cost = 0.01  # $ per GB
        self.cloud_latency = 50.0  # ms
        
        # Edge cost parameters
        self.edge_bandwidth_cost = 0.005  # $ per GB
        self.edge_deployment_cost = 0.02  # $ per deployment
        self.edge_energy_cost = 0.001  # $ per hour per efficiency point
        
        # Workload queue for when all nodes busy
        self.workload_queue = deque()
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'edge_deployments': 0,
            'cloud_deployments': 0,
            'queued_requests': 0,
            'total_cost': 0.0,
            'total_latency': 0.0,
            'total_edge_cost': 0.0,
            'total_cloud_cost': 0.0,
            'nodes_activated': 0,
            'nodes_deactivated': 0
        }
        
        # Active allocations
        self.active_allocations = []
    
    def log(self, message: str, level: str = "INFO"):
        """Log message with timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        formatted = f"[{timestamp}] [{level}] {message}"
        self.logger(formatted)
    
    def calculate_edge_cost(self, workload: WorkloadRequest, 
                           nodes_used: List[EdgeNode],
                           duration: float) -> float:
        """
        Calculate total edge deployment cost
        Cost = Σ(node_base_cost * duration) + bandwidth_cost + deployment_cost + energy_cost
        """
        total_cost = 0.0
        
        for node in nodes_used:
            # Base hourly cost
            hourly_cost = node.cost * (duration / 3600.0)
            total_cost += hourly_cost
            
            # Energy cost (based on efficiency)
            energy_cost = self.edge_energy_cost * (1 - node.efficiency) * (duration / 3600.0)
            total_cost += energy_cost
        
        # Deployment cost (one-time)
        total_cost += self.edge_deployment_cost * len(nodes_used)
        
        # Bandwidth cost
        data_gb = workload.data_size / 1024.0
        bandwidth_cost = data_gb * self.edge_bandwidth_cost
        total_cost += bandwidth_cost
        
        return total_cost
    
    def calculate_cloud_cost(self, workload: WorkloadRequest, duration: float) -> float:
        """Calculate cloud deployment cost"""
        # Base cost
        cost = self.cloud_base_cost * (duration / 3600.0)
        
        # CPU cost
        cpu_cost = workload.cpu_requirement * self.cloud_cpu_cost * (duration / 3600.0)
        cost += cpu_cost
        
        # Bandwidth cost
        data_gb = workload.data_size / 1024.0
        bandwidth_cost = data_gb * self.cloud_bandwidth_cost
        cost += bandwidth_cost
        
        return cost
    
    def get_active_nodes(self) -> List[EdgeNode]:
        """Get list of currently active nodes"""
        return [n for n in self.nodes if n.is_active]
    
    def get_inactive_nodes(self) -> List[EdgeNode]:
        """Get list of inactive nodes"""
        return [n for n in self.nodes if not n.is_active]
    
    def process_request(self, workload: WorkloadRequest) -> DeploymentDecision:
        """
        Process incoming request through HIRM pipeline
        
        Pipeline:
        1. Load prediction (Random Forest) - O(k*d)
        2. Active node check - O(n)
        3. Load balancing (if edge deployment) - O(n)
        4. Cost-aware scheduling (PPO) - O(1)
        5. Node selection (DDPG) - O(n*m + n*log(n))
        6. Hysteresis evaluation - O(1)
        7. Cost calculation - O(n)
        """
        self.log("="*70, "INFO")
        self.log(f"Processing Request: {workload.request_id}", "INFO")
        self.log(f"Priority: {workload.priority}, CPU: {workload.cpu_requirement}, "
                f"Latency: {workload.latency_sensitivity}ms", "INFO")
        self.log("="*70, "INFO")
        
        # Step 1: Load Prediction
        self.log("Step 1: Load Prediction (Random Forest)", "STEP")
        load_category = self.load_balancer.predict_load_category(workload)
        self.log(f"Workload classified as: {load_category.upper()}", "RESULT")
        
        # Step 2: Check Active Nodes
        self.log("Step 2: Checking Active Edge Nodes", "STEP")
        active_nodes = self.get_active_nodes()
        inactive_nodes = self.get_inactive_nodes()
        
        self.log(f"Active nodes: {len(active_nodes)}, Inactive nodes: {len(inactive_nodes)}", "RESULT")
        
        # Calculate total available capacity
        total_capacity = sum((100 - n.current_load) for n in active_nodes)
        required_capacity = workload.cpu_requirement * 1.2  # 20% overhead
        
        self.log(f"Available capacity: {total_capacity:.1f}, Required: {required_capacity:.1f}", "RESULT")
        
        decision = DeploymentDecision(location=DeploymentLocation.EDGE)
        
        # Step 3: Determine if we need to activate more nodes
        need_more_nodes = False
        if total_capacity < required_capacity:
            self.log("Insufficient capacity in active nodes", "WARNING")
            if inactive_nodes:
                self.log(f"Heavy load detected - attempting to activate {len(inactive_nodes)} available node(s)", "STEP")
                need_more_nodes = True
            else:
                self.log("No inactive nodes available - request will be queued or sent to cloud", "WARNING")
        
        # Step 4: Cost-Aware Scheduling (PPO)
        self.log("Step 3: PPO-based Cost-Aware Scheduling", "STEP")
        location, policy_value = self.ppo_scheduler.compute_policy_value(
            workload, load_category, active_nodes, inactive_nodes
        )
        self.log(f"Policy value: {policy_value:.3f}, Decision: {location.value.upper()}", "RESULT")
        
        decision.location = location
        
        # Step 5: Handle Edge Deployment
        if location == DeploymentLocation.EDGE:
            self.log("Step 4: Edge Deployment Selected", "STEP")
            
            # Activate additional nodes if needed
            if need_more_nodes and inactive_nodes:
                self.log("Step 4a: Activating Additional Nodes (DDPG Selection)", "STEP")
                
                # Select best inactive nodes using DDPG
                candidate_scores = self.ddpg_selector.select_nodes(
                    inactive_nodes, workload, max_nodes=2, only_active=False
                )
                
                for node, score in candidate_scores:
                    if total_capacity < required_capacity:
                        # Force activation for heavy loads
                        if self.hysteresis.should_activate(node.node_id, score, force=True):
                            node.is_active = True
                            active_nodes.append(node)
                            total_capacity += (100 - node.current_load)
                            self.stats['nodes_activated'] += 1
                            self.log(f"Activated {node.node_id} (score={score:.3f})", "SUCCESS")
            
            # Step 5: Load Balancing Across Active Nodes
            self.log("Step 5: Load Balancing (Random Forest Distribution)", "STEP")
            node_distribution = self.load_balancer.distribute_workload(workload, active_nodes)
            
            if not node_distribution:
                self.log("No suitable nodes available after distribution", "WARNING")
                
                # Queue the request
                if workload.priority >= 3:
                    self.log("High priority request - queuing for next available node", "WARNING")
                    decision.location = DeploymentLocation.QUEUED
                    decision.queued = True
                    decision.reason = "Queued - waiting for node availability"
                    self.workload_queue.append(workload)
                    self.stats['queued_requests'] += 1
                else:
                    self.log("Falling back to cloud deployment", "WARNING")
                    decision.location = DeploymentLocation.CLOUD
                    decision.reason = "No edge capacity - cloud fallback"
            else:
                # Allocate workload to nodes
                self.log(f"Distributing across {len(node_distribution)} node(s)", "RESULT")
                
                allocations = []
                total_cost = 0.0
                selected_nodes = []
                
                for node, fraction in node_distribution:
                    allocated_cpu = workload.cpu_requirement * fraction
                    allocated_memory = workload.memory_requirement * fraction
                    allocated_bandwidth = workload.bandwidth_requirement * fraction
                    
                    # Create allocation
                    allocation = WorkloadAllocation(
                        request_id=workload.request_id,
                        node_id=node.node_id,
                        allocated_cpu=allocated_cpu,
                        allocated_memory=allocated_memory,
                        allocated_bandwidth=allocated_bandwidth,
                        start_time=time.time(),
                        estimated_end_time=time.time() + workload.duration_estimate
                    )
                    
                    allocations.append(allocation)
                    selected_nodes.append(node.node_id)
                    
                    # Update node load
                    node.current_load += allocated_cpu
                    node.deployed_workloads.append(workload.request_id)
                    
                    self.log(f"  {node.node_id}: {fraction*100:.1f}% of workload "
                           f"(CPU: {allocated_cpu:.1f}, Load: {node.current_load:.1f}%)", "DEBUG")
                
                # Calculate edge cost
                nodes_used = [n for n in active_nodes if n.node_id in selected_nodes]
                total_cost = self.calculate_edge_cost(workload, nodes_used, 
                                                     workload.duration_estimate)
                
                # Calculate latency (average deployment time)
                avg_deployment_time = np.mean([n.deployment_time for n in nodes_used])
                expected_latency = avg_deployment_time * 1000  # Convert to ms
                
                decision.node_allocations = allocations
                decision.selected_nodes = selected_nodes
                decision.total_cost = total_cost
                decision.expected_latency = expected_latency
                decision.reason = f"Deployed to {len(selected_nodes)} edge node(s)"
                
                self.stats['edge_deployments'] += 1
                self.stats['total_edge_cost'] += total_cost
                
                self.log(f"Edge deployment successful - Cost: ${total_cost:.4f}", "SUCCESS")
        
        # Step 6: Handle Cloud Deployment
        if decision.location == DeploymentLocation.CLOUD:
            self.log("Step 6: Cloud Deployment", "STEP")
            
            cloud_cost = self.calculate_cloud_cost(workload, workload.duration_estimate)
            
            decision.total_cost = cloud_cost
            decision.expected_latency = self.cloud_latency
            
            if not decision.reason:
                decision.reason = "Cloud deployment selected by policy"
            
            self.stats['cloud_deployments'] += 1
            self.stats['total_cloud_cost'] += cloud_cost
            
            self.log(f"Cloud deployment configured - Cost: ${cloud_cost:.4f}", "SUCCESS")
        
        # Update statistics
        self.stats['total_requests'] += 1
        self.stats['total_cost'] += decision.total_cost
        self.stats['total_latency'] += decision.expected_latency
        
        # Update PPO scheduler with deployment outcome
        self.ppo_scheduler.update_from_deployment(decision, decision.total_cost)
        
        # Update DDPG weights (assume success for now)
        for node in active_nodes:
            if node.node_id in decision.selected_nodes:
                self.ddpg_selector.update_weights_from_performance(node, True)
        
        # Store active allocations
        if decision.node_allocations:
            self.active_allocations.extend(decision.node_allocations)
        
        # Final summary
        self.log("="*70, "INFO")
        self.log("DEPLOYMENT DECISION", "SUCCESS")
        self.log("="*70, "INFO")
        self.log(f"Location: {decision.location.value.upper()}", "RESULT")
        self.log(f"Reason: {decision.reason}", "RESULT")
        if decision.selected_nodes:
            self.log(f"Selected Nodes: {', '.join(decision.selected_nodes)}", "RESULT")
        self.log(f"Total Cost: ${decision.total_cost:.4f}", "RESULT")
        self.log(f"Expected Latency: {decision.expected_latency:.2f} ms", "RESULT")
        if decision.queued:
            self.log(f"Status: QUEUED (position: {len(self.workload_queue)})", "WARNING")
        self.log("="*70, "INFO")
        self.log("", "INFO")
        
        return decision
    
    def release_workload(self, request_id: str):
        """Release a workload from nodes"""
        # Find and remove allocations
        allocations_to_remove = []
        for alloc in self.active_allocations:
            if alloc.request_id == request_id:
                allocations_to_remove.append(alloc)
                
                # Find node and update load
                for node in self.nodes:
                    if node.node_id == alloc.node_id:
                        node.current_load = max(0, node.current_load - alloc.allocated_cpu)
                        if request_id in node.deployed_workloads:
                            node.deployed_workloads.remove(request_id)
                        break
        
        # Remove allocations
        for alloc in allocations_to_remove:
            self.active_allocations.remove(alloc)
        
        self.log(f"Released workload {request_id} from {len(allocations_to_remove)} node(s)", "INFO")
        
        # Check if queued requests can now be processed
        if self.workload_queue:
            self.log("Checking queued requests...", "INFO")
    
    def add_node(self, node: EdgeNode):
        """Add a new node to the system"""
        self.nodes.append(node)
        self.log(f"Added new node: {node.node_id}", "SUCCESS")
    
    def remove_node(self, node_id: str) -> bool:
        """Remove a node from the system"""
        for node in self.nodes:
            if node.node_id == node_id:
                if node.is_active and node.deployed_workloads:
                    self.log(f"Cannot remove {node_id} - has active workloads", "ERROR")
                    return False
                self.nodes.remove(node)
                self.log(f"Removed node: {node_id}", "SUCCESS")
                return True
        return False

# ============================================================================
# PyQt5 GUI
# ============================================================================

class AddNodeDialog(QDialog):
    """Dialog for adding new edge nodes"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add New Edge Node")
        self.setModal(True)
        self.setMinimumWidth(500)
        
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        form_layout = QGridLayout()
        
        # Node ID
        form_layout.addWidget(QLabel("Node ID:"), 0, 0)
        self.node_id_input = QLineEdit()
        self.node_id_input.setPlaceholderText("e.g., edge-06")
        form_layout.addWidget(self.node_id_input, 0, 1)
        
        # Efficiency
        form_layout.addWidget(QLabel("Efficiency (0-1):"), 1, 0)
        self.efficiency_input = QDoubleSpinBox()
        self.efficiency_input.setRange(0, 1)
        self.efficiency_input.setValue(0.8)
        self.efficiency_input.setSingleStep(0.05)
        form_layout.addWidget(self.efficiency_input, 1, 1)
        
        # Cost
        form_layout.addWidget(QLabel("Cost ($/hour):"), 2, 0)
        self.cost_input = QDoubleSpinBox()
        self.cost_input.setRange(0, 1)
        self.cost_input.setValue(0.05)
        self.cost_input.setSingleStep(0.01)
        self.cost_input.setPrefix("$")
        form_layout.addWidget(self.cost_input, 2, 1)
        
        # Bandwidth
        form_layout.addWidget(QLabel("Bandwidth (Mbps):"), 3, 0)
        self.bandwidth_input = QDoubleSpinBox()
        self.bandwidth_input.setRange(10, 1000)
        self.bandwidth_input.setValue(100)
        self.bandwidth_input.setSuffix(" Mbps")
        form_layout.addWidget(self.bandwidth_input, 3, 1)
        
        # Deployment Time
        form_layout.addWidget(QLabel("Deployment Time (sec):"), 4, 0)
        self.deploy_time_input = QDoubleSpinBox()
        self.deploy_time_input.setRange(0.5, 10)
        self.deploy_time_input.setValue(2.0)
        self.deploy_time_input.setSuffix(" sec")
        form_layout.addWidget(self.deploy_time_input, 4, 1)
        
        # Feature Vector
        form_layout.addWidget(QLabel("Reliability (0-1):"), 5, 0)
        self.reliability_input = QDoubleSpinBox()
        self.reliability_input.setRange(0, 1)
        self.reliability_input.setValue(0.9)
        self.reliability_input.setSingleStep(0.05)
        form_layout.addWidget(self.reliability_input, 5, 1)
        
        form_layout.addWidget(QLabel("Availability (0-1):"), 6, 0)
        self.availability_input = QDoubleSpinBox()
        self.availability_input.setRange(0, 1)
        self.availability_input.setValue(0.85)
        self.availability_input.setSingleStep(0.05)
        form_layout.addWidget(self.availability_input, 6, 1)
        
        form_layout.addWidget(QLabel("Power Efficiency (0-1):"), 7, 0)
        self.power_input = QDoubleSpinBox()
        self.power_input.setRange(0, 1)
        self.power_input.setValue(0.8)
        self.power_input.setSingleStep(0.05)
        form_layout.addWidget(self.power_input, 7, 1)
        
        form_layout.addWidget(QLabel("Network Latency (0-1):"), 8, 0)
        self.network_input = QDoubleSpinBox()
        self.network_input.setRange(0, 1)
        self.network_input.setValue(0.7)
        self.network_input.setSingleStep(0.05)
        form_layout.addWidget(self.network_input, 8, 1)
        
        # Active checkbox
        self.active_checkbox = QCheckBox("Start as Active")
        form_layout.addWidget(self.active_checkbox, 9, 0, 1, 2)
        
        layout.addLayout(form_layout)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.add_button = QPushButton("Add Node")
        self.add_button.clicked.connect(self.accept)
        self.add_button.setStyleSheet("background-color: #27ae60; color: white; font-weight: bold; padding: 10px;")
        button_layout.addWidget(self.add_button)
        
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(cancel_button)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def get_node_data(self) -> EdgeNode:
        """Get the node data from inputs"""
        return EdgeNode(
            node_id=self.node_id_input.text(),
            efficiency=self.efficiency_input.value(),
            cost=self.cost_input.value(),
            bandwidth_limit=self.bandwidth_input.value(),
            deployment_time=self.deploy_time_input.value(),
            feature_vector=[
                self.reliability_input.value(),
                self.availability_input.value(),
                self.power_input.value(),
                self.network_input.value()
            ],
            is_active=self.active_checkbox.isChecked()
        )

class HIRMSystemGUI(QMainWindow):
    """Main GUI for Hierarchical Intelligent Resource Manager"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hybrid Intelligent Resource Manager (HIRM)")
        self.setGeometry(50, 50, 1600, 1000)
        
        # Initialize system
        self.nodes = self.create_example_nodes()
        self.hirm = HIRMSystem(self.nodes, logger=self.log_to_terminal)
        
        self.request_counter = 0
        
        # Auto-release timer
        self.release_timer = QTimer()
        self.release_timer.timeout.connect(self.auto_release_workloads)
        self.release_timer.start(5000)  # Check every 5 seconds
        
        # Setup UI
        self.init_ui()
        self.setStyleSheet(self.get_stylesheet())
    
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # Title
        title = QLabel("🚀 Hybrid Intelligent Resource Manager (HIRM)")
        title_font = QFont("Arial", 20, QFont.Bold)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #2c3e50; padding: 15px; background-color: #ecf0f1; border-radius: 5px;")
        main_layout.addWidget(title)
        
        # Subtitle
        subtitle = QLabel("Load Balancer (Random Forest) → PPO Scheduler → DDPG Node Selector → Hysteresis Control")
        subtitle.setFont(QFont("Arial", 10))
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("color: #7f8c8d; padding: 5px;")
        main_layout.addWidget(subtitle)
        
        # Create splitter
        splitter = QSplitter(Qt.Horizontal)
        
        # Left panel
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(10)
        
        # Workload input
        workload_group = self.create_workload_group()
        left_layout.addWidget(workload_group)
        
        # Node management
        node_mgmt_group = self.create_node_management_group()
        left_layout.addWidget(node_mgmt_group)
        
        # Node status table
        node_status_group = self.create_node_status_group()
        left_layout.addWidget(node_status_group)
        
        left_layout.addStretch()
        
        # Right panel
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Statistics
        stats_group = self.create_statistics_group()
        right_layout.addWidget(stats_group)
        
        # Terminal
        terminal_label = QLabel("📊 System Terminal Output (Real-time Logs)")
        terminal_label.setFont(QFont("Arial", 12, QFont.Bold))
        right_layout.addWidget(terminal_label)
        
        self.terminal = QTextEdit()
        self.terminal.setReadOnly(True)
        self.terminal.setFont(QFont("Consolas", 9))
        self.terminal.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #d4d4d4;
                border: 2px solid #3498db;
                border-radius: 5px;
                padding: 10px;
            }
        """)
        right_layout.addWidget(self.terminal)
        
        # Clear button
        clear_btn = QPushButton("🗑️ Clear Terminal")
        clear_btn.clicked.connect(self.terminal.clear)
        right_layout.addWidget(clear_btn)
        
        # Add panels to splitter
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        
        main_layout.addWidget(splitter)
        
        # Initial message
        self.log_to_terminal("="*70)
        self.log_to_terminal("HIRM System Initialized Successfully")
        self.log_to_terminal(f"Total Edge Nodes: {len(self.nodes)}")
        self.log_to_terminal(f"Active Nodes: {len(self.hirm.get_active_nodes())}")
        self.log_to_terminal("Components: Random Forest Load Balancer, PPO Scheduler, DDPG Selector, Hysteresis Controller")
        self.log_to_terminal("="*70)
        self.log_to_terminal("")
    
    def create_workload_group(self) -> QGroupBox:
        group = QGroupBox("⚙️ Workload Configuration")
        layout = QGridLayout()
        
        layout.addWidget(QLabel("CPU Requirement:"), 0, 0)
        self.cpu_spin = QDoubleSpinBox()
        self.cpu_spin.setRange(1, 100)
        self.cpu_spin.setValue(30)
        self.cpu_spin.setSuffix(" units")
        layout.addWidget(self.cpu_spin, 0, 1)
        
        layout.addWidget(QLabel("Memory:"), 1, 0)
        self.memory_spin = QDoubleSpinBox()
        self.memory_spin.setRange(128, 4096)
        self.memory_spin.setValue(512)
        self.memory_spin.setSuffix(" MB")
        layout.addWidget(self.memory_spin, 1, 1)
        
        layout.addWidget(QLabel("Bandwidth:"), 2, 0)
        self.bandwidth_spin = QDoubleSpinBox()
        self.bandwidth_spin.setRange(1, 200)
        self.bandwidth_spin.setValue(20)
        self.bandwidth_spin.setSuffix(" Mbps")
        layout.addWidget(self.bandwidth_spin, 2, 1)
        
        layout.addWidget(QLabel("Latency Sensitivity:"), 3, 0)
        self.latency_spin = QDoubleSpinBox()
        self.latency_spin.setRange(10, 500)
        self.latency_spin.setValue(50)
        self.latency_spin.setSuffix(" ms")
        layout.addWidget(self.latency_spin, 3, 1)
        
        layout.addWidget(QLabel("Duration:"), 4, 0)
        self.duration_spin = QDoubleSpinBox()
        self.duration_spin.setRange(10, 3600)
        self.duration_spin.setValue(300)
        self.duration_spin.setSuffix(" sec")
        layout.addWidget(self.duration_spin, 4, 1)
        
        layout.addWidget(QLabel("Data Size:"), 5, 0)
        self.data_size_spin = QDoubleSpinBox()
        self.data_size_spin.setRange(10, 1000)
        self.data_size_spin.setValue(100)
        self.data_size_spin.setSuffix(" MB")
        layout.addWidget(self.data_size_spin, 5, 1)
        
        layout.addWidget(QLabel("Priority:"), 6, 0)
        self.priority_spin = QSpinBox()
        self.priority_spin.setRange(1, 5)
        self.priority_spin.setValue(3)
        layout.addWidget(self.priority_spin, 6, 1)
        
        button_layout = QHBoxLayout()
        
        self.process_btn = QPushButton("🚀 Process Request")
        self.process_btn.clicked.connect(self.process_workload)
        self.process_btn.setStyleSheet("background-color: #27ae60; color: white; font-weight: bold; padding: 10px;")
        button_layout.addWidget(self.process_btn)
        
        self.random_btn = QPushButton("🎲 Random Request")
        self.random_btn.clicked.connect(self.generate_random_workload)
        button_layout.addWidget(self.random_btn)
        
        layout.addLayout(button_layout, 7, 0, 1, 2)
        
        group.setLayout(layout)
        return group
    
    def create_node_management_group(self) -> QGroupBox:
        group = QGroupBox("🔧 Node Management")
        layout = QVBoxLayout()
        
        button_layout = QHBoxLayout()
        
        add_node_btn = QPushButton("➕ Add New Node")
        add_node_btn.clicked.connect(self.add_node_dialog)
        add_node_btn.setStyleSheet("background-color: #3498db; color: white; font-weight: bold;")
        button_layout.addWidget(add_node_btn)
        
        remove_node_btn = QPushButton("➖ Remove Selected Node")
        remove_node_btn.clicked.connect(self.remove_selected_node)
        remove_node_btn.setStyleSheet("background-color: #e74c3c; color: white; font-weight: bold;")
        button_layout.addWidget(remove_node_btn)
        
        layout.addLayout(button_layout)
        
        toggle_layout = QHBoxLayout()
        
        activate_btn = QPushButton("✅ Activate Selected")
        activate_btn.clicked.connect(lambda: self.toggle_node_status(True))
        toggle_layout.addWidget(activate_btn)
        
        deactivate_btn = QPushButton("⏸️ Deactivate Selected")
        deactivate_btn.clicked.connect(lambda: self.toggle_node_status(False))
        toggle_layout.addWidget(deactivate_btn)
        
        layout.addLayout(toggle_layout)
        
        group.setLayout(layout)
        return group
    
    def create_node_status_group(self) -> QGroupBox:
        group = QGroupBox("🖥️ Edge Node Status")
        layout = QVBoxLayout()
        
        self.node_table = QTableWidget()
        self.node_table.setColumnCount(7)
        self.node_table.setHorizontalHeaderLabels([
            "Node ID", "Status", "Efficiency", "Cost ($/hr)", "Load (%)", 
            "Workloads", "Success Rate"
        ])
        self.node_table.horizontalHeader().setStretchLastSection(True)
        self.node_table.setSelectionBehavior(QTableWidget.SelectRows)
        
        self.update_node_table()
        
        layout.addWidget(self.node_table)
        group.setLayout(layout)
        return group
    
    def create_statistics_group(self) -> QGroupBox:
        group = QGroupBox("📈 System Statistics & Metrics")
        layout = QGridLayout()
        
        # Create labels
        self.total_requests_label = QLabel("0")
        self.edge_deploy_label = QLabel("0")
        self.cloud_deploy_label = QLabel("0")
        self.queued_label = QLabel("0")
        self.total_cost_label = QLabel("$0.00")
        self.edge_cost_label = QLabel("$0.00")
        self.cloud_cost_label = QLabel("$0.00")
        self.avg_latency_label = QLabel("0.00 ms")
        self.active_nodes_label = QLabel("0")
        self.nodes_activated_label = QLabel("0")
        
        labels_font = QFont("Arial", 10, QFont.Bold)
        for label in [self.total_requests_label, self.edge_deploy_label, 
                     self.cloud_deploy_label, self.queued_label, self.total_cost_label,
                     self.edge_cost_label, self.cloud_cost_label, self.avg_latency_label,
                     self.active_nodes_label, self.nodes_activated_label]:
            label.setFont(labels_font)
            label.setStyleSheet("color: #2980b9;")
        
        row = 0
        layout.addWidget(QLabel("📊 Request Statistics"), row, 0, 1, 2)
        row += 1
        
        layout.addWidget(QLabel("Total Requests:"), row, 0)
        layout.addWidget(self.total_requests_label, row, 1)
        row += 1
        
        layout.addWidget(QLabel("Edge Deployments:"), row, 0)
        layout.addWidget(self.edge_deploy_label, row, 1)
        row += 1
        
        layout.addWidget(QLabel("Cloud Deployments:"), row, 0)
        layout.addWidget(self.cloud_deploy_label, row, 1)
        row += 1
        
        layout.addWidget(QLabel("Queued Requests:"), row, 0)
        layout.addWidget(self.queued_label, row, 1)
        row += 1
        
        # Progress bar
        self.edge_progress = QProgressBar()
        self.edge_progress.setMaximum(100)
        layout.addWidget(QLabel("Edge Utilization %:"), row, 0)
        layout.addWidget(self.edge_progress, row, 1)
        row += 1
        
        # Cost statistics
        layout.addWidget(QLabel("💰 Cost Statistics"), row, 0, 1, 2)
        row += 1
        
        layout.addWidget(QLabel("Total Cost:"), row, 0)
        layout.addWidget(self.total_cost_label, row, 1)
        row += 1
        
        layout.addWidget(QLabel("Edge Cost:"), row, 0)
        layout.addWidget(self.edge_cost_label, row, 1)
        row += 1
        
        layout.addWidget(QLabel("Cloud Cost:"), row, 0)
        layout.addWidget(self.cloud_cost_label, row, 1)
        row += 1
        
        # Performance statistics
        layout.addWidget(QLabel("⚡ Performance Metrics"), row, 0, 1, 2)
        row += 1
        
        layout.addWidget(QLabel("Avg Latency:"), row, 0)
        layout.addWidget(self.avg_latency_label, row, 1)
        row += 1
        
        layout.addWidget(QLabel("Active Nodes:"), row, 0)
        layout.addWidget(self.active_nodes_label, row, 1)
        row += 1
        
        layout.addWidget(QLabel("Nodes Activated:"), row, 0)
        layout.addWidget(self.nodes_activated_label, row, 1)
        
        group.setLayout(layout)
        return group
    
    def update_node_table(self):
        self.node_table.setRowCount(len(self.nodes))
        
        for i, node in enumerate(self.nodes):
            # Node ID
            self.node_table.setItem(i, 0, QTableWidgetItem(node.node_id))
            
            # Status
            status_item = QTableWidgetItem("🟢 ACTIVE" if node.is_active else "⚪ INACTIVE")
            if node.is_active:
                status_item.setForeground(QColor("#27ae60"))
            else:
                status_item.setForeground(QColor("#95a5a6"))
            self.node_table.setItem(i, 1, status_item)
            
            # Efficiency
            self.node_table.setItem(i, 2, QTableWidgetItem(f"{node.efficiency:.2f}"))
            
            # Cost
            self.node_table.setItem(i, 3, QTableWidgetItem(f"${node.cost:.3f}"))
            
            # Load
            load_item = QTableWidgetItem(f"{node.current_load:.1f}%")
            if node.current_load > 80:
                load_item.setForeground(QColor("#e74c3c"))
            elif node.current_load > 50:
                load_item.setForeground(QColor("#f39c12"))
            else:
                load_item.setForeground(QColor("#27ae60"))
            self.node_table.setItem(i, 4, load_item)
            
            # Workloads
            self.node_table.setItem(i, 5, QTableWidgetItem(str(len(node.deployed_workloads))))
            
            # Success Rate
            self.node_table.setItem(i, 6, QTableWidgetItem(f"{node.success_rate*100:.1f}%"))
    
    def update_statistics(self):
        """Update statistics display with correct calculations"""
        stats = self.hirm.stats
        
        # Basic counts
        self.total_requests_label.setText(str(stats['total_requests']))
        self.edge_deploy_label.setText(str(stats['edge_deployments']))
        self.cloud_deploy_label.setText(str(stats['cloud_deployments']))
        self.queued_label.setText(str(stats['queued_requests']))
        
        # Cost statistics
        self.total_cost_label.setText(f"${stats['total_cost']:.4f}")
        self.edge_cost_label.setText(f"${stats['total_edge_cost']:.4f}")
        self.cloud_cost_label.setText(f"${stats['total_cloud_cost']:.4f}")
        
        # Node activation statistics
        self.nodes_activated_label.setText(str(stats['nodes_activated']))
        
        # Active nodes count
        active_count = len(self.hirm.get_active_nodes())
        self.active_nodes_label.setText(str(active_count))
        
        # Calculate edge utilization percentage
        if stats['total_requests'] > 0:
            # Average latency
            avg_latency = stats['total_latency'] / stats['total_requests']
            self.avg_latency_label.setText(f"{avg_latency:.2f} ms")
            
            # Edge utilization: percentage of requests deployed to edge
            edge_percent = (stats['edge_deployments'] / stats['total_requests']) * 100
            self.edge_progress.setValue(int(edge_percent))
            
            # Update edge utilization label with actual percentage
            self.edge_utilization_label.setText(f"{edge_percent:.1f}%")
        else:
            self.avg_latency_label.setText("0.00 ms")
            self.edge_progress.setValue(0)
            self.edge_utilization_label.setText("0.0%")
        
        # Calculate current system load across all active nodes
        active_nodes = self.hirm.get_active_nodes()
        if active_nodes:
            total_load = sum(n.current_load for n in active_nodes)
            total_capacity = len(active_nodes) * 100
            system_utilization = (total_load / total_capacity) * 100
            self.system_load_label.setText(f"{total_load:.1f} / {total_capacity:.0f} units ({system_utilization:.1f}%)")
        else:
            self.system_load_label.setText("0 / 0 units (0%)")
    
    def log_to_terminal(self, message: str):
        # Color coding
        if "[SUCCESS]" in message:
            color = "#2ecc71"
        elif "[WARNING]" in message:
            color = "#f39c12"
        elif "[ERROR]" in message:
            color = "#e74c3c"
        elif "[STEP]" in message:
            color = "#3498db"
        elif "[RESULT]" in message:
            color = "#9b59b6"
        elif "[DEBUG]" in message:
            color = "#95a5a6"
        else:
            color = "#d4d4d4"
        
        formatted = f'<span style="color: {color};">{message}</span>'
        self.terminal.append(formatted)
        
        cursor = self.terminal.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.terminal.setTextCursor(cursor)
    
    def generate_random_workload(self):
        self.cpu_spin.setValue(random.uniform(10, 90))
        self.memory_spin.setValue(random.choice([256, 512, 1024, 2048]))
        self.bandwidth_spin.setValue(random.uniform(10, 100))
        self.latency_spin.setValue(random.uniform(20, 300))
        self.duration_spin.setValue(random.uniform(100, 600))
        self.data_size_spin.setValue(random.uniform(50, 500))
        self.priority_spin.setValue(random.randint(1, 5))
        
        QTimer.singleShot(100, self.process_workload)
    
    def process_workload(self):
        self.request_counter += 1
        
        workload = WorkloadRequest(
            request_id=f"req-{self.request_counter:03d}",
            cpu_requirement=self.cpu_spin.value(),
            memory_requirement=self.memory_spin.value(),
            bandwidth_requirement=self.bandwidth_spin.value(),
            latency_sensitivity=self.latency_spin.value(),
            duration_estimate=self.duration_spin.value(),
            data_size=self.data_size_spin.value(),
            priority=self.priority_spin.value()
        )
        
        decision = self.hirm.process_request(workload)
        
        self.update_node_table()
        self.update_statistics()
    
    def auto_release_workloads(self):
        """Automatically release completed workloads"""
        current_time = time.time()
        completed = []
        
        for alloc in self.hirm.active_allocations:
            if current_time >= alloc.estimated_end_time:
                completed.append(alloc.request_id)
        
        for req_id in set(completed):
            self.hirm.release_workload(req_id)
        
        if completed:
            self.update_node_table()
            self.update_statistics()
    
    def add_node_dialog(self):
        dialog = AddNodeDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            node = dialog.get_node_data()
            
            # Check if node ID already exists
            if any(n.node_id == node.node_id for n in self.nodes):
                QMessageBox.warning(self, "Error", f"Node ID '{node.node_id}' already exists!")
                return
            
            self.nodes.append(node)
            self.hirm.add_node(node)
            self.update_node_table()
            self.update_statistics()
            
            QMessageBox.information(self, "Success", f"Node '{node.node_id}' added successfully!")
    
    def remove_selected_node(self):
        selected = self.node_table.currentRow()
        if selected < 0:
            QMessageBox.warning(self, "Warning", "Please select a node to remove!")
            return
        
        node = self.nodes[selected]
        
        if node.is_active and node.deployed_workloads:
            QMessageBox.warning(self, "Cannot Remove", 
                              f"Node '{node.node_id}' has active workloads!\n"
                              f"Workloads: {', '.join(node.deployed_workloads)}")
            return
        
        reply = QMessageBox.question(self, "Confirm Removal",
                                     f"Remove node '{node.node_id}'?",
                                     QMessageBox.Yes | QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            if self.hirm.remove_node(node.node_id):
                self.nodes.remove(node)
                self.update_node_table()
                self.update_statistics()
                QMessageBox.information(self, "Success", f"Node '{node.node_id}' removed!")
    
    def toggle_node_status(self, activate: bool):
        selected = self.node_table.currentRow()
        if selected < 0:
            QMessageBox.warning(self, "Warning", "Please select a node!")
            return
        
        node = self.nodes[selected]
        
        if activate and not node.is_active:
            # Check hysteresis
            score = 0.8  # Manual activation score
            if self.hirm.hysteresis.should_activate(node.node_id, score, force=True):
                node.is_active = True
                self.hirm.stats['nodes_activated'] += 1
                self.log_to_terminal(f"[SUCCESS] Manually activated node: {node.node_id}")
            else:
                cooldown = self.hirm.hysteresis.get_cooldown_remaining(node.node_id)
                QMessageBox.warning(self, "Cooldown Active", 
                                  f"Node is in cooldown period.\n"
                                  f"Remaining: {cooldown:.1f} seconds")
                return
        
        elif not activate and node.is_active:
            if node.deployed_workloads:
                QMessageBox.warning(self, "Cannot Deactivate",
                                  f"Node has active workloads!\n"
                                  f"Workloads: {', '.join(node.deployed_workloads)}")
                return
            
            node.is_active = False
            self.hirm.stats['nodes_deactivated'] += 1
            self.log_to_terminal(f"[WARNING] Manually deactivated node: {node.node_id}")
        
        self.update_node_table()
        self.update_statistics()
    
    def create_example_nodes(self) -> List[EdgeNode]:
        return [
            EdgeNode(
                node_id="edge-01",
                efficiency=0.85,
                cost=0.05,
                bandwidth_limit=100.0,
                deployment_time=2.0,
                feature_vector=[0.9, 0.7, 0.85, 0.8],
                is_active=True
            ),
            EdgeNode(
                node_id="edge-02",
                efficiency=0.70,
                cost=0.03,
                bandwidth_limit=80.0,
                deployment_time=3.5,
                feature_vector=[0.8, 0.9, 0.75, 0.85],
                is_active=True
            ),
            EdgeNode(
                node_id="edge-03",
                efficiency=0.95,
                cost=0.08,
                bandwidth_limit=150.0,
                deployment_time=1.5,
                feature_vector=[0.95, 0.85, 0.9, 0.75]
            ),
            EdgeNode(
                node_id="edge-04",
                efficiency=0.60,
                cost=0.02,
                bandwidth_limit=60.0,
                deployment_time=4.0,
                feature_vector=[0.7, 0.8, 0.7, 0.9]
            ),
            EdgeNode(
                node_id="edge-05",
                efficiency=0.88,
                cost=0.06,
                bandwidth_limit=120.0,
                deployment_time=2.5,
                feature_vector=[0.85, 0.75, 0.88, 0.82]
            ),
        ]
    
    def get_stylesheet(self) -> str:
        return """
            QMainWindow {
                background-color: #f5f6fa;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #bdc3c7;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 8px 15px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #21618c;
            }
            QSpinBox, QDoubleSpinBox {
                padding: 5px;
                border: 2px solid #bdc3c7;
                border-radius: 3px;
            }
            QTableWidget {
                border: 2px solid #bdc3c7;
                border-radius: 5px;
                background-color: white;
            }
            QTableWidget::item {
                padding: 5px;
            }
            QHeaderView::section {
                background-color: #34495e;
                color: white;
                padding: 5px;
                border: 1px solid #2c3e50;
                font-weight: bold;
            }
            QProgressBar {
                border: 2px solid #bdc3c7;
                border-radius: 5px;
                text-align: center;
                height: 20px;
            }
            QProgressBar::chunk {
                background-color: #27ae60;
                border-radius: 3px;
            }
        """

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    window = HIRMSystemGUI()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()