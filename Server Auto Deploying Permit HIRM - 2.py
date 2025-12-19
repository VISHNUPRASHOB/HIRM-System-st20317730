# -*- coding: utf-8 -*-
"""
Created on Tue Dec 16 22:03:28 2025

@author: vishnuprashob
"""

# -*- coding: utf-8 -*-
"""
Upgraded HIRM System: Automated Workload & Dynamic Node Scaling
Key Features:
1. Toggleable Auto-Runner (Same button to Start/Stop)
2. Automated Request Generation with workload spikes (>100 units)
3. Dynamic Edge Activation based on workload thresholds
4. Node Jumping: Automatic jumping to another node if current node lacks space/memory
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
# DATA STRUCTURES & CORE LOGIC
# ============================================================================

class DeploymentLocation(Enum):
    EDGE = "edge"
    CLOUD = "cloud"
    QUEUED = "queued"

@dataclass
class EdgeNode:
    node_id: str
    efficiency: float
    cost: float
    bandwidth_limit: float
    deployment_time: float
    feature_vector: List[float]
    current_load: float = 0.0
    memory_capacity: float = 2048.0  # MB
    current_memory_used: float = 0.0
    is_active: bool = False
    deployed_workloads: List[str] = field(default_factory=list)
    total_processed: int = 0
    success_rate: float = 1.0

@dataclass
class WorkloadRequest:
    request_id: str
    cpu_requirement: float
    memory_requirement: float
    bandwidth_requirement: float
    latency_sensitivity: float
    duration_estimate: float
    data_size: float = 100.0
    priority: int = 1

@dataclass
class DeploymentDecision:
    location: DeploymentLocation
    selected_nodes: List[str] = field(default_factory=list)
    node_allocations: List[dict] = field(default_factory=list)
    total_cost: float = 0.0
    reason: str = ""

# ============================================================================
# UPGRADED HIRM SYSTEM MANAGER
# ============================================================================

class HIRMSystem:
    def __init__(self, nodes: List[EdgeNode], logger=None):
        self.nodes = nodes
        self.logger = logger or print
        self.active_allocations = []
        self.stats = {
            'total_requests': 0, 'edge_deployments': 0, 'cloud_deployments': 0,
            'total_cost': 0.0, 'nodes_activated': 0
        }

    def log(self, message: str, level: str = "INFO"):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.logger(f"[{timestamp}] [{level}] {message}")

    def process_request(self, workload: WorkloadRequest) -> DeploymentDecision:
        """
        Logic for Edge Activation, Deployment, and Node Jumping
        """
        self.log(f"New Request {workload.request_id}: CPU Req: {workload.cpu_requirement:.1f}", "STEP")
        
        # 1. TRIGGER EDGE DEVICE ACTIVATION
        # If request is high workload (>100 units) or current active capacity is low
        active_nodes = [n for n in self.nodes if n.is_active]
        total_available_cpu = sum(100 - n.current_load for n in active_nodes)

        if workload.cpu_requirement >= 100 or total_available_cpu < workload.cpu_requirement:
            inactive = [n for n in self.nodes if not n.is_active]
            if inactive:
                new_node = inactive[0]
                new_node.is_active = True
                active_nodes.append(new_node)
                self.stats['nodes_activated'] += 1
                self.log(f"LIMIT REACHED: Deploying new Edge Device {new_node.node_id}", "SUCCESS")

        # 2. NODE JUMPING & LOAD BALANCING
        # Jump to another node if there is no sufficient space or memory
        decision = DeploymentDecision(location=DeploymentLocation.EDGE)
        remaining_cpu = workload.cpu_requirement
        
        # Sort nodes to find best available (Node Jumping sequence)
        sorted_nodes = sorted(active_nodes, key=lambda x: (100 - x.current_load), reverse=True)
        
        for node in sorted_nodes:
            if remaining_cpu <= 0: break
            
            # Check space (CPU) and Memory
            avail_cpu = 100 - node.current_load
            avail_mem = node.memory_capacity - node.current_memory_used
            
            if avail_cpu > 5 and avail_mem > (workload.memory_requirement * 0.1):
                # Calculate how much this node can take
                cpu_share = min(remaining_cpu, avail_cpu)
                mem_share = (cpu_share / workload.cpu_requirement) * workload.memory_requirement
                
                # Allocation
                node.current_load += cpu_share
                node.current_memory_used += mem_share
                node.deployed_workloads.append(workload.request_id)
                
                self.active_allocations.append({
                    'request_id': workload.request_id,
                    'node_id': node.node_id,
                    'cpu': cpu_share,
                    'mem': mem_share,
                    'end_time': time.time() + workload.duration_estimate
                })
                
                decision.selected_nodes.append(node.node_id)
                remaining_cpu -= cpu_share
                self.log(f"Allocated to {node.node_id}. Jumping to next if needed...", "DEBUG")
            else:
                self.log(f"Node {node.node_id} full. Jumping to another node...", "WARNING")

        if remaining_cpu > 0:
            # If even after jumping nodes we can't fit it, fallback to Cloud
            decision.location = DeploymentLocation.CLOUD
            decision.reason = "Edge Cluster full - Cloud Fallback"
            self.stats['cloud_deployments'] += 1
        else:
            self.stats['edge_deployments'] += 1
            decision.reason = "Distributed across Edge"

        self.stats['total_requests'] += 1
        return decision

    def release_workload(self, req_id):
        to_remove = [a for a in self.active_allocations if a['request_id'] == req_id]
        for alloc in to_remove:
            for node in self.nodes:
                if node.node_id == alloc['node_id']:
                    node.current_load = max(0, node.current_load - alloc['cpu'])
                    node.current_memory_used = max(0, node.current_memory_used - alloc['mem'])
                    if req_id in node.deployed_workloads: node.deployed_workloads.remove(req_id)
            self.active_allocations.remove(alloc)

# ============================================================================
# GUI WITH AUTO-RUN FUNCTIONALITY
# ============================================================================

class HIRMSystemGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HIRM System - Auto-Runner & Node Scaling")
        self.setGeometry(100, 100, 1200, 800)
        
        self.nodes = self.create_nodes()
        self.hirm = HIRMSystem(self.nodes, logger=self.log_to_terminal)
        
        self.auto_mode = False
        self.request_counter = 0
        
        # Timer for Auto-Runner
        self.auto_timer = QTimer()
        self.auto_timer.timeout.connect(self.generate_auto_request)
        
        # Timer for cleaning up finished tasks
        self.cleanup_timer = QTimer()
        self.cleanup_timer.timeout.connect(self.cleanup)
        self.cleanup_timer.start(2000)

        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)

        # Left Control Panel
        controls = QVBoxLayout()
        
        self.auto_btn = QPushButton("▶ START AUTO-GENERATOR")
        self.auto_btn.setFixedHeight(60)
        self.auto_btn.setStyleSheet("background-color: #27ae60; color: white; font-weight: bold; font-size: 14px;")
        self.auto_btn.clicked.connect(self.toggle_auto_mode)
        controls.addWidget(self.auto_btn)

        self.node_table = QTableWidget(len(self.nodes), 4)
        self.node_table.setHorizontalHeaderLabels(["Node ID", "Status", "CPU Load %", "Mem (MB)"])
        controls.addWidget(self.node_table)
        
        self.stats_label = QLabel("Ready for Workload...")
        self.stats_label.setStyleSheet("font-weight: bold; color: #2c3e50;")
        controls.addWidget(self.stats_label)

        # Right Log Terminal
        self.terminal = QTextEdit()
        self.terminal.setReadOnly(True)
        self.terminal.setStyleSheet("background-color: #1e1e1e; color: #00ff00; font-family: Consolas;")
        
        layout.addLayout(controls, 1)
        layout.addWidget(self.terminal, 2)
        self.update_ui()

    def toggle_auto_mode(self):
        """Toggle auto-run by clicking the same button to start/stop"""
        self.auto_mode = not self.auto_mode
        if self.auto_mode:
            self.auto_btn.setText("■ STOP AUTO-GENERATOR")
            self.auto_btn.setStyleSheet("background-color: #e74c3c; color: white; font-weight: bold; font-size: 14px;")
            self.auto_timer.start(1000) # Generate every 1 second
        else:
            self.auto_btn.setText("▶ START AUTO-GENERATOR")
            self.auto_btn.setStyleSheet("background-color: #27ae60; color: white; font-weight: bold; font-size: 14px;")
            self.auto_timer.stop()

    def generate_auto_request(self):
        """Auto-generates requests with workload spikes to trigger node activation"""
        self.request_counter += 1
        
        # High workload logic: Randomly spike above 100 to force edge device deployment
        if random.random() > 0.7:
            cpu = random.uniform(105, 150) # Spike
            self.log_to_terminal("[ALERT] High Workload Spike Triggered!")
        else:
            cpu = random.uniform(20, 60) # Normal

        workload = WorkloadRequest(
            f"REQ-{self.request_counter}", cpu, random.uniform(200, 800),
            20, 50, random.uniform(5, 12)
        )
        self.hirm.process_request(workload)
        self.update_ui()

    def cleanup(self):
        curr = time.time()
        completed = set([a['request_id'] for a in self.hirm.active_allocations if curr >= a['end_time']])
        for rid in completed: self.hirm.release_workload(rid)
        self.update_ui()

    def update_ui(self):
        for i, node in enumerate(self.nodes):
            self.node_table.setItem(i, 0, QTableWidgetItem(node.node_id))
            status = "ACTIVE" if node.is_active else "OFFLINE"
            self.node_table.setItem(i, 1, QTableWidgetItem(status))
            self.node_table.setItem(i, 2, QTableWidgetItem(f"{node.current_load:.1f}%"))
            self.node_table.setItem(i, 3, QTableWidgetItem(f"{node.current_memory_used:.0f} MB"))
        
        s = self.hirm.stats
        self.stats_label.setText(f"Requests: {s['total_requests']} | Edge: {s['edge_deployments']} | Cloud: {s['cloud_deployments']}")

    def log_to_terminal(self, msg):
        self.terminal.append(msg)
        self.terminal.moveCursor(QTextCursor.End)

    def create_nodes(self):
        return [EdgeNode(f"Edge-Node-{i:02d}", 0.8, 0.05, 100, 2, [0.9]*4, is_active=(i==0)) for i in range(8)]

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = HIRMSystemGUI()
    gui.show()
    sys.exit(app.exec_())