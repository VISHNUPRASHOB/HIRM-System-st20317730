# Hybrid Intelligent Resource Management (HIRM)

## Overview
The **Hybrid Intelligent Resource Management (HIRM)** system is a Python-based framework designed to optimize **serverless cloud and edge resource allocation** under dynamic workloads. The system integrates **machine learning, reinforcement learning, and greedy optimization techniques** to minimize deployment cost while maintaining performance stability and scalability.

HIRM is particularly relevant to **modern cloud providers (AWS, Azure, Google Cloud)** and **edge computing platforms**, where efficient allocation of compute, bandwidth, and deployment resources is critical.

---

## Key Objectives
- Minimize operational and deployment costs  
- Optimize edge and cloud resource utilization  
- Maintain low latency for latency-sensitive workloads  
- Prevent unstable deployment oscillations  
- Support real-time, automated workload scheduling  

---

## Core Architecture
HIRM follows a **hierarchical decision pipeline**:

1. **Random Forest Load Balancer**  
   Predicts workload intensity and classifies incoming API requests.

2. **PPO-Inspired Cost-Aware Scheduler**  
   Decides between edge, cloud, or queued execution based on cost, latency, and capacity.

3. **DDPG-Inspired Node Selector**  
   Selects optimal edge nodes using a weighted feature-based greedy approximation.

4. **Hysteresis Stability Controller**  
   Prevents frequent deployment and removal cycles using upper and lower thresholds.

5. **Auto-Scaling & Execution Manager**  
   Dynamically activates or deactivates edge nodes under workload spikes.

---

## Algorithms Used
- **Random Forest (ML):** Workload prediction and classification  
- **Proximal Policy Optimization (PPO):** Cost-aware scheduling decisions  
- **Deep Deterministic Policy Gradient (DDPG):** Continuous action-space node selection  
- **Greedy Approximation:** Fast near-optimal node scoring  
- **Hysteresis Thresholding:** Deployment stability control  
- **Linear Programming (Conceptual):** Constraint-based optimization modeling  

---

## Mathematical Foundation
HIRM formulates resource allocation as a **constrained optimization problem**, maximizing efficiency and heuristic value while minimizing cost and deployment latency.  
The solution combines **greedy approximation rules** with **reinforcement learning-guided policies** to remain computationally feasible for real-time systems.

---

## Features
- Automated request generation and scheduling  
- Real-time edge node scaling  
- Cost and latency monitoring  
- GUI-based system visualization  
- Real-time execution logs  
- Modular and extensible design  

---

## Technology Stack
- **Language:** Python  
- **GUI:** PyQt5  
- **Algorithms:** ML + RL + Greedy Optimization  
- **Visualization:** Real-time metrics and logs  
- **Version Control:** Git & GitHub  

---

## Use Cases
- Serverless cloud platforms  
- Edge-cloud hybrid infrastructures  
- IoT workload orchestration  
- Cost-sensitive cloud applications  
- Research and academic experimentation  

---

## Repository Structure
