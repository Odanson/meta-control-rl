#  Meta-Control via Uncertainty  
### When Should an Agent Explore?

---

## Overview

This project implements a minimal computational model of **meta-control in decision-making**: an agent that learns **not only what to do**, but also **when to explore vs exploit**.

The core idea is that exploration should not be fixed. Instead, it should be **regulated by uncertainty** about the environment.

We demonstrate this in a **multi-armed bandit task** under both stable and volatile conditions.

---

## Problem Setting

We consider a standard **K-armed bandit**:

- At each time step $t$, the agent selects an action $a_t \in \{1, \dots, K\}$
- It receives a reward $r_t \in \{0,1\}$
- Each arm has an unknown reward probability $p_a$

In volatile environments, these probabilities can change over time.

---

## Learning Rule

The agent maintains value estimates $Q(a)$ for each action.

We use a simple incremental update:

$$
Q_{t+1}(a) = Q_t(a) + \alpha \big( r_t - Q_t(a) \big)
$$

where:
- $\alpha$ is the learning rate  
- $r_t - Q_t(a)$ is the **prediction error**

---

## Exploration Strategy

### Baseline (Fixed Exploration)

A standard approach is **epsilon-greedy**:

- With probability $\epsilon$: explore (random action)
- Otherwise: exploit (choose $\arg\max_a Q(a)$)

This uses a **fixed** exploration rate $\epsilon$.

---

## Meta-Control: Adaptive Exploration

In this project, exploration is controlled dynamically:

$$
\epsilon_t = \epsilon_{\min} + \lambda \, U_t
$$

where:
- $U_t$ = uncertainty signal  
- $\lambda$ = scaling factor  

---

## Uncertainty Signals

We combine multiple sources of uncertainty:

---

### 1. Sampling Uncertainty (Counts)

$$
U_{\text{count}}(a) = \frac{1}{N(a) + 1}
$$

- $N(a)$: number of times action $a$ has been selected  
- Less-sampled actions $\Rightarrow$ higher uncertainty  

---

### 2. Value Ambiguity (Gap)

$$
U_{\text{value}} = \frac{1}{\lvert Q_{\text{best}} - Q_{\text{second-best}} \rvert + \varepsilon}
$$

(Here $\varepsilon$ is a small constant for numerical stability, not the exploration rate.)

- Small gap $\Rightarrow$ high ambiguity  
- Encourages exploration when options are similar  

---

### 3. Combined Uncertainty

$$
U_t = w_1 \, U_{\text{count}} + w_2 \, U_{\text{value}}
$$

---

##  Volatility & Surprise

We model environmental change using **prediction error**:

$$
\delta_t = \lvert r_t - Q_t(a_t) \rvert
$$

Large prediction error suggests:
> “Something has changed”

We implement **adaptive forgetting**:

- If $\delta_t > \theta$: strong reset  
- Otherwise: slow decay  

This enables the agent to:
- forget outdated knowledge  
- re-explore after changes  

---

## Temporal Smoothing

To avoid noisy behavior, we smooth exploration:

$$
\epsilon_t^{\text{smooth}} = (1 - \beta)\,\epsilon_{t-1} + \beta\,\epsilon_t
$$

This ensures:
- stable control  
- interpretable dynamics  

---

## Key Result

Across multiple runs:

- Meta-control achieves **higher mean reward**
- It shows **lower variance** (more stable)
- It adapts better in **volatile environments**

---

## Interpretation

The agent integrates:

| Mechanism | Cognitive Interpretation |
|----------|------------------------|
| Counts | Confidence / knowledge |
| Value gap | Decision ambiguity |
| Prediction error | Surprise / change detection |
| Smoothing | Control stability |

---

## Demo

A Pygame simulation visualizes the agent:

- The agent moves between options  
- Exploration is visible as movement variability  
- Press `C` to trigger environmental change  

You can observe:
- exploration → learning → exploitation  
- re-exploration after change  

###️ Controls

- Press `C` to trigger a sudden change in the environment (volatility)

---

## How to Run

```bash
git clone https://github.com/Odanson/meta-control-rl.git
cd meta-control-rl

python3 -m venv .venv
source .venv/bin/activate

pip install -e .
pip install pygame pytest

python3 analyze.py
python3 evaluate_agents.py
python3 pygame_demo.py
```

---

## Requirements

- Python 3.10+
- numpy
- matplotlib
- pygame
- pyyaml
- tqdm


