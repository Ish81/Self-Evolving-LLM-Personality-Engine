# Liquid Personality LLM

**Self-Evolving LLM Personality using a Neural ODE (Liquid Neural Network)**

---

## Overview

This project implements a **self-evolving personality layer** for a locally running Large Language Model (LLM).  
Instead of fine-tuning the LLM or relying on fixed rules, the system models personality as a **continuous dynamical system** that evolves over time based on user behavior.

The personality dynamics are governed by a **Neural Ordinary Differential Equation (Neural ODE)**, inspired by **Liquid Neural Networks**.  
The LLM itself remains frozen; only the personality controller evolves.

---

## Key Idea

Most LLMs respond statically across conversations. This project introduces:

- A continuous personality state  
- Smooth evolution over time  
- Adaptation to user tone, brevity, and pressure  
- No rule-based switching  
- No LLM fine-tuning  

Personality **emerges from dynamics** rather than being predefined.

---

## Personality State

The system maintains a latent personality vector:

| Dimension | Description |
|--------|------------|
| Warmth | Empathy and friendliness |
| Verbosity | Response length and detail |
| Technicality | Depth and formality |
| Patience | Tolerance to pressure |

These values are **continuous** and evolve smoothly over the conversation.

---

## Feature Extraction

Each user message is converted into numerical features:

- **Sentiment** (VADER compound score)
- **Message length** (normalized word count)
- **Latency** (time gap between messages)

These features act as external inputs to the personality dynamics.

---

## System Architecture

### High-Level Flow

User Input
|
v
Feature Extraction
(sentiment, length, latency)
|
v
Liquid Neural Network
(Neural ODE Personality State)
|
v
System Prompt Controller
|
v
Local LLM (Ollama + Qwen2)
|
v
Assistant Response


---

### Internal Liquid Neural Network (LNN)

    Personality State s(t)
 [warmth, verbosity, technicality, patience]
               |
               v
    Concatenate(s(t), user_features)
               |
               v
    Neural Network (vector field f)
               |
         ds/dt = f(s, u)
               |
    Euler Integration + Damping
               |
         Updated State s(t+Δt)

---

## Mathematical Formulation

The personality evolution is modeled as:

\[
\frac{d\mathbf{s}}{dt} = f_\theta(\mathbf{s}, \mathbf{u}) - \lambda \mathbf{s}
\]

Where:

- \( \mathbf{s} \) = personality state  
- \( \mathbf{u} \) = user feature vector  
- \( f_\theta \) = neural network (vector field)  
- \( \lambda \) = damping factor (stability)

This ensures smooth evolution and prevents runaway behavior.

---

## Stability Mechanisms 

To ensure stable dynamics:

- Small time-step Euler integration  
- State damping toward neutral  
- Output clamping  
- Tanh bounding of states  

These prevent oscillations and saturation.

---

## Visualization & Evaluation

Personality states are logged at every turn and saved as a NumPy array.

A plotting script generates a trajectory graph showing:

- Smooth changes over time  
- Gradual response to user pressure  
- No abrupt jumps  

This provides visual proof of liquid behavior.

---

## Project Structure

├── main_step8.py # Final system (Steps 1–8)
├── plot_states.py # Personality trajectory plotting
├── state_trajectory.npy # Logged personality states
├── liquid_personality_states.png # Generated plot
└── README.md


---

## How to Run

### Requirements

- Python 3.9+
- Ollama running locally
- Qwen2 model pulled in Ollama

---

### Install Dependencies

```
pip install torch vaderSentiment numpy matplotlib requests
```
Run the System 
```
python main_ode.py
```
Have a conversation, then type:
```
exit
```
Plot Personality Evolution 
```
python plot_states.py
```
This will generate the file:
```
liquid_personality_states.png
```

### Why This Is Novel

- No fine-tuning of the LLM
- No discrete personality labels
- No rule-based behavior switching
- Continuous-time adaptation
- Fully local and explainable
- Personality emerges from dynamics

---

### Applications

- Conversational agents with long-term adaptation
- Research on human–AI interaction
- Emotion-aware assistants
- Personal AI companions
- Explainable adaptive systems
