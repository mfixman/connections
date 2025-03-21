# Connection Environments

The Connections package provides reinforcement learning environments for various connection calculi. These environments follow the OpenAI Gym interface pattern.

## Environment Interface

All connection environments implement the following interface:

```python
observation = env.reset()  # Reset environment and get initial observation
action = env.action_space[0]  # Select an action from the action space
observation, reward, done, info = env.step(action)  # Take a step in the environment
```

## Available Environments

### Classical Logic
```python
from connections.calculi.classical import ConnectionEnv

env = ConnectionEnv("problem_path")
```

### Intuitionistic Logic
```python
from connections.calculi.intuitionistic import ConnectionEnv

env = ConnectionEnv("problem_path")
```

### Modal Logics
```python
from connections.calculi.modal import ConnectionEnv
from connections.env import Settings

# Available modal logics: S4, S5, D, T
settings = Settings(logic="S4", domain="constant")
env = ConnectionEnv("problem_path", settings=settings)
```

## Observation Space

The observation returned by the environment contains:

- `proof_sequence`: List of proof steps taken so far
- `tableau`: Current state of the proof tableau
- `available_actions`: List of valid actions that can be taken
- Additional state information specific to each logic

## Action Space

The action space consists of valid proof steps that can be taken at each state. These include:

- Start steps (`st`)
- Extension steps (`ex`)
- Reduction steps (`re`)
- Logic-specific steps

## Rewards

The environment provides rewards based on:

- Successful proof completion
- Valid/invalid action selection
- Proof length optimization
- Logic-specific criteria

## Example Usage

Here's a complete example showing how to use the environment features:

```python
from connections.calculi.classical import ConnectionEnv
from connections.env import Settings

# Create environment with custom settings
settings = Settings(
    logic="classical",
    max_steps=100,
    reward_config={
        "proof_found": 1.0,
        "invalid_action": -0.1,
        "step_penalty": -0.01
    }
)

env = ConnectionEnv("problem_path", settings=settings)
observation = env.reset()

# Main interaction loop
while True:
    # Get available actions
    actions = observation.available_actions
    
    # Select an action (here we just take the first available)
    action = actions[0]
    
    # Take a step
    observation, reward, done, info = env.step(action)
    
    # Print current state
    print(f"Step reward: {reward}")
    print(f"Proof sequence: {observation.proof_sequence}")
    
    if done:
        print(f"Final status: {info['status']}")
        break
```

## Custom Settings

The `Settings` class allows you to customize various aspects of the environment:

```python
settings = Settings(
    # Logic settings
    logic="S4",  # Logic type
    domain="constant",  # Domain type for modal logics
    
    # Environment settings
    max_steps=100,  # Maximum steps before termination
    
    # Reward configuration
    reward_config={
        "proof_found": 1.0,  # Reward for finding a proof
        "invalid_action": -0.1,  # Penalty for invalid actions
        "step_penalty": -0.01  # Small penalty per step to encourage efficiency
    }
)
```

## Integration with RL Frameworks

While the environments don't directly inherit from `gym.Env`, they follow the same interface pattern and can be easily wrapped for use with popular RL frameworks:

```python
from your_rl_framework import Agent

env = ConnectionEnv("problem_path")
agent = Agent(env.observation_space, env.action_space)

for episode in range(num_episodes):
    observation = env.reset()
    episode_reward = 0
    
    while True:
        action = agent.select_action(observation)
        observation, reward, done, info = env.step(action)
        episode_reward += reward
        
        if done:
            break
            
    agent.update()
``` 