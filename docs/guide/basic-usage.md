# Basic Usage

Connections provides reinforcement learning environments for connection calculi. This guide will show you how to get started with the basic functionality.

## Quick Start

Here's a simple example of using the classical connection calculus environment:

```python
from connections.calculi.classical import ConnectionEnv

# Create an environment with a problem file
env = ConnectionEnv("problem_path")
observation = env.reset()

# Simple loop to find a proof
while True:
    action = env.action_space[0]  # Take the first available action
    observation, reward, done, info = env.step(action)
    
    if done:
        print(observation.proof_sequence)
        print(info)
        break
```

## Environment Settings

You can customize the environment by specifying the logic and domain:

```python
from connections.env import Settings

# Create settings for S5 modal logic with varying domain
settings = Settings(logic="S5", domain="varying")

# Create environment with custom settings
env = ConnectionEnv("problem_path", settings=settings)
```

### Available Logics
- `"classical"` - Classical first-order logic
- `"intuitionistic"` - Intuitionistic first-order logic
- `"S4"` - Modal logic S4
- `"S5"` - Modal logic S5
- `"D"` - Modal logic D
- `"T"` - Modal logic T

### Domain Types
For modal logics, you can specify the domain type:
- `"constant"` - Constant domain (default)
- `"cumulative"` - Cumulative domain
- `"varying"` - Varying domain

## Working with Proofs

The environment provides detailed information about the proof process:

```python
# Get the proof sequence
print(observation.proof_sequence)

# View the tableau
print(observation.tableau)

# Check if a theorem was proven
print(info["status"])  # "Theorem" if proven
```

## A Worked Example

Let's look at a concrete example using double-negation elimination:

```python
from connections.calculi.classical import ConnectionEnv

# Load the double-negation elimination problem
env = ConnectionEnv("tests/cnf_problems/SYN001+1.p")
observation = env.reset()

while True:
    action = env.action_space[0]
    observation, reward, done, info = env.step(action)
    if done:
        break

# The proof sequence shows how the theorem was proven
print(observation.proof_sequence)
# [st0: [p_defini(1), p_defini(2)], ex0: p_defini(1) -> [-p_defini(1), p], ...]

print(info)
# {'status': 'Theorem'}
```

## Next Steps

- Learn more about the [Environments](environments.md) and their capabilities
- Try out the [PyCoP Provers](pycop.md) for standalone theorem proving
- Explore [TPTP/QMLTP Translation](translation.md) for working with standard problem formats 