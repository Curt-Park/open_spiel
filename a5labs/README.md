## Test Environment
- MacBook Pro 16 (M2), 32GB RAM

## Prerequisites
- Install [brew](https://brew.sh/)
- Install mise for python dev env: `brew install mise`
- Install python: `mise trust && mise install`

## Setup
- Reference: ./docs/install.md
```bash
git clone git@github.com:Curt-Park/open_spiel.git
cd open_spiel

# Install system packages (e.g. cmake) and download some dependencies
./install.sh
# Building and testing from source
uv pip install -r requirements.txt
./open_spiel/scripts/build_and_run_tests.sh

# Building and testing using PIP
uv pip install .
echo "export PYTHONPATH=$PYTHONPATH:$PWD" >> .venv/bin/activate
echo "export PYTHONPATH=$PYTHONPATH:$PWD/build/python" >> .venv/bin/activate

# Run an example
./build/examples/example --game=tic_tac_toe
```

## Kuhn Poker Example with DQN Agent

### Setup
Install additional packages:
```bash
uv pip install -r a5labs/requirements.txt
```

### Configuration
The script uses a YAML configuration file (`a5labs/config.yaml`) for hyperparameters:
- Network architecture (hidden layers)
- Training hyperparameters (learning rate, batch size, etc.)
- Evaluation settings
- Model save path

### Usage

#### Training Mode (Default)
Train a DQN agent from scratch:
```bash
python a5labs/kuhn_poker_dqn.py
```

Train with custom config file:
```bash
python a5labs/kuhn_poker_dqn.py --config path/to/custom_config.yaml
```

#### Evaluation Mode
Evaluate a trained model:
```bash
python a5labs/kuhn_poker_dqn.py --eval
```

The script will:
1. Load the model from the path specified in `config.yaml`
2. Run evaluation episodes against a random agent
3. Report the average reward

## System Design & Data Flow

The following diagram illustrates the system architecture and data flow for the Kuhn Poker DQN example:

```mermaid
graph TB
    subgraph "Python Layer"
        Script[kuhn_poker_dqn.py]
        DQN[DQN Agent<br/>PyTorch]
        RLEnv[RL Environment<br/>rl_environment.py]
    end
    
    subgraph "Python-C++ Bridge"
        Pybind11[Pybind11<br/>pyspiel.cc]
        PyState[Python State<br/>Wrapper]
    end
    
    subgraph "C++ Core Engine"
        CppState[C++ State<br/>spiel.h]
        CppGame[C++ Game<br/>Kuhn Poker]
    end
    
    Script -->|creates| DQN
    Script -->|creates| RLEnv
    RLEnv -->|queries| PyState
    PyState -->|calls| Pybind11
    Pybind11 -->|binds| CppState
    CppState -->|manages| CppGame
    
    DQN -->|step| RLEnv
    RLEnv -->|observations| DQN
```

### Data Flow Sequence

```mermaid
sequenceDiagram
    participant DQN as DQN Agent
    participant RLEnv as RL Environment
    participant PyState as Python State
    participant CppState as C++ State
    participant CppGame as C++ Game Engine
    
    DQN->>RLEnv: step(time_step)
    RLEnv->>PyState: state.observation_tensor(player_id)
    PyState->>CppState: ObservationTensor(player_id)
    CppState->>CppGame: Compute observation tensor
    CppGame-->>CppState: std::vector<float>
    CppState-->>PyState: Python list[float]
    PyState-->>RLEnv: observation tensor
    RLEnv-->>DQN: TimeStep(observations, rewards)
    
    DQN->>DQN: Select action (epsilon-greedy)
    DQN->>RLEnv: step([action])
    RLEnv->>PyState: state.apply_action(action)
    PyState->>CppState: ApplyAction(action)
    CppState->>CppGame: Update game state
    CppGame-->>CppState: New state
    CppState-->>PyState: Updated state
    PyState-->>RLEnv: Updated Python state
    RLEnv-->>DQN: New TimeStep
```

### Key Components

1. **Python Layer**: `kuhn_poker_dqn.py` orchestrates the DQN agent and RL environment
2. **RL Environment**: Wraps the C++ game engine with an RL-friendly API
3. **Pybind11 Bridge**: Provides seamless type conversion between C++ and Python
4. **C++ Core Engine**: Implements game logic (Kuhn Poker) and state management
5. **Data Conversion**: Automatic conversion of `std::vector<float>` ↔ Python `list[float]` via Pybind11

## Profiling
top ten functions consuming most compute time.

## Optimization
.