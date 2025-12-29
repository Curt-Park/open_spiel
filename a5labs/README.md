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
Setup additional packages:
```bash
uv pip install -r a5labs/requirements.txt
```

Run:
```bash
python a5labs/kuhn_poker_dqn.py
```