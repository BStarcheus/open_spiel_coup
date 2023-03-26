# Coup Experiments

This directory contains:
- scripts for:
    - training agents to play Coup with:
        - Deep CFR
        - NFSP
    - testing agents with:
        - approximate exploitability (RL response)
        - policy analysis
        - agent vs. agent games
        - human vs. agent games
- flagfile examples
- flagfiles for the final trained agents from the thesis
- utilites


## Training agents to play Coup

1. Add one or more flagfiles to `scripts/flags/` to configure algorithm hyperparameters, one flagfile for each agent to be trained. See `scripts/flags/examples/`.
2. From within `scripts/` run `./run.sh`
3. Find logs in `data/` (or where you configured in your flagfile)


## Playing a human v. bot game
- Save a policy from Deep CFR or NFSP to a directory
- Write a flagfile using `scripts/flags/examples/human_game-ex.cfg`. Keep the 3 included flags, and add the flags used for training whichever algorithm you're playing. See `scripts/human_game.py` for which flags are required.
- From within `scripts/` run `python3 human_game.py --flagfile=yourflagfile`