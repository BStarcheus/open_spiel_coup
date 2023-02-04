# Running algorithms on coup

1. Add flag files to `scripts/flags/` to configure runs. See `scripts/flags/examples/`.
2. `./run.sh`
3. Find logs in `data/`


# Playing a human v. bot game
- Save a policy from deep cfr or nfsp to a directory
- Write a flag file using `scripts/flags/examples/human_game-ex.cfg`. Keep the 3 included flags, and add the flags used for training whichever algorithm you're playing. See `scripts/human_game.py` for which flags are required.
- `python3 human_game.py --flagfile=yourflagfile`