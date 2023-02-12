"""Test how an agent's policy reacts differently to specific game 
states as it learns. Run all tests for both player perspectives, to see
if behavior is learned in both."""

from absl import app
from absl import flags
from absl import logging

import tensorflow.compat.v1 as tf

import pyspiel
import numpy as np

from coup_experiments.utils.get_bots import *

# Temporarily disable TF2 behavior until we update the code.
tf.disable_v2_behavior()

FLAGS = flags.FLAGS

flags.DEFINE_string("game", "coup", "Game name")
flags.DEFINE_string("algo", "deep_cfr", "Algorithm which has a saved model in the given directory. deep_cfr or nfsp")
flags.DEFINE_string("saved_dir", "/agents/deep_cfr/", "Directory with saved model")
flags.DEFINE_string("checkpoint_id", "iter10", "The ID of the checkpoint. deep_cfr: iter{iter-num}, nfsp: ep{ep-num}")

# Deep CFR
flags.DEFINE_integer("num_iterations", 100, "Number of iterations")
flags.DEFINE_integer("num_traversals", 100, "Number of traversals/games")
flags.DEFINE_list("policy_network_layers", [16], 
                  "Number of hidden units in the policy network.")
flags.DEFINE_list("advantage_network_layers", [16], 
                  "Number of hidden units in the advantage network.")
flags.DEFINE_float("learning_rate", 1e-3,
                   "Learning rate for inner rl agent.")
flags.DEFINE_integer("batch_size_advantage", 128,
                     "Batch size to sample from advantage memories.")
flags.DEFINE_integer("batch_size_strategy", 1024,
                     "Batch size to sample from strategy memories.")
flags.DEFINE_integer("memory_capacity", int(1e7),
                     "Number of samples that can be stored in memory.")
flags.DEFINE_integer("policy_network_train_steps", 400,
                     "Number of policy network training steps.")
flags.DEFINE_integer("advantage_network_train_steps", 20,
                     "Number of advantage network training steps.")
flags.DEFINE_bool("reinitialize_advantage_networks", False,
                  "Reinit advantage network before training each iter.")

# NFSP
flags.DEFINE_list("hidden_layers_sizes", [128,], 
                  "Number of hidden units in the avg-net and Q-net.")
flags.DEFINE_integer("replay_buffer_capacity", int(2e5),
                     "Size of the replay buffer.")
flags.DEFINE_integer("reservoir_buffer_capacity", int(2e6),
                     "Size of the reservoir buffer.")
flags.DEFINE_integer("min_buffer_size_to_learn", 1000,
                     "Number of samples in buffer before learning begins.")
flags.DEFINE_float("anticipatory_param", 0.1,
                   "Prob of using the rl best response as episode policy.")
flags.DEFINE_integer("batch_size", 128,
                     "Number of transitions to sample at each learning step.")
flags.DEFINE_integer("learn_every", 64,
                     "Number of steps between learning updates.")
flags.DEFINE_float("rl_learning_rate", 0.01,
                   "Learning rate for inner rl agent.")
flags.DEFINE_float("sl_learning_rate", 0.01,
                   "Learning rate for avg-policy sl network.")
flags.DEFINE_string("optimizer_str", "sgd",
                    "Optimizer, choose from 'adam', 'sgd'.")
flags.DEFINE_string("loss_str", "mse",
                    "Loss function, choose from 'mse', 'huber'.")
flags.DEFINE_integer("update_target_network_every", 19200,
                     "Number of steps between DQN target network updates.")
flags.DEFINE_float("discount_factor", 1.0,
                   "Discount factor for future rewards.")
flags.DEFINE_integer("epsilon_decay_duration", int(20e6),
                     "Number of game steps over which epsilon is decayed.")
flags.DEFINE_float("epsilon_start", 0.06,
                   "Starting exploration parameter.")
flags.DEFINE_float("epsilon_end", 0.001,
                   "Final exploration parameter.")

def log_action_probs(bot, state, indent=0):
  probs, _ = bot.step_with_policy(state)
  sp = " " * indent
  logging.info(sp + f"{probs}")

def foreign_aid_test(game, bots):
  '''
  Test FA prob before and after being blocked by Duke.
  Should see a decrease in prob.
  '''
  logging.info("Foreign aid test")
  for p in range(2):
    logging.info(f"  P{p+1} version")
    bot = bots[p]
    state = game.new_initial_state()
    if p == 0:
      state.apply_action(1) # Ambassador
    state.apply_action(4) # Duke
    state.apply_action(3) # Contessa
    state.apply_action(4) # Duke
    if p == 1:
      state.apply_action(1) # Ambassador
      state.apply_action(0) # Income
    for i in range(2):
      logging.info(f"  FA attempt {i+1}")
      log_action_probs(bot, state, indent=2)
      state.apply_action(1) # FA
      state.apply_action(10) # Block
      state.apply_action(9) # Pass
      state.apply_action(3) # Tax
      state.apply_action(9) # Pass
    logging.info("  FA attempt 3")
    log_action_probs(bot, state, indent=2)
  logging.info("_____________________________________________")

def coup_test(game, bots):
  '''
  Test Coup prob for 7-9 coins.
  '''
  logging.info("Coup test")
  state = game.new_initial_state()
  state.apply_action(1) # Ambassador
  state.apply_action(1) # Ambassador
  state.apply_action(3) # Contessa
  state.apply_action(3) # Contessa
  for _ in range(11):
    state.apply_action(0) # Income
  logging.info("  P2: 7 coins")
  log_action_probs(bots[1], state, indent=2)
  state.apply_action(0) # Income
  logging.info("  P1: 7 coins")
  log_action_probs(bots[0], state, indent=2)
  state.apply_action(0) # Income
  logging.info("  P2: 8 coins")
  log_action_probs(bots[1], state, indent=2)
  state.apply_action(0) # Income
  logging.info("  P1: 8 coins")
  log_action_probs(bots[0], state, indent=2)
  state.apply_action(0) # Income
  logging.info("  P2: 9 coins")
  log_action_probs(bots[1], state, indent=2)
  state.apply_action(0) # Income
  logging.info("  P1: 9 coins")
  log_action_probs(bots[0], state, indent=2)
  logging.info("_____________________________________________")

def exchange_test(game, bots):
  '''
  Player sees Duke cards during exchange.
  Next turn, opp tries to tax.
  Should challenge.
  '''
  logging.info("Exchange test")
  for p in range(2):
    logging.info(f"  P{p+1} version")
    bot = bots[p]
    state = game.new_initial_state()
    state.apply_action(1) # Ambassador
    state.apply_action(1) # Ambassador
    state.apply_action(3) # Contessa
    state.apply_action(3) # Contessa
    if p == 0:
      state.apply_action(0) # Income
    state.apply_action(3) # Tax
    logging.info("  Tax challenge? Dukes seen: 0")
    log_action_probs(bot, state, indent=2)
    state.apply_action(9) # Pass
    state.apply_action(5) # Exchange
    state.apply_action(9) # Pass
    state.apply_action(3) # Contessa
    state.apply_action(4) # Duke
    state.apply_action(15) # p keep Amb, Duke
    state.apply_action(3) # Tax
    logging.info("  Tax challenge? Dukes seen: 1")
    log_action_probs(bot, state, indent=2)
    state.apply_action(9) # Pass
    state.apply_action(5) # Exchange
    state.apply_action(9) # Pass
    state.apply_action(4) # Duke
    state.apply_action(4) # Duke
    state.apply_action(12) # p keep Duke, Duke
    state.apply_action(3) # Tax
    logging.info("  Tax challenge? Dukes seen: 3")
    log_action_probs(bot, state, indent=2)
  logging.info("_____________________________________________")

def assassinate_test(game, bots):
  '''
  See prob of assassinate with/without card in hand.
  '''
  logging.info("Assassinate test")
  for p in range(2):
    logging.info(f"  P{p+1} version")
    bot = bots[p]

    logging.info("  With assassin in hand")
    state = game.new_initial_state()
    state.apply_action(0) # Assassin
    state.apply_action(0) # Assassin
    state.apply_action(1) # Ambassador
    state.apply_action(1) # Ambassador
    x = 4 if p == 0 else 3
    for _ in range(x): # get player to 3 coins
      state.apply_action(0) # Income
    log_action_probs(bot, state, indent=2)

    logging.info("  Without assassin in hand")
    state = game.new_initial_state()
    state.apply_action(1) # Ambassador
    state.apply_action(1) # Ambassador
    state.apply_action(2) # Captain
    state.apply_action(2) # Captain
    for _ in range(x): # get player to 3 coins
      state.apply_action(0) # Income
    log_action_probs(bot, state, indent=2)
  logging.info("_____________________________________________")

def counter_assassinate_test(game, bots):
  '''
  See prob of block or challenge an asassination.
  With 2 cards remaining, should not risk double elimination.
  With 1 card remaining, nothing to lose.
  '''
  logging.info("Counter Assassinate test")
  for p in range(2):
    logging.info(f"  P{p+1} version")
    bot = bots[p]
    state = game.new_initial_state()
    state.apply_action(0) # Assassin
    state.apply_action(0) # Assassin
    state.apply_action(1) # Ambassador
    state.apply_action(1) # Ambassador
    x = 3 if p == 0 else 4
    for _ in range(x): # get opp to 3 coins
      state.apply_action(0) # Income
    state.apply_action(4) # Assassinate
    logging.info("With 2 cards left")
    log_action_probs(bot, state, indent=2)
    state.apply_action(8) # Lose card 2
    for _ in range(7): # get opp to 3 coins
      state.apply_action(0) # Income
    state.apply_action(4) # Assassinate
    logging.info("With 1 card left")
    log_action_probs(bot, state, indent=2)
  logging.info("_____________________________________________")

def steal_test(game, bots):
  '''
  Test Steal prob before and after being blocked.
  Should see a decrease in prob.
  '''
  logging.info("Steal test")
  for p in range(2):
    logging.info(f"  P{p+1} version")
    bot = bots[p]
    state = game.new_initial_state()
    state.apply_action(1) # Ambassador
    state.apply_action(1) # Ambassador
    state.apply_action(2) # Captain
    state.apply_action(2) # Captain
    if p == 1:
      state.apply_action(0) # Income
    for i in range(2):
      logging.info(f"  Steal attempt {i+1}")
      log_action_probs(bot, state, indent=2)
      state.apply_action(6) # Steal
      state.apply_action(10) # Block
      state.apply_action(9) # Pass
      state.apply_action(0) # Income
    logging.info("  Steal attempt 3")
    log_action_probs(bot, state, indent=2)
  logging.info("_____________________________________________")

def bluff_seq_test(game, bots):
  '''
  Opponent bluffs for most of game, pretending to have Duke and Captain.
  At the end opponent uses Exchange. 
  Should player have higher prob to challenge?
  '''
  logging.info("Bluff seq test")
  for p in range(2):
    logging.info(f"  P{p+1} version")
    bot = bots[p]
    state = game.new_initial_state()
    state.apply_action(1) # Ambassador
    state.apply_action(1) # Ambassador
    state.apply_action(3) # Contessa
    state.apply_action(3) # Contessa
    if p == 0:
      state.apply_action(0) # Income
    for i in range(2):
      state.apply_action(3) # Tax
      logging.info(f"  Counter probs to Tax {i+1}")
      log_action_probs(bot, state, indent=2)
      state.apply_action(9) # Pass
      state.apply_action(0) # Income
      state.apply_action(3) # Steal
      logging.info(f"  Counter probs to Steal {i+1}")
      log_action_probs(bot, state, indent=2)
      state.apply_action(10) # Block
      state.apply_action(9) # Pass
      state.apply_action(0) # Income
    logging.info("  Sudden switch to exchange. Challenge?")
    state.apply_action(5) # Exchange
    log_action_probs(bot, state, indent=2)
  logging.info("_____________________________________________")

def random_seq_test(game, bots):
  '''
  Test a random action sequence for opponent.
  Player should be more likely to challenge.
  '''
  logging.info("Random seq test")
  for p in range(2):
    logging.info(f"  P{p+1} version")
    state = game.new_initial_state()
    state.apply_action(1) # Ambassador
    state.apply_action(1) # Ambassador
    state.apply_action(3) # Contessa
    state.apply_action(3) # Contessa
    it = 0
    while not state.is_terminal() and it < 10:
      if state.is_chance_node():
        # Sample chance
        outcomes_with_probs = state.chance_outcomes()
        action_list, prob_list = zip(*outcomes_with_probs)
        action = np.random.choice(action_list, p=prob_list)
        state.apply_action(action)
      elif state.current_player() == 1-p: # opp
        bot = bots[1-p]
        action = np.random.choice(state.legal_actions())
        state.apply_action(action)
      else: # player p
        legal = state.legal_actions()
        bot = bots[p]
        if 11 in legal: # Challenge
          it += 1
          log_action_probs(bot, state, indent=2)
          state.apply_action(min(legal)) # Don't challenge
        else:
          action = bot.step(state)
          state.apply_action(action)
  logging.info("_____________________________________________")

def policy_tests(game, bots):
  foreign_aid_test(game, bots)
  coup_test(game, bots)
  exchange_test(game, bots)
  assassinate_test(game, bots)
  counter_assassinate_test(game, bots)
  steal_test(game, bots)
  bluff_seq_test(game, bots)
  random_seq_test(game, bots)

def main(_):
  game = pyspiel.load_game(FLAGS.game)
  with tf.Session() as sess:
    if FLAGS.algo == "deep_cfr":
      bots = [get_deep_cfr_bot(pid, sess, game, tf, FLAGS, FLAGS.saved_dir, FLAGS.checkpoint_id) for pid in range(2)]
    elif FLAGS.algo == "nfsp":
      bots = [get_nfsp_bot(pid, sess, FLAGS, FLAGS.saved_dir, FLAGS.checkpoint_id) for pid in range(2)]
    else:
      logging.info(f"Algorithm {FLAGS.algo} not recognized")
      return
    
    policy_tests(game, bots)

if __name__ == "__main__":
  app.run(main)