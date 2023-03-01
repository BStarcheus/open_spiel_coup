"""Run a Human v. Bot game of Coup"""

from absl import app
from absl import flags
from absl import logging

import tensorflow.compat.v1 as tf

from open_spiel.python.bots.human import HumanBot
from open_spiel.python.bots.uniform_random import UniformRandomBot
import pyspiel
import numpy as np

from coup_experiments.utils.get_bots import *

# Temporarily disable TF2 behavior until we update the code.
tf.disable_v2_behavior()

FLAGS = flags.FLAGS

flags.DEFINE_string("game", "coup", "Game name")
flags.DEFINE_string("algo", "deep_cfr", "Algorithm which has a saved model in the given directory. deep_cfr, nfsp, random")
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

def main(_):
  game = pyspiel.load_game(FLAGS.game)
  with tf.Session() as sess:
    if FLAGS.algo == "deep_cfr":
      bot = get_deep_cfr_bot(1, sess, game, tf, FLAGS, FLAGS.saved_dir, FLAGS.checkpoint_id)
    elif FLAGS.algo == "nfsp":
      bot = get_nfsp_bot(1, sess, FLAGS, FLAGS.saved_dir, FLAGS.checkpoint_id)
    elif FLAGS.algo == "random":
      bot = UniformRandomBot(1, np.random.RandomState(4321))
    else:
      logging.info(f"Algorithm {FLAGS.algo} not recognized")
      return
    
    human = HumanBot()
    players = [human, bot]

    state = game.new_initial_state()
    print(state.observation_string(0))
    while not state.is_terminal():
      if state.is_chance_node():
        # Sample chance
        outcomes_with_probs = state.chance_outcomes()
        action_list, prob_list = zip(*outcomes_with_probs)
        action = np.random.choice(action_list, p=prob_list)
        state.apply_action(action)
      else:
        pid = state.current_player()
        p = players[pid]
        action = p.step(state)
        state.apply_action(action)
      
      print(state.observation_string(0))

    print("Opponent's view:")
    print(state.observation_string(1))

if __name__ == "__main__":
  app.run(main)