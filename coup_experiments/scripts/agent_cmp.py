"""Test a list of saved agents for two algorithms against each other"""

from absl import app
from absl import flags
from absl import logging

import tensorflow.compat.v1 as tf

import pyspiel
import numpy as np
from coup_experiments.utils.logging import *
from coup_experiments.utils.get_bots import *

# Temporarily disable TF2 behavior until we update the code.
tf.disable_v2_behavior()

FLAGS = flags.FLAGS

flags.DEFINE_string("game", "coup", "Game name")
flags.DEFINE_string("algo1", "deep_cfr", "Algorithm  of the first list of saved models. deep_cfr or nfsp")
flags.DEFINE_string("algo2", "nfsp", "Algorithm of the second list of saved models. deep_cfr or nfsp")
flags.DEFINE_string("saved_dir1", "/agents/deep_cfr/", "Directory with saved model")
flags.DEFINE_string("saved_dir2", "/agents/nfsp/", "Directory with saved model")
flags.DEFINE_list("algo1_checkpoint_ids", [f"iter{10 * (i+1)}" for i in range(5)],
                  "List of ID's of the checkpoint. deep_cfr: iter{iter-num}, nfsp: ep{ep-num}")
flags.DEFINE_list("algo2_checkpoint_ids", [f"ep{1000000 * (i+1)}" for i in range(5)],
                  "List of ID's of the checkpoint. deep_cfr: iter{iter-num}, nfsp: ep{ep-num}")
flags.DEFINE_integer("cmp_test_eps", 1000, "Number of episodes to test per agent matchup")
flags.DEFINE_string("log_file", "", "File to output log to")

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
  if len(FLAGS.log_file):
    log_to_file(FLAGS.log_file)
  log_flags(FLAGS, ["cmp_test_eps"])

  game = pyspiel.load_game(FLAGS.game)
  algo1_checkpoint_ids = FLAGS.algo1_checkpoint_ids
  algo2_checkpoint_ids = FLAGS.algo2_checkpoint_ids

  with tf.Session() as sess:
    players = [None]*2

    for a1_c_id in algo1_checkpoint_ids:
      if FLAGS.algo1 == "deep_cfr":
        players[0] = get_deep_cfr_bot(0, sess, game, tf, FLAGS, FLAGS.saved_dir1, a1_c_id)
      elif FLAGS.algo1 == "nfsp":
        players[0] = get_nfsp_bot(0, sess, FLAGS, FLAGS.saved_dir1, a1_c_id)
      else:
        logging.info(f"Algorithm {FLAGS.algo1} not recognized")
        return

      for a2_c_id in algo2_checkpoint_ids:
        if FLAGS.algo2 == "deep_cfr":
          players[1] = get_deep_cfr_bot(1, sess, game, tf, FLAGS, FLAGS.saved_dir2, a2_c_id)
        elif FLAGS.algo2 == "nfsp":
          players[1] = get_nfsp_bot(1, sess, FLAGS, FLAGS.saved_dir2, a2_c_id)
        else:
          logging.info(f"Algorithm {FLAGS.algo2} not recognized")
          return

        logging.info(f"{FLAGS.saved_dir1} {a1_c_id}  vs.  {FLAGS.saved_dir2} {a2_c_id}")

        # Test agents against each other
        p0_total_reward = 0
        total_ep_steps = 0
        for _ in range(FLAGS.cmp_test_eps):

          state = game.new_initial_state()
          while not state.is_terminal():
            total_ep_steps += 1
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

          p0_total_reward += state.returns()[0]

        p0_mean_reward = p0_total_reward / FLAGS.cmp_test_eps
        # p1 mean reward is * -1 (zero sum game)
        logging.info(f"  Mean rewards: [{p0_mean_reward}, {-1*p0_mean_reward}]")
        logging.info(f"  Mean ep length: {total_ep_steps / FLAGS.cmp_test_eps}")


if __name__ == "__main__":
  app.run(main)