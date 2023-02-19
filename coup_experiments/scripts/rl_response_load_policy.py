"""Load a saved policy agent and run rl_response"""

from absl import app
from absl import flags
from absl import logging
import tensorflow.compat.v1 as tf
import pyspiel
from coup_experiments.utils.restore_policy import *
from coup_experiments.algorithms.rl_response import rl_resp
from coup_experiments.utils.logging import *

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

flags.DEFINE_integer("rl_resp_train_episodes", 10000,
                     "Number of training episodes for rl_resp model")
flags.DEFINE_integer("rl_resp_eval_every", 1000,
                     "How often to evaluate trained rl_resp model")
flags.DEFINE_integer("rl_resp_eval_episodes", 1000,
                     "Number of episodes per rl_resp evaluation")
flags.DEFINE_string("log_file", "", "File to output log to")

def main(_):
  if len(FLAGS.log_file):
    log_to_file(FLAGS.log_file)
  log_flags(FLAGS, ["saved_dir", "checkpoint_id"])
  game = pyspiel.load_game(FLAGS.game)
  with tf.Session() as sess:
    if FLAGS.algo == "deep_cfr":
      pol = restore_deep_cfr(sess, game, tf, FLAGS, FLAGS.saved_dir, FLAGS.checkpoint_id)
    elif FLAGS.algo == "nfsp":
      pol = restore_nfsp(sess, FLAGS, FLAGS.saved_dir, FLAGS.checkpoint_id)
    else:
      logging.info(f"Algorithm {FLAGS.algo} not recognized")
      return
    
    rl_resp(exploitee=pol,
            num_train_episodes=FLAGS.rl_resp_train_episodes,
            eval_every=FLAGS.rl_resp_eval_every,
            eval_episodes=FLAGS.rl_resp_eval_episodes)


if __name__ == "__main__":
  app.run(main)