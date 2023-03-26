# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Python Deep CFR trained on Coup."""

from absl import app
from absl import flags
from absl import logging

import tensorflow.compat.v1 as tf

from open_spiel.python.algorithms import deep_cfr
import pyspiel
from coup_experiments.algorithms.rl_response import rl_resp
from coup_experiments.utils.logging import *
import time

# Temporarily disable TF2 behavior until we update the code.
tf.disable_v2_behavior()

FLAGS = flags.FLAGS

flags.DEFINE_integer("num_iterations", 100, "Number of iterations")
flags.DEFINE_integer("num_traversals", 100, "Number of traversals/games")
flags.DEFINE_string("game_name", "coup", "Name of the game")
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
flags.DEFINE_string("sampling_method", "outcome",
                    "outcome or e-outcome sampling")
flags.DEFINE_integer("outcome_factor", 1,
                     "Number of actions to sample in outcome sampling.")
flags.DEFINE_float("e_outcome", 0,
                   "Epsilon value for epsilon-outcome sampling. When random < epsilon, use external sampling.")
flags.DEFINE_bool("iter_net_train", False, "Train networks iteratively.")
flags.DEFINE_integer("adv_net_reinit_every", 1,
                     "If reinit=true, reinit adv net every x iterations")
flags.DEFINE_bool("use_checkpoints", True, "Save/load neural network weights.")
flags.DEFINE_string("checkpoint_dir", "/tmp/deep_cfr",
                    "Directory to save/load the agent.")
flags.DEFINE_integer("save_every", 10,
                     "Iteration frequency to save the networks. Must be multiple of eval_every.")

flags.DEFINE_integer("eval_every", 10,
                     "Iteration frequency at which the agents are evaluated.")
flags.DEFINE_integer("rl_resp_train_episodes", 10000,
                     "Number of training episodes for rl_resp model")
flags.DEFINE_integer("rl_resp_eval_every", 1000,
                     "How often to evaluate trained rl_resp model")
flags.DEFINE_integer("rl_resp_eval_episodes", 1000,
                     "Number of episodes per rl_resp evaluation")
flags.DEFINE_string("log_file", "", "File to output log to")


def main(unused_argv):
  if len(FLAGS.log_file):
    log_to_file(FLAGS.log_file)
  log_flags(FLAGS, ["num_iterations", "num_traversals", "policy_network_layers",
      "advantage_network_layers", "learning_rate", "batch_size_advantage",
      "batch_size_strategy", "memory_capacity", "policy_network_train_steps",
      "advantage_network_train_steps", "reinitialize_advantage_networks",
      "sampling_method", "outcome_factor", "e_outcome", "iter_net_train", "adv_net_reinit_every",
      "eval_every", "rl_resp_train_episodes", "rl_resp_eval_every", "rl_resp_eval_episodes"])
  logging.info("Loading %s", FLAGS.game_name)
  game = pyspiel.load_game(FLAGS.game_name)
  policy_network_layers = [int(l) for l in FLAGS.policy_network_layers]
  advantage_network_layers = [int(l) for l in FLAGS.advantage_network_layers]
  with tf.Session() as sess:
    deep_cfr_solver = deep_cfr.DeepCFRSolver(
        sess,
        game,
        policy_network_layers=policy_network_layers,
        advantage_network_layers=advantage_network_layers,
        num_iterations=FLAGS.num_iterations,
        num_traversals=FLAGS.num_traversals,
        learning_rate=FLAGS.learning_rate,
        batch_size_advantage=FLAGS.batch_size_advantage,
        batch_size_strategy=FLAGS.batch_size_strategy,
        memory_capacity=FLAGS.memory_capacity,
        policy_network_train_steps=FLAGS.policy_network_train_steps,
        advantage_network_train_steps=FLAGS.advantage_network_train_steps,
        reinitialize_advantage_networks=FLAGS.reinitialize_advantage_networks,
        sampling_method=FLAGS.sampling_method,
        outcome_factor=FLAGS.outcome_factor,
        e_outcome=FLAGS.e_outcome,
        iter_net_train=FLAGS.iter_net_train,
        adv_net_reinit_every=FLAGS.adv_net_reinit_every,
        eval_func=rl_resp,
        eval_every=FLAGS.eval_every,
        eval_train_episodes=FLAGS.rl_resp_train_episodes,
        eval_test_every=FLAGS.rl_resp_eval_every,
        eval_test_episodes=FLAGS.rl_resp_eval_episodes,
        use_checkpoints=FLAGS.use_checkpoints,
        checkpoint_dir=FLAGS.checkpoint_dir,
        save_every=FLAGS.save_every)
    sess.run(tf.global_variables_initializer())
    start = time.time()
    first_start = start
    _, advantage_losses, policy_loss = deep_cfr_solver.solve()
    delta = time.time() - start
    for player, losses in advantage_losses.items():
      logging.info("Advantage for player %d: %s", player,
                   losses[:2] + ["..."] + losses[-2:])
      logging.info("Advantage Buffer Size for player %s: '%s'", player,
                   len(deep_cfr_solver.advantage_buffers[player]))
    logging.info("Strategy Buffer Size: '%s'",
                 len(deep_cfr_solver.strategy_buffer))
    logging.info("Final policy loss: '%s'", policy_loss)
    logging.info("Algo run time: %s sec", delta)

    final_end = time.time()
    total_time = final_end - first_start
    logging.info("Total run time: %s sec", total_time)

if __name__ == "__main__":
  app.run(main)
