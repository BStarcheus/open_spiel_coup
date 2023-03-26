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

"""NFSP agents trained on Coup."""

from absl import app
from absl import flags
from absl import logging
import tensorflow.compat.v1 as tf

from open_spiel.python import rl_environment
from open_spiel.python.algorithms import nfsp

from coup_experiments.algorithms.rl_response import rl_resp
from coup_experiments.utils.logging import *
from coup_experiments.utils.nfsp_policies import *
import time

FLAGS = flags.FLAGS

flags.DEFINE_string("game_name", "coup",
                    "Name of the game.")
flags.DEFINE_integer("num_train_episodes", 1000000,
                     "Number of training episodes.")
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
flags.DEFINE_bool("use_checkpoints", True, "Save/load neural network weights.")
flags.DEFINE_string("checkpoint_dir", "/tmp/nfsp",
                    "Directory to save/load the agent.")
flags.DEFINE_integer("save_every", 10000,
                     "How often to save the networks. Must be multiple of eval_every.")

flags.DEFINE_integer("eval_every", 10000,
                     "Episode frequency at which the agents are evaluated.")
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
  log_flags(FLAGS, ["num_train_episodes", "hidden_layers_sizes",
      "replay_buffer_capacity", "reservoir_buffer_capacity", 
      "min_buffer_size_to_learn", "anticipatory_param", "batch_size",
      "learn_every", "rl_learning_rate", "sl_learning_rate",
      "update_target_network_every", "epsilon_decay_duration", "epsilon_start",
      "epsilon_end", "eval_every", "rl_resp_train_episodes",
      "rl_resp_eval_every", "rl_resp_eval_episodes"])
  logging.info("Loading %s", FLAGS.game_name)
  game = FLAGS.game_name
  num_players = 2

  env = rl_environment.Environment(game)
  info_state_size = env.observation_spec()["info_state"][0]
  num_actions = env.action_spec()["num_actions"]

  hidden_layers_sizes = [int(l) for l in FLAGS.hidden_layers_sizes]
  kwargs = {
      "replay_buffer_capacity": FLAGS.replay_buffer_capacity,
      "reservoir_buffer_capacity": FLAGS.reservoir_buffer_capacity,
      "min_buffer_size_to_learn": FLAGS.min_buffer_size_to_learn,
      "anticipatory_param": FLAGS.anticipatory_param,
      "batch_size": FLAGS.batch_size,
      "learn_every": FLAGS.learn_every,
      "rl_learning_rate": FLAGS.rl_learning_rate,
      "sl_learning_rate": FLAGS.sl_learning_rate,
      "optimizer_str": FLAGS.optimizer_str,
      "loss_str": FLAGS.loss_str,
      "update_target_network_every": FLAGS.update_target_network_every,
      "discount_factor": FLAGS.discount_factor,
      "epsilon_decay_duration": FLAGS.epsilon_decay_duration,
      "epsilon_start": FLAGS.epsilon_start,
      "epsilon_end": FLAGS.epsilon_end,
  }

  with tf.Session() as sess:
    # pylint: disable=g-complex-comprehension
    agents = [
        nfsp.NFSP(sess, idx, info_state_size, num_actions, hidden_layers_sizes,
                  **kwargs) for idx in range(num_players)
    ]
    joint_avg_policy = NFSPPolicies(env, agents, nfsp.MODE.average_policy)

    sess.run(tf.global_variables_initializer())

    total_rl_resp_time = 0
    first_start = time.time()
    for ep in range(FLAGS.num_train_episodes):
      time_step = env.reset()
      while not time_step.last():
        player_id = time_step.observations["current_player"]
        agent_output = agents[player_id].step(time_step)
        action_list = [agent_output.action]
        time_step = env.step(action_list)

      # Episode is over, step all agents with final info state.
      for agent in agents:
        agent.step(time_step)

      # Evaluation
      if (ep + 1) % FLAGS.eval_every == 0:
        losses = [agent.loss for agent in agents]
        logging.info("Losses: %s", losses)

        start = time.time()
        rl_resp(exploitee=joint_avg_policy,
                num_train_episodes=FLAGS.rl_resp_train_episodes,
                eval_every=FLAGS.rl_resp_eval_every,
                eval_episodes=FLAGS.rl_resp_eval_episodes)
        delta = time.time() - start
        total_rl_resp_time += delta
        logging.info("rl_resp run time: %s sec", delta)

        if FLAGS.use_checkpoints and (ep + 1) % FLAGS.save_every == 0:
          for agent in agents:
            agent.save(FLAGS.checkpoint_dir, f"ep{ep+1}")
        logging.info("_____________________________________________")
    
    total_time = time.time() - first_start
    logging.info("Total algo run time: %s sec", total_time - total_rl_resp_time)
    logging.info("Total rl_resp run time: %s sec", total_rl_resp_time)
    logging.info("Total run time: %s sec", total_time)


if __name__ == "__main__":
  app.run(main)
