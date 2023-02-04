"""Run a Human v. Bot game of Coup"""

from absl import app
from absl import flags
from absl import logging

import tensorflow.compat.v1 as tf

from open_spiel.python import policy as pol
from open_spiel.python.bots.policy import PolicyBot
from open_spiel.python.bots.human import HumanBot
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import deep_cfr, nfsp
import pyspiel
import np

# Temporarily disable TF2 behavior until we update the code.
tf.disable_v2_behavior()

FLAGS = flags.FLAGS

flags.DEFINE_string("game", "coup", "Game name")
flags.DEFINE_string("algo", "deep_cfr", "Algorithm which has a saved model in the given directory. deep_cfr or nfsp")
flags.DEFINE_string("saved_dir", "/agents/deep_cfr/", "Directory with saved model")

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


class NFSPPolicies(pol.Policy):
  """Joint policy to be evaluated."""

  def __init__(self, env, nfsp_policies, mode):
    game = env.game
    player_ids = list(range(FLAGS.num_players))
    super(NFSPPolicies, self).__init__(game, player_ids)
    self._policies = nfsp_policies
    self._mode = mode
    self._obs = {
        "info_state": [None] * FLAGS.num_players,
        "legal_actions": [None] * FLAGS.num_players
    }

  def action_probabilities(self, state, player_id=None):
    cur_player = state.current_player()
    legal_actions = state.legal_actions(cur_player)

    self._obs["current_player"] = cur_player
    self._obs["info_state"][cur_player] = (
        state.information_state_tensor(cur_player))
    self._obs["legal_actions"][cur_player] = legal_actions

    info_state = rl_environment.TimeStep(
        observations=self._obs, rewards=None, discounts=None, step_type=None)

    with self._policies[cur_player].temp_mode_as(self._mode):
      p = self._policies[cur_player].step(info_state, is_evaluation=True).probs
    prob_dict = {action: p[action] for action in legal_actions}
    return prob_dict

def main():
  game = pyspiel.load_game(FLAGS.game)
  with tf.Session() as sess:
    if FLAGS.algo == "deep_cfr":
      policy_network_layers = [int(l) for l in FLAGS.policy_network_layers]
      advantage_network_layers = [int(l) for l in FLAGS.advantage_network_layers]
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
          sampling_method="outcome")
      sess.run(tf.global_variables_initializer())
      deep_cfr_solver.restore_policy_network(FLAGS.saved_dir)
      bot = PolicyBot(1, np.random.RandomState(4321), deep_cfr_solver)

    elif FLAGS.algo == "nfsp":
      env = rl_environment.Environment(FLAGS.game)
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
      agents = [
        nfsp.NFSP(sess, idx, info_state_size, num_actions, hidden_layers_sizes,
                  **kwargs) for idx in range(2)
      ]
      joint_avg_policy = NFSPPolicies(env, agents, nfsp.MODE.average_policy)
      for agent in agents:
        agent.restore(FLAGS.saved_dir)

      bot = PolicyBot(1, np.random.RandomState(4321), joint_avg_policy)

    else:
      logging.info(f"Algorithm {FLAGS.algo} not recognized")
      return
    
    human = HumanBot()
    players = [human, bot]

    state = game.new_initial_state()
    print(state.observation_string())
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
      
      print(state.observation_string())


if __name__ == "__main__":
  app.run(main)