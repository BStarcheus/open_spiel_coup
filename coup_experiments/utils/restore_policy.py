from open_spiel.python.algorithms import deep_cfr, nfsp
from open_spiel.python import rl_environment
from coup_experiments.utils.nfsp_policies import *

def restore_deep_cfr(sess, game, tf, flags, saved_dir, checkpoint_id):
  policy_network_layers = [int(l) for l in flags.policy_network_layers]
  advantage_network_layers = [int(l) for l in flags.advantage_network_layers]
  deep_cfr_solver = deep_cfr.DeepCFRSolver(
      sess,
      game,
      policy_network_layers=policy_network_layers,
      advantage_network_layers=advantage_network_layers,
      num_iterations=flags.num_iterations,
      num_traversals=flags.num_traversals,
      learning_rate=flags.learning_rate,
      batch_size_advantage=flags.batch_size_advantage,
      batch_size_strategy=flags.batch_size_strategy,
      memory_capacity=flags.memory_capacity,
      policy_network_train_steps=flags.policy_network_train_steps,
      advantage_network_train_steps=flags.advantage_network_train_steps,
      reinitialize_advantage_networks=flags.reinitialize_advantage_networks,
      sampling_method="outcome")
  sess.run(tf.global_variables_initializer())
  deep_cfr_solver.restore_policy_network(saved_dir, checkpoint_id)
  return deep_cfr_solver

def restore_nfsp(sess, flags, saved_dir, checkpoint_id):
  env = rl_environment.Environment(flags.game)
  info_state_size = env.observation_spec()["info_state"][0]
  num_actions = env.action_spec()["num_actions"]

  hidden_layers_sizes = [int(l) for l in flags.hidden_layers_sizes]
  kwargs = {
      "replay_buffer_capacity": flags.replay_buffer_capacity,
      "reservoir_buffer_capacity": flags.reservoir_buffer_capacity,
      "min_buffer_size_to_learn": flags.min_buffer_size_to_learn,
      "anticipatory_param": flags.anticipatory_param,
      "batch_size": flags.batch_size,
      "learn_every": flags.learn_every,
      "rl_learning_rate": flags.rl_learning_rate,
      "sl_learning_rate": flags.sl_learning_rate,
      "optimizer_str": flags.optimizer_str,
      "loss_str": flags.loss_str,
      "update_target_network_every": flags.update_target_network_every,
      "discount_factor": flags.discount_factor,
      "epsilon_decay_duration": flags.epsilon_decay_duration,
      "epsilon_start": flags.epsilon_start,
      "epsilon_end": flags.epsilon_end,
  }
  agents = [
    nfsp.NFSP(sess, idx, info_state_size, num_actions, hidden_layers_sizes,
              **kwargs) for idx in range(2)
  ]
  joint_avg_policy = NFSPPolicies(env, agents, nfsp.MODE.average_policy)
  for agent in agents:
    agent.restore(saved_dir, checkpoint_id)
  return joint_avg_policy