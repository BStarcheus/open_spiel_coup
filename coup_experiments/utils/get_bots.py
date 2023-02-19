from open_spiel.python.bots.policy import PolicyBot
from coup_experiments.utils.restore_policy import *
import numpy as np

def get_deep_cfr_bot(player_ind, sess, game, tf, flags, saved_dir, checkpoint_id):
  deep_cfr_solver = restore_deep_cfr(sess, game, tf, flags, saved_dir, checkpoint_id)
  return PolicyBot(player_ind, np.random.RandomState(4321), deep_cfr_solver)

def get_nfsp_bot(player_ind, sess, flags, saved_dir, checkpoint_id):
  joint_avg_policy = restore_nfsp(sess, flags, saved_dir, checkpoint_id)
  return PolicyBot(player_ind, np.random.RandomState(4321), joint_avg_policy)