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

"""MCCFR algorithm on Coup."""

from absl import app
from absl import flags
from absl import logging

from open_spiel.python.algorithms import outcome_sampling_mccfr as outcome_mccfr
import pyspiel

from coup_experiments.algorithms.rl_response import rl_resp
from utils import *
import time

FLAGS = flags.FLAGS

flags.DEFINE_integer("iterations", 1000000, "Number of iterations")
flags.DEFINE_string("game", "coup", "Name of the game")
flags.DEFINE_integer("eval_every", 100000,
                     "How often to evaluate model")

flags.DEFINE_integer("rl_resp_train_episodes", 10000,
                     "Number of training episodes for rl_resp model")
flags.DEFINE_integer("rl_resp_eval_every", 1000,
                     "How often to evaluate trained rl_resp model")
flags.DEFINE_integer("rl_resp_eval_episodes", 1000,
                     "Number of episodes per rl_resp evaluation")
flags.DEFINE_string("log_file", "", "File to output log to")

def main(_):
  log_to_file(FLAGS.log_file)
  log_flags(FLAGS, ["iterations", "eval_every", "rl_resp_train_episodes",
                    "rl_resp_eval_every", "rl_resp_eval_episodes"])
  logging.info("Loading %s", FLAGS.game_name)
  game = pyspiel.load_game(FLAGS.game)
  cfr_solver = outcome_mccfr.OutcomeSamplingSolver(game)

  total_rl_resp_time = 0
  first_start = time.time()
  for i in range(FLAGS.iterations):
    cfr_solver.iteration()
    if i % FLAGS.eval_every == 0:
      start = time.time()
      rl_resp(exploitee=cfr_solver.average_policy(),
              num_train_episodes=FLAGS.rl_resp_train_episodes,
              eval_every=FLAGS.rl_resp_eval_every,
              eval_episodes=FLAGS.rl_resp_eval_episodes)
      delta = time.time() - start
      total_rl_resp_time += delta
      logging.info("rl_resp run time: %s sec", delta)
      logging.info("_____________________________________________")
  
  total_time = time.time() - first_start
  logging.info("Total algo run time: %s sec", total_time - total_rl_resp_time)
  logging.info("Total rl_resp run time: %s sec", total_rl_resp_time)
  logging.info("Total run time: %s sec", total_time)


if __name__ == "__main__":
  app.run(main)
