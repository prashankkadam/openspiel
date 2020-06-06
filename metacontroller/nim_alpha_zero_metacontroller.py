"""
Created on Fri 06/05/2020 12:07:20 2020
This piece of software is bound by The Apache License
Copyright (c) 2019 Prashank Kadam
Code written by : Prashank Kadam
User name - prashank
Email ID : kadam.pr@husky.neu.edu
version : 1.0
"""
# Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Simple AlphaZero tic tac toe example.

Take a look at the log-learner.txt in the output directory.

If you want more control, check out `alpha_zero.py`.

This version of Alpha-Zero uses a non-stationary multiarm bandit algorithm called
Sliding window UCB along with the greedy epsilon algorithm, this together is 
known as the Metacontroller.
"""

from absl import app
from absl import flags

from open_spiel.python.algorithms.metacontroller import alpha_zero_metacontroller as alpha_zero
from open_spiel.python.utils import spawn

import pudb

flags.DEFINE_string("path", None, "Where to save checkpoints.")
FLAGS = flags.FLAGS


def main(unused_argv):
  config = alpha_zero.Config(
      game="nim",
      path=FLAGS.path,
      learning_rate=0.01,
      weight_decay=1e-4,
      train_batch_size=128,
      replay_buffer_size=2 ** 14,
      replay_buffer_reuse=4,
      max_steps=5,
      checkpoint_freq=5,

      actors=2,
      evaluators=2,
      uct_c=1,
      greedy_e=0.7,
      tau_c=50,
      max_simulations=10,
      policy_alpha=0.25,
      policy_epsilon=1,
      temperature=1,
      temperature_drop=4,
      evaluation_window=50,
      eval_levels=7,

      nn_model="resnet",
      nn_width=32,
      nn_depth=2,
      observation_shape=None,
      output_size=None,

      quiet=True,
  )

  # pudb.set_trace()

  alpha_zero.alpha_zero(config)


if __name__ == "__main__":
  with spawn.main_handler():
    app.run(main)
