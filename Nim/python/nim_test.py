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

# Lint as python3
"""Tests for Python Nim."""

from absl.testing import absltest
import numpy as np
from open_spiel.python.games import nim

import random

# import pudb

_LEGAL_PICK = 5

class NimTest(absltest.TestCase):

  def test_create(self):
    game = nim.NimGame()
    print(game.num_distinct_actions())
    clone = game.clone()
    print(clone.num_distinct_actions())
    state = game.new_initial_state()
    clone = state.clone()
    print(state)
    print(clone)

  def test_random_game(self):
    # This is here mostly to show the API by example.
    # More serious simulation tests are done in python/tests/game_sim_test.py.
    # Those test the conformance to the API thoroughly.
    game = nim.NimGame()
    state = game.new_initial_state()
    while not state.is_terminal():

      # pudb.set_trace()

      print(state)
      cur_player = state.current_player()
      legal_actions = state.legal_actions()
      # action = np.random.choice(legal_actions)   # Change the logic

      if len(legal_actions) > _LEGAL_PICK:
        action = np.random.randint(low=legal_actions[0], high=legal_actions[_LEGAL_PICK-1])
      else:
        # action = np.random.randint(low=legal_actions[0], high=legal_actions[-1])
        action = random.choice(legal_actions)

      print("Player {} chooses action {}".format(cur_player, action))
      state.apply_action(action)
    print(state)
    print("Returns: {}".format(state.returns()))


if __name__ == "__main__":
  absltest.main()
