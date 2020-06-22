// Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "open_spiel/games/hsr_temp.h"

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/tensor_view.h"

namespace open_spiel {
namespace hsr_temp {
namespace {

// Facts about the game.
const GameType kGameType{
    /*short_name=*/"hsr_temp",
    /*long_name=*/"HSR Temp",
    GameType::Dynamics::kSequential,
    GameType::ChanceMode::kDeterministic,
    GameType::Information::kPerfectInformation,
    GameType::Utility::kZeroSum,
    GameType::RewardModel::kTerminal,
    /*max_num_players=*/2,
    /*min_num_players=*/2,
    /*provides_information_state_string=*/true,
    /*provides_information_state_tensor=*/false,
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/true,
    /*parameter_specification=*/{}  // no parameters
};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new HSRTempGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

}  // namespace

CellState PlayerToState(Player player) {
  switch (player) {
    case 0:
      return CellState::kCross;
    case 1:
      return CellState::kNought;
    default:
      SpielFatalError(absl::StrCat("Invalid player id ", player));
      return CellState::kEmpty;
  }
}

std::string StateToString(CellState state) {
  switch (state) {
    case CellState::kEmpty:
      return ".";
    case CellState::kNought:
      return "o";
    case CellState::kCross:
      return "x";
    default:
      SpielFatalError("Unknown state.");
  }
}

void HSRTempState::DoApplyAction(Action move) {
  if (current_player_ == 0) {
    SPIEL_CHECK_EQ(board_[move], CellState::kEmpty);
    board_[move] = PlayerToState(CurrentPlayer());
    if (board_[move - 1] == CellState::kCross && board_[move + 1] == CellState::kCross) {
       outcome_ = Player{0};
       return;
    } else if (board_[move - 1] == CellState::kCross && move == kNumCells - 1) {
       outcome_ = Player{0};
       return;
    } else if (move == 0 && board_[move + 1] == CellState::kCross) {
       outcome_ = Player{0};
       return;
    }
    previous_move_ = move;
    num_moves_ += 1;
  } else if (current_player_ == 1) {
    current_part_ = move;
  }
  current_player_ = 1 - current_player_;
}

std::vector<Action> HSRTempState::LegalActions() const {
  if (IsTerminal()) return {};
  // Can move in any empty cell.
  std::vector<Action> moves;

  if (current_player_ == 0) {
    for (int cell = 0; cell < 4; ++cell) {
      if (board_[cell] == CellState::kEmpty) {
        moves.push_back(cell);
      }
    }
  } else if (current_player_ == 1) {
      for (int cell = 4; cell < kNumCells; ++cell) {
        if (board_[cell] == CellState::kEmpty) {
          moves.push_back(cell);
        }
      }
      moves.push_back(-1);
  }
  return moves;
}

std::string HSRTempState::ActionToString(Player player,
                                           Action action_id) const {
  return absl::StrCat(StateToString(PlayerToState(player)), "(",
                      action_id / kNumCols, ",", action_id % kNumCols, ")");
}

bool HSRTempState::HasLine(Player player) const {
  CellState c = PlayerToState(player);
  return (board_[0] == c && board_[1] == c && board_[2] == c) ||
         (board_[3] == c && board_[4] == c && board_[5] == c) ||
         (board_[6] == c && board_[7] == c && board_[8] == c) ;
//         (board_[0] == c && board_[3] == c && board_[6] == c) ||
//         (board_[1] == c && board_[4] == c && board_[7] == c) ||
//         (board_[2] == c && board_[5] == c && board_[8] == c) ||
//         (board_[0] == c && board_[4] == c && board_[8] == c) ||
//         (board_[2] == c && board_[4] == c && board_[6] == c);
}

bool HSRTempState::IsFull() const { return num_moves_ == kNumCells; }

HSRTempState::HSRTempState(std::shared_ptr<const Game> game) : State(game) {
  std::fill(begin(board_), end(board_), CellState::kEmpty);
}

std::string HSRTempState::ToString() const {
  std::string str;
  for (int r = 0; r < kNumRows; ++r) {
    for (int c = 0; c < kNumCols; ++c) {
      absl::StrAppend(&str, StateToString(BoardAt(r, c)));
    }
    if (r < (kNumRows - 1)) {
      absl::StrAppend(&str, "\n");
    }
  }
  return str;
}

bool HSRTempState::IsTerminal() const {
  return outcome_ != kInvalidPlayer || IsFull();
}

std::vector<double> HSRTempState::Returns() const {
  if (outcome_ == Player{0}) {
    return {1.0, -1.0};
  } else if (outcome_ == Player{1}) {
    return {-1.0, 1.0};
  } else {
    return {0.0, 0.0};
  }
}

std::string HSRTempState::InformationStateString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return HistoryString();
}

std::string HSRTempState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return ToString();
}

void HSRTempState::ObservationTensor(Player player,
                                       std::vector<double>* values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  // Treat `values` as a 2-d tensor.
  TensorView<2> view(values, {kCellStates, kNumCells}, true);
  for (int cell = 0; cell < kNumCells; ++cell) {
    view[{static_cast<int>(board_[cell]), cell}] = 1.0;
  }
}

void HSRTempState::UndoAction(Player player, Action move) {
  board_[move] = CellState::kEmpty;
  current_player_ = player;
  outcome_ = kInvalidPlayer;
  num_moves_ -= 1;
  history_.pop_back();
}

std::unique_ptr<State> HSRTempState::Clone() const {
  return std::unique_ptr<State>(new HSRTempState(*this));
}

HSRTempGame::HSRTempGame(const GameParameters& params)
    : Game(kGameType, params) {}

}  // namespace hsr_temp
}  // namespace open_spiel
