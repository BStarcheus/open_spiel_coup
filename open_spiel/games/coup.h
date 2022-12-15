// Copyright 2019 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Coup
// 
// 
// Parameters:

#ifndef OPEN_SPIEL_GAMES_COUP_H_
#define OPEN_SPIEL_GAMES_COUP_H_

#include <array>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/observer.h"
#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"

namespace open_spiel {
namespace coup {

// Parameters

inline constexpr int kNumPlayers = 2;

class CoupGame;
class CoupObserver;

enum class CardType {
  kNone       = -1,
  kAssassin   = 0,
  kAmbassador = 1,
  kCaptain    = 2,
  kContessa   = 3,
  kDuke       = 4
};

enum class CardStateType {
  kNone     = -1,
  kFaceDown = 0,
  kFaceUp   = 1
};

enum class ActionType : Action {
  kNone                      = -1,
  kIncome                    = 0,
  kForeignAid                = 1,
  kCoup                      = 2,
  kTax                       = 3,
  kAssassinate               = 4,
  kExchange                  = 5,
  kSteal                     = 6,
  kLoseCard1                 = 7,
  kLoseCard2                 = 8,
  kPassFA                    = 9,
  kPassFABlock               = 10,
  kPassTax                   = 11,
  kPassExchange              = 12,
  kPassAssassinateBlock      = 13,
  kPassSteal                 = 14,
  kPassStealBlock            = 15,
  kBlockFA                   = 16,
  kBlockAssassinate          = 17,
  kBlockSteal                = 18,
  kChallengeFABlock          = 19,
  kChallengeTax              = 20,
  kChallengeExchange         = 21,
  kChallengeAssassinate      = 22,
  kChallengeAssassinateBlock = 23,
  kChallengeSteal            = 24,
  kChallengeStealBlock       = 25,
  kExchangeReturn12          = 26,
  kExchangeReturn13          = 27,
  kExchangeReturn14          = 28,
  kExchangeReturn23          = 29,
  kExchangeReturn24          = 30,
  kExchangeReturn34          = 31
};

struct CoupCard {
  CardType value;
  CardStateType state;
};

struct CoupPlayer {
  // Cards in hand
  std::vector<CoupCard> cards;
  // Number of coins
  int coins;
  // Last action taken
  ActionType last_action;
  // Whether player has lost a challenge & it needs to be resolved
  std::vector<bool> lost_challenge;
};

class CoupState : public State {
 public:
  explicit CoupState(std::shared_ptr<const Game> game);

  Player CurrentPlayer() const override;
  std::string ActionToString(Player player, Action move) const override;
  std::string ToString() const override;
  bool IsTerminal() const override;
  std::vector<double> Rewards() const override;
  std::vector<double> Returns() const override;
  std::string InformationStateString(Player player) const override;
  std::string ObservationString(Player player) const override;
  void InformationStateTensor(Player player,
                              absl::Span<float> values) const override;
  void ObservationTensor(Player player,
                         absl::Span<float> values) const override;
  std::unique_ptr<State> Clone() const override;
  // The probability of taking each possible action in a particular info state.
  std::vector<std::pair<Action, double>> ChanceOutcomes() const override;

  std::vector<Action> LegalActions() const override;

  // Returns a vector of MaxGameLength containing all of the betting actions
  // taken so far. If the round has ended, the actions are kInvalidAction.
  std::unique_ptr<State> ResampleFromInfostate(
      int player_id, std::function<double()> rng) const override;

  std::vector<Action> ActionsConsistentWithInformationFrom(
      Action action) const override {
    return {action};
  }

 protected:
  // The meaning of `action_id` varies:
  // - At decision nodes, one of ActionType.
  // - At a chance node, indicates the CardType to be dealt to the player
  void DoApplyAction(Action move) override;

 private:
  friend class CoupObserver;

  std::vector<Action> LegalLoseCardActions() const;

  void NextPlayerTurn();
  void NextPlayerMove();
  int NumObservableCards() const;
  int MaxBetsPerRound() const;

  // Counts of each card in deck. Index for each CardType (5). Count 0-3.
  std::vector<int> deck_;
  std::vector<CoupPlayer> players_;

  // "Turn" defines the overall turn of the game,
  // which can contain several sub-moves
  Player cur_player_turn_;
  // "Move" defines the current decision
  // Could be sub-move (response) in the turn (pass/block/challenge)
  Player cur_player_move_;
  // Opponent of cur_player_move_
  Player opp_player_;
  // Whether it is the beginning of a player's turn
  bool is_turn_begin_;
  // Track turns in addition to move_number_
  int turn_number_;
};

class CoupGame : public Game {
 public:
  explicit CoupGame(const GameParameters& params);

  int NumDistinctActions() const override { return 32; }
  std::unique_ptr<State> NewInitialState() const override;
  int MaxChanceOutcomes() const override;
  int NumPlayers() const override { return kNumPlayers; }
  double MinUtility() const override { return -2; }
  double MaxUtility() const override { return 2; }
  double UtilitySum() const override { return 0; }
  std::vector<int> InformationStateTensorShape() const override;
  std::vector<int> ObservationTensorShape() const override;

  // If neither player is playing to win, could be infinite.
  // Unlike chess, no rules on repeated moves.
  // Could continue to steal from eachother, or exchange with deck forever.
  // Choosing arbitrary large value.
  int MaxGameLength() const override { return 1000; }
  // Given MaxGameLength, overestimating chance nodes
  int MaxChanceNodesInHistory() const override { return 400; }
  
  //Serialize? ToString?
  std::string ActionToString(Player player, Action action) const override;
  // New Observation API
  std::shared_ptr<Observer> MakeObserver(
      absl::optional<IIGObservationType> iig_obs_type,
      const GameParameters& params) const override;

  // Used to implement the old observation API.
  std::shared_ptr<CoupObserver> default_observer_;
  std::shared_ptr<CoupObserver> info_state_observer_;
};

// Returns policy that always folds.
TabularPolicy GetAlwaysFoldPolicy(const Game& game);

// Returns policy that always calls.
TabularPolicy GetAlwaysCallPolicy(const Game& game);

// Returns policy that always raises.
TabularPolicy GetAlwaysRaisePolicy(const Game& game);

}  // namespace coup
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_COUP_H_
