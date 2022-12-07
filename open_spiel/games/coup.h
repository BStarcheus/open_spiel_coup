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

// A generalized version of a Leduc poker, a simple but non-trivial poker game
// described in http://poker.cs.ualberta.ca/publications/UAI05.pdf .
//
// Taken verbatim from the linked paper above: "In Leduc hold'em, the deck
// consists of two suits with three cards in each suit. There are two rounds.
// In the first round a single private card is dealt to each player. In the
// second round a single board card is revealed. There is a two-bet maximum,
// with raise amounts of 2 and 4 in the first and second round, respectively.
// Both players start the first round with 1 already in the pot.
//
// So the maximin sequence is of the form:
// private card player 0, private card player 1, [bets], public card, [bets]
//
// Parameters:
//     "players"           int    number of players          (default = 2)
//     "action_mapping"    bool   regard all actions as legal and internally
//                                map otherwise illegal actions to check/call
//                                                           (default = false)
//     "suit_isomorphism"  bool   player observations do not distinguish
//                                between cards of different suits with
//                                the same rank              (default = false)

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

inline constexpr int kInvalidCard = -10000;
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

class CoupState : public State {
 public:
  explicit CoupState(std::shared_ptr<const Game> game);

  Player CurrentPlayer() const override;
  std::string ActionToString(Player player, Action move) const override;
  std::string ToString() const override;
  bool IsTerminal() const override;
  //Rewards
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
  // - At a chance node, indicates the card to be dealt to the player or
  // revealed publicly. The interpretation of each chance outcome depends on
  // the number of players, but always follows:
  //    lowest value of first suit,
  //    lowest value of second suit,
  //    next lowest value of first suit,
  //    next lowest value of second suit,
  //             .
  //             .
  //             .
  //    highest value of first suit,
  //    highest value of second suit.
  // So, e.g. in the two player case (6 cards): 0 = Jack1, 1 = Jack2,
  // 2 = Queen1, ... , 5 = King2.
  void DoApplyAction(Action move) override;

 private:
  friend class CoupObserver;

  int NextPlayer() const;
  int NumObservableCards() const;
  int MaxBetsPerRound() const;

  // Fields sets to bad/invalid values. Use Game::NewInitialState().
  Player cur_player_;

  // Cards by value (0-6 for standard 2-player game, -1 if no longer in the
  // deck.)
  std::vector<int> deck_;
};

class CoupGame : public Game {
 public:
  explicit CoupGame(const GameParameters& params);

  int NumDistinctActions() const override { return 32; }
  std::unique_ptr<State> NewInitialState() const override;
  int MaxChanceOutcomes() const override;
  int NumPlayers() const override { return num_players_; }
  double MinUtility() const override;
  double MaxUtility() const override;
  double UtilitySum() const override { return 0; }
  std::vector<int> InformationStateTensorShape() const override;
  std::vector<int> ObservationTensorShape() const override;
  int MaxGameLength() const override {
    // 2 rounds.
    return 2 * MaxBetsPerRound();
  }
  int MaxChanceNodesInHistory() const override { return 3; }
  //Serialize? ToString?
  std::string ActionToString(Player player, Action action) const override;
  // New Observation API
  std::shared_ptr<Observer> MakeObserver(
      absl::optional<IIGObservationType> iig_obs_type,
      const GameParameters& params) const override;

  // Used to implement the old observation API.
  std::shared_ptr<CoupObserver> default_observer_;
  std::shared_ptr<CoupObserver> info_state_observer_;

 private:
  int num_players_;  // Number of players.
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
