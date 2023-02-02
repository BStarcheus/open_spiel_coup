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
// 2-player version only
// 
// A board game based on deception. The goal is to eliminate opponents' cards
// and be the last player standing. Use your cards' abilities, or bluff and
// use other abilities. Challenge opponents if you think they are bluffing.
// https://www.ultraboardgames.com/coup/game-rules.php

#ifndef OPEN_SPIEL_GAMES_COUP_H_
#define OPEN_SPIEL_GAMES_COUP_H_

#include <array>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <queue>
#include <map>

#include "open_spiel/observer.h"
#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"

namespace open_spiel {
namespace coup {

// Constants
inline constexpr int kNumPlayers = 2;
inline constexpr int kMaxCardsInHand = 4;
inline constexpr int kNumCardTypes = 5;
inline constexpr int kNumEachCardInDeck = 3;

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
  kPass                      = 9,
  kBlock                     = 10,
  kChallenge                 = 11,
  kExchangeReturn12          = 12,
  kExchangeReturn13          = 13,
  kExchangeReturn14          = 14,
  kExchangeReturn23          = 15,
  kExchangeReturn24          = 16,
  kExchangeReturn34          = 17
};

struct CoupCard {
  CardType value;
  CardStateType state;

  bool operator < (const CoupCard &obj) const {
    return (value < obj.value ||
              (value == obj.value && state < obj.state));
  }
};

struct CoupPlayer {
  // Cards in hand
  std::vector<CoupCard> cards;
  // Number of coins
  int coins;
  // Last action taken
  ActionType last_action;
  // Whether player has lost a challenge & it needs to be resolved
  bool lost_challenge;
  bool HasFaceDownCard(CardType card);
  // Sort cards to reduce state space
  void SortCards();
};

class CoupState : public State {
 public:
  CoupState(std::shared_ptr<const Game> game);
  CoupState(const CoupState&) = default;

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

  std::vector<Action> ActionsConsistentWithInformationFrom(
      Action action) const override {
    return {action};
  }

  // Convenience accessors
  std::vector<CardType> GetCardsValue(Player player) const;
  std::vector<CardStateType> GetCardsState(Player player) const;
  int GetCoins(Player player) const;
  Action GetLastAction(Player player) const;

 protected:
  // The meaning of `action_id` varies:
  // - At decision nodes, one of ActionType.
  // - At a chance node, indicates the CardType to be dealt to the player
  void DoApplyAction(Action move) override;

 private:
  friend class CoupObserver;

  std::vector<Action> LegalLoseCardActions() const;

  // Helper used when cur_player_move_ challenged opp and lost,
  // so opp needs a new card
  void ChallengeFailReplaceCard(CardType card);

  void NextPlayerTurn();
  void NextPlayerMove();

  // Counts of each card in deck. Index for each CardType (5). Count 0-3.
  std::vector<int> deck_;
  std::vector<CoupPlayer> players_;

  // Queue of which player to deal cards to.
  // Game should stay in chance nodes until queue is empty.
  std::queue<Player> deal_card_to_;

  // "Turn" defines the overall turn of the game,
  // which can contain several sub-moves
  Player cur_player_turn_;
  // "Move" defines the current decision
  // Could be sub-move (response) in the turn (pass/block/challenge)
  Player cur_player_move_;
  // Opponent of cur_player_move_
  Player opp_player_;
  // Track which player is being dealt a card at chance nodes.
  // Map the index of the chance node in history_ to the player.
  // Used for creating action sequence for perfect recall.
  std::map<int, int> history_chance_deal_player_;
  // Whether it is the beginning of a player's turn
  bool is_turn_begin_;
  // Track turns in addition to move_number_
  int turn_number_;
  // Whether currently at a chance node.
  // Exists for Challenge Fails:
  // cp challenges, loses, must replace op card (chance node),
  // but then return to cp move so they can lose a card.
  // cur_player_move_ must have the player to go back to,
  // so need to track chance here instead.
  bool is_chance_;
  // Track the current reward in a single time step (by each player's perspective)
  // Reset each player's reward at the beginning of their move (DoApplyAction)
  std::vector<double> cur_rewards_;
};

class CoupGame : public Game {
 public:
  explicit CoupGame(const GameParameters& params);

  int NumDistinctActions() const override { return 18; }
  std::unique_ptr<State> NewInitialState() const override;
  int MaxChanceOutcomes() const override { return kNumCardTypes; }
  int NumPlayers() const override { return kNumPlayers; }
  double MinUtility() const override { return -2; }
  double MaxUtility() const override { return 2; }
  double UtilitySum() const override { return 0; }
  std::vector<int> InformationStateTensorShape() const override;
  std::vector<int> ObservationTensorShape() const override;

  // If neither player is playing to win, could be infinite.
  // Unlike chess, no rules on repeated moves.
  // Could continue to steal from eachother, or exchange with deck forever,
  // but we don't want to allow those games.
  // Choosing game length based on possible game where
  // P2 always exchanging, P1 always taking income unless forced to Coup.
  int MaxGameLength() const override { return 90; }
  int MaxChanceNodesInHistory() const override { return 45; }

  std::string ActionToString(Player player, Action action) const override;
  // New Observation API
  std::shared_ptr<Observer> MakeObserver(
      absl::optional<IIGObservationType> iig_obs_type,
      const GameParameters& params) const override;

  // Used to implement the old observation API.
  std::shared_ptr<CoupObserver> default_observer_;
  std::shared_ptr<CoupObserver> info_state_observer_;
};
}  // namespace coup
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_COUP_H_
