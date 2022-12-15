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

#include "open_spiel/games/coup.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <numeric>
#include <utility>

#include "open_spiel/abseil-cpp/absl/algorithm/container.h"
#include "open_spiel/abseil-cpp/absl/memory/memory.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/abseil-cpp/absl/strings/str_join.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/observer.h"
#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace coup {
namespace {

const GameType kGameType{/*short_name=*/"coup",
                         /*long_name=*/"Coup",
                         GameType::Dynamics::kSequential,
                         GameType::ChanceMode::kExplicitStochastic,
                         GameType::Information::kImperfectInformation,
                         GameType::Utility::kZeroSum,
                         GameType::RewardModel::kRewards,
                         /*max_num_players=*/2,
                         /*min_num_players=*/2,
                         /*provides_information_state_string=*/true,
                         /*provides_information_state_tensor=*/true,
                         /*provides_observation_string=*/true,
                         /*provides_observation_tensor=*/true,
                         /*parameter_specification=*/
                         {{"players", GameParameter(kNumPlayers)},
                          {"action_mapping", GameParameter(false)},
                          {"suit_isomorphism", GameParameter(false)}}};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new CoupGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

std::string StatelessActionToString(Action action) {
  if (action == ActionType::kFold) {
    return "Fold";
  } else if (action == ActionType::kCall) {
    return "Call";
  } else if (action == ActionType::kRaise) {
    return "Raise";
  } else {
    SpielFatalError(absl::StrCat("Unknown action: ", action));
    return "Will not return.";
  }
}

// Provides the observations / infostates as defined on the state
// as a single tensor.
std::shared_ptr<Observer> MakeSingleTensorObserver(
    const Game& game, absl::optional<IIGObservationType> iig_obs_type,
    const GameParameters& params) {
  return std::shared_ptr<Observer>(game.MakeBuiltInObserver(iig_obs_type));
}

ObserverRegisterer single_tensor(
    kGameType.short_name, "single_tensor", MakeSingleTensorObserver);
}  // namespace

// The Observer class is responsible for creating representations of the game
// state for use in learning algorithms. It handles both string and tensor
// representations, and any combination of public information and private
// information (none, observing player only, or all players).
//
// If a perfect recall observation is requested, it must be possible to deduce
// all previous observations for the same information type from the current
// observation.

class CoupObserver : public Observer {
 public:
  CoupObserver(IIGObservationType iig_obs_type)
      : Observer(/*has_string=*/true, /*has_tensor=*/true),
        iig_obs_type_(iig_obs_type) {}

  //
  // These helper methods each write a piece of the tensor observation.
  //

  // Identity of the observing player. One-hot vector of size num_players.
  static void WriteObservingPlayer(const CoupState& state, int player,
                                   Allocator* allocator) {
    auto out = allocator->Get("player", {state.num_players_});
    out.at(player) = 1;
  }

  // Private card of the observing player. One-hot vector of size num_cards.
  static void WriteSinglePlayerCard(const CoupState& state, int player,
                                    Allocator* allocator) {
    auto out = allocator->Get("private_card", {state.NumObservableCards()});
    int card = state.private_cards_[player];
    if (card != kInvalidCard) out.at(card) = 1;
  }

  // Private cards of all players. Tensor of shape [num_players, num_cards].
  static void WriteAllPlayerCards(const CoupState& state,
                                  Allocator* allocator) {
    auto out = allocator->Get("private_cards",
                              {state.num_players_, state.NumObservableCards()});
    for (int p = 0; p < state.num_players_; ++p) {
      int card = state.private_cards_[p];
      if (card != kInvalidCard) out.at(p, state.private_cards_[p]) = 1;
    }
  }

  // Community card (if any). One-hot vector of size num_cards.
  static void WriteCommunityCard(const CoupState& state,
                                 Allocator* allocator) {
    auto out = allocator->Get("community_card", {state.NumObservableCards()});
    if (state.public_card_ != kInvalidCard) {
      out.at(state.public_card_) = 1;
    }
  }

  // Betting sequence; shape [num_rounds, bets_per_round, num_actions].
  static void WriteBettingSequence(const CoupState& state,
                                   Allocator* allocator) {
    const int kNumRounds = 2;
    const int kBitsPerAction = 2;
    const int max_bets_per_round = state.MaxBetsPerRound();
    auto out = allocator->Get("betting",
                              {kNumRounds, max_bets_per_round, kBitsPerAction});
    for (int round : {0, 1}) {
      const auto& bets =
          (round == 0) ? state.round1_sequence_ : state.round2_sequence_;
      for (int i = 0; i < bets.size(); ++i) {
        if (bets[i] == ActionType::kCall) {
          out.at(round, i, 0) = 1;  // Encode call as 10.
        } else if (bets[i] == ActionType::kRaise) {
          out.at(round, i, 1) = 1;  // Encode raise as 01.
        }
      }
    }
  }

  // Pot contribution per player (integer per player).
  static void WritePotContribution(const CoupState& state,
                                   Allocator* allocator) {
    auto out = allocator->Get("pot_contribution", {state.num_players_});
    for (auto p = Player{0}; p < state.num_players_; p++) {
      out.at(p) = state.ante_[p];
    }
  }

  // Writes the complete observation in tensor form.
  // The supplied allocator is responsible for providing memory to write the
  // observation into.
  void WriteTensor(const State& observed_state, int player,
                   Allocator* allocator) const override {
    auto& state = open_spiel::down_cast<const CoupState&>(observed_state);
    SPIEL_CHECK_GE(player, 0);
    SPIEL_CHECK_LT(player, state.num_players_);

    // Observing player.
    WriteObservingPlayer(state, player, allocator);

    // Private card(s).
    if (iig_obs_type_.private_info == PrivateInfoType::kSinglePlayer) {
      WriteSinglePlayerCard(state, player, allocator);
    } else if (iig_obs_type_.private_info == PrivateInfoType::kAllPlayers) {
      WriteAllPlayerCards(state, allocator);
    }

    // Public information.
    if (iig_obs_type_.public_info) {
      WriteCommunityCard(state, allocator);
      iig_obs_type_.perfect_recall ? WriteBettingSequence(state, allocator)
                                   : WritePotContribution(state, allocator);
    }
  }

  // Writes an observation in string form. It would be possible just to
  // turn the tensor observation into a string, but we prefer something
  // somewhat human-readable.

  std::string StringFrom(const State& observed_state,
                         int player) const override {
    auto& state = open_spiel::down_cast<const CoupState&>(observed_state);
    SPIEL_CHECK_GE(player, 0);
    SPIEL_CHECK_LT(player, state.num_players_);
    std::string result;

    // Private card(s).
    if (iig_obs_type_.private_info == PrivateInfoType::kSinglePlayer) {
      absl::StrAppend(&result, "[Observer: ", player, "]");
      absl::StrAppend(&result, "[Private: ", state.private_cards_[player], "]");
    } else if (iig_obs_type_.private_info == PrivateInfoType::kAllPlayers) {
      absl::StrAppend(
          &result, "[Privates: ", absl::StrJoin(state.private_cards_, ""), "]");
    }

    // Public info. Not all of this is strictly necessary, but it makes the
    // string easier to understand.
    if (iig_obs_type_.public_info) {
      absl::StrAppend(&result, "[Round ", state.round_, "]");
      absl::StrAppend(&result, "[Player: ", state.cur_player_, "]");
      absl::StrAppend(&result, "[Pot: ", state.pot_, "]");
      absl::StrAppend(&result, "[Money: ", absl::StrJoin(state.money_, " "),
                      "]");
      if (state.public_card_ != kInvalidCard) {
        absl::StrAppend(&result, "[Public: ", state.public_card_, "]");
      }
      if (iig_obs_type_.perfect_recall) {
        // Betting Sequence (for the perfect recall case)
        absl::StrAppend(
            &result, "[Round1: ", absl::StrJoin(state.round1_sequence_, " "),
            "][Round2: ", absl::StrJoin(state.round2_sequence_, " "), "]");
      } else {
        // Pot contributions (imperfect recall)
        absl::StrAppend(&result, "[Ante: ", absl::StrJoin(state.ante_, " "),
                        "]");
      }
    }

    // Done.
    return result;
  }

 private:
  IIGObservationType iig_obs_type_;
};

CoupState::CoupState(std::shared_ptr<const Game> game)
    : State(game),
      cur_player_(kChancePlayerId) {
  
  // Start game. Create deck
  // Deal cards is a chance node, so wait
}

int CoupState::CurrentPlayer() const {
  if (IsTerminal()) {
    return kTerminalPlayerId;
  } else {
    return cur_player_;
  }
}

// In a chance node, `move` should be the card to deal to the current
// underlying player.
// On a player node, it should be ActionType
void CoupState::DoApplyAction(Action move) {
  if (IsChanceNode()) {
    SPIEL_CHECK_GE(move, 0);
    SPIEL_CHECK_LT(move, deck_.size());
    if (suit_isomorphism_) {
      // One of the two identical cards must be left in the deck.
      SPIEL_CHECK_TRUE(deck_[move * 2] != kInvalidCard ||
                       deck_[move * 2 + 1] != kInvalidCard);
    } else {
      SPIEL_CHECK_NE(deck_[move], kInvalidCard);
    }

    if (private_cards_dealt_ < num_players_) {
      SetPrivate(private_cards_dealt_, move);
    } else {
      // Round 2: A single public card.
      if (suit_isomorphism_) {
        public_card_ = move;
        if (deck_[move * 2] != kInvalidCard) {
          deck_[move * 2] = kInvalidCard;
        } else if (deck_[move * 2 + 1] != kInvalidCard) {
          deck_[move * 2 + 1] = kInvalidCard;
        } else {
          SpielFatalError("Suit isomorphism error.");
        }
        deck_size_--;
      } else {
        public_card_ = deck_[move];
        deck_[move] = kInvalidCard;
        deck_size_--;
      }

      // We have finished the public card, let's bet!
      cur_player_ = NextPlayer();
    }
  } else {
    // Player node.
    if (action_mapping_) {
      // Map otherwise illegal actions to kCall.
      if (move == ActionType::kFold) {
        if (stakes_ <= ante_[cur_player_]) {
          move = ActionType::kCall;
        }
      } else if (move == ActionType::kRaise) {
        if (num_raises_ >= 2) {
          move = ActionType::kCall;
        }
      }
    }

    if (move == ActionType::kFold) {
      SPIEL_CHECK_NE(cur_player_, kChancePlayerId);
      SequenceAppendMove(ActionType::kFold);

      // Player is now out.
      folded_[cur_player_] = true;
      remaining_players_--;

      if (IsTerminal()) {
        ResolveWinner();
      } else if (ReadyForNextRound()) {
        NewRound();
      } else {
        cur_player_ = NextPlayer();
      }
    } else if (move == ActionType::kCall) {
      SPIEL_CHECK_NE(cur_player_, kChancePlayerId);

      // Current player puts in an amount of money equal to the current level
      // (stakes) minus what they have contributed to level their contribution
      // off. Note: this action also acts as a 'check' where the stakes are
      // equal to each player's ante.
      SPIEL_CHECK_GE(stakes_, ante_[cur_player_]);
      int amount = stakes_ - ante_[cur_player_];
      Ante(cur_player_, amount);
      num_calls_++;
      SequenceAppendMove(ActionType::kCall);

      if (IsTerminal()) {
        ResolveWinner();
      } else if (ReadyForNextRound()) {
        NewRound();
      } else {
        cur_player_ = NextPlayer();
      }
    } else if (move == ActionType::kRaise) {
      SPIEL_CHECK_NE(cur_player_, kChancePlayerId);

      // This player matches the current stakes and then brings the stakes up.
      SPIEL_CHECK_LT(num_raises_, kMaxRaises);
      int call_amount = stakes_ - ante_[cur_player_];

      // First, match the current stakes if necessary
      SPIEL_CHECK_GE(call_amount, 0);
      if (call_amount > 0) {
        Ante(cur_player_, call_amount);
      }

      // Now, raise the stakes.
      int raise_amount = (round_ == 1 ? kFirstRaiseAmount : kSecondRaiseAmount);
      stakes_ += raise_amount;
      Ante(cur_player_, raise_amount);
      num_raises_++;
      num_calls_ = 0;
      SequenceAppendMove(ActionType::kRaise);

      if (IsTerminal()) {
        ResolveWinner();
      } else {
        cur_player_ = NextPlayer();
      }
    } else {
      SpielFatalError(absl::StrCat("Move ", move, " is invalid. ChanceNode?",
                                   IsChanceNode()));
    }
  }
}

std::vector<Action> CoupState::LegalLoseCardActions() const {
  // Show which cards the player can lose
  std::vector<Action> legal;
  CoupPlayer &p = players_.at(cur_player_move_);
  if (p.cards.at(0).state == CardStateType::kFaceDown) {
    legal.push_back(ActionType::kLoseCard1)
  }
  if (p.cards.at(1).state == CardStateType::kFaceDown) {
    legal.push_back(ActionType::kLoseCard2)
  }
  return legal;
}

std::vector<Action> CoupState::LegalActions() const {
  if (IsTerminal()) return {};
  std::vector<Action> legal;

  if (IsChanceNode()) {
    // All chance nodes are the action of drawing a card
    // 5 types of cards
    legal = {0, 1, 2, 3, 4};
    return legal;
  }
  
  // else Decision node
  CoupPlayer &cp = players_.at(cur_player_move_);
  CoupPlayer &op = players_.at(opp_player_);
  if (is_turn_begin_) {
    if (cp.coins >= 10) {
      legal.push_back(ActionType::kCoup);
      return legal;
    }

    legal.push_back(ActionType::kIncome);
    legal.push_back(ActionType::kForeignAid);
    if (cp.coins >= 7) legal.push_back(ActionType::kCoup);
    legal.push_back(ActionType::kTax);
    if (cp.coins >= 3) legal.push_back(ActionType::kAssassinate);
    legal.push_back(ActionType::kExchange);
    if (op.coins > 0) legal.push_back(ActionType::kSteal);
    return legal;

  } else if (cp.lost_challenge) {
    // Player lost challenge and needs to lose a card
    return LegalLoseCardActions();

  } else if (cur_player_move_ != cur_player_turn_) {
    // opponent's turn, so cur_player_move_ can
    // choose to block or challenge for certain actions
    if (op.last_action == ActionType::kForeignAid) {
      legal = {ActionType::kPassFA, 
               ActionType::kBlockFA};
      return legal;
    } else if (op.last_action == ActionType::kTax) {
      legal = {ActionType::kPassTax, 
               ActionType::kChallengeTax};
      return legal;
    } else if (op.last_action == ActionType::kExchange) {
      legal = {ActionType::kPassExchange,
               ActionType::kChallengeExchange};
      return legal;
    } else if (op.last_action == ActionType::kSteal) {
      legal = {ActionType::kPassSteal, 
               ActionType::kBlockSteal,
               ActionType::kChallengeSteal};
      return legal;
    } else if (op.last_action == ActionType::kAssassinate) {
      legal = LegalLoseCardActions();
      legal.push_back(ActionType::kBlockAssassinate);
      legal.push_back(ActionType::kChallengeAssassinate);
      return legal;
    } else if (op.last_action == ActionType::kCoup) {
      legal = LegalLoseCardActions();
      return legal;
    } else {
      SpielFatalError("Error in LegalActions(): Invalid action progression");
    }

  } else if (cp.last_action == ActionType::kExchange) {
    // Opponent has passed, so cp can continue with exchange
    if (cp.cards.size() < 4) {
      SpielFatalError("Error in LegalActions(): Player mid-exchange should have 4 cards");
    }

    // Find index of single face up card, if any
    int faceUpInd = -1;
    for (int i = 0; i < cp.cards.size(); ++i) {
      if (cp.cards.at(i).state == CardStateType::kFaceUp) {
        faceUpInd = i;
        break;
      }
    }

    if (faceUpInd == -1) {
      legal = {ActionType::kExchangeReturn12,
               ActionType::kExchangeReturn13,
               ActionType::kExchangeReturn14,
               ActionType::kExchangeReturn23,
               ActionType::kExchangeReturn24,
               ActionType::kExchangeReturn34};
    } else if (faceUpInd == 0) {
      legal = {ActionType::kExchangeReturn23,
               ActionType::kExchangeReturn24,
               ActionType::kExchangeReturn34};
    } else if (faceUpInd == 1) {
      legal = {ActionType::kExchangeReturn13,
               ActionType::kExchangeReturn14,
               ActionType::kExchangeReturn34};
    } else if (faceUpInd == 2) {
      legal = {ActionType::kExchangeReturn12,
               ActionType::kExchangeReturn14,
               ActionType::kExchangeReturn24};
    } else if (faceUpInd == 3) {
      legal = {ActionType::kExchangeReturn12,
               ActionType::kExchangeReturn13,
               ActionType::kExchangeReturn23};
    }
    return legal;

  } else if (op.last_action == ActionType::kBlockFA) {
    legal = {ActionType::kPassFABlock,
             ActionType::kChallengeFABlock};
    return legal;

  } else if (cp.last_action == ActionType::kBlockAssassinate) {
    legal = {ActionType::kPassAssassinateBlock,
             ActionType::kChallengeAssassinateBlock};
    return legal;

  } else if (cp.last_action == ActionType::kBlockSteal) {
    legal = {ActionType::kPassStealBlock,
             ActionType::kChallengeStealBlock};
    return legal;

  } else {
    SpielFatalError("Error in LegalActions(): Invalid action progression");
  }
}

std::string CoupState::ActionToString(Player player, Action move) const {
  return GetGame()->ActionToString(player, move);
}

std::string CoupState::ToString() const {
  std::string result;

  absl::StrAppend(&result, "Round: ", round_, "\nPlayer: ", cur_player_,
                  "\nPot: ", pot_, "\nMoney (p1 p2 ...):");
  for (auto p = Player{0}; p < num_players_; p++) {
    absl::StrAppend(&result, " ", money_[p]);
  }
  absl::StrAppend(&result, "\nCards (public p1 p2 ...): ", public_card_, " ");
  for (Player player_index = 0; player_index < num_players_; player_index++) {
    absl::StrAppend(&result, private_cards_[player_index], " ");
  }

  absl::StrAppend(&result, "\nRound 1 sequence: ");
  for (int i = 0; i < round1_sequence_.size(); ++i) {
    Action action = round1_sequence_[i];
    if (i > 0) absl::StrAppend(&result, ", ");
    absl::StrAppend(&result, StatelessActionToString(action));
  }
  absl::StrAppend(&result, "\nRound 2 sequence: ");
  for (int i = 0; i < round2_sequence_.size(); ++i) {
    Action action = round2_sequence_[i];
    if (i > 0) absl::StrAppend(&result, ", ");
    absl::StrAppend(&result, StatelessActionToString(action));
  }
  absl::StrAppend(&result, "\n");

  return result;
}

bool CoupState::IsTerminal() const {
  int numPlayersAlive = 0;
  for (auto &p: players_) {
    for (auto &c: p.cards) {
      if (c.state == CardStateType::kFaceDown) {
        numPlayersAlive += 1; 
        break;
      }
    }
  }
  if (numPlayersAlive > 1) return false;
  else return true;
}

std::vector<double> Rewards() const {

}

std::vector<double> CoupState::Returns() const {
  if (!IsTerminal()) {
    return std::vector<double>(num_players_, 0.0);
  }

  std::vector<double> returns(num_players_);
  for (auto player = Player{0}; player < num_players_; ++player) {
    // Money vs money at start.
    returns[player] = money_[player] - kStartingMoney;
  }

  return returns;
}

// Information state is card then bets.
std::string CoupState::InformationStateString(Player player) const {
  const CoupGame& game = open_spiel::down_cast<const CoupGame&>(*game_);
  return game.info_state_observer_->StringFrom(*this, player);
}

// Observation is card then contribution of each players to the pot.
std::string CoupState::ObservationString(Player player) const {
  const CoupGame& game = open_spiel::down_cast<const CoupGame&>(*game_);
  return game.default_observer_->StringFrom(*this, player);
}

void CoupState::InformationStateTensor(Player player,
                                        absl::Span<float> values) const {
  ContiguousAllocator allocator(values);
  const CoupGame& game = open_spiel::down_cast<const CoupGame&>(*game_);
  game.info_state_observer_->WriteTensor(*this, player, &allocator);
}

void CoupState::ObservationTensor(Player player,
                                   absl::Span<float> values) const {
  ContiguousAllocator allocator(values);
  const CoupGame& game = open_spiel::down_cast<const CoupGame&>(*game_);
  game.default_observer_->WriteTensor(*this, player, &allocator);
}

std::unique_ptr<State> CoupState::Clone() const {
  return std::unique_ptr<State>(new CoupState(*this));
}

std::vector<std::pair<Action, double>> CoupState::ChanceOutcomes() const {
  SPIEL_CHECK_TRUE(IsChanceNode());
  std::vector<std::pair<Action, double>> outcomes;

  if (suit_isomorphism_) {
    const double p = 1.0 / deck_size_;
    // Consecutive cards in deck are viewed identically.
    for (int card = 0; card < deck_.size() / 2; card++) {
      if (deck_[card * 2] != kInvalidCard &&
          deck_[card * 2 + 1] != kInvalidCard) {
        outcomes.push_back({card, p * 2});
      } else if (deck_[card * 2] != kInvalidCard ||
                 deck_[card * 2 + 1] != kInvalidCard) {
        outcomes.push_back({card, p});
      }
    }
    return outcomes;
  }

  const double p = 1.0 / deck_size_;
  for (int card = 0; card < deck_.size(); card++) {
    // This card is still in the deck, prob is 1/decksize.
    if (deck_[card] != kInvalidCard) outcomes.push_back({card, p});
  }
  return outcomes;
}

void CoupState::NextPlayerTurn() {
  cur_player_turn_ = 1 - cur_player_turn_;
  // Player always has first move on their turn
  cur_player_move_ = cur_player_turn_;
  ++turn_number_;
  is_turn_begin_ = true;
}

void CoupState::NextPlayerMove() {
  cur_player_move_ = 1 - cur_player_move_;
  is_turn_begin_ = false;
}

std::unique_ptr<State> CoupState::ResampleFromInfostate(
    int player_id, std::function<double()> rng) const {
  std::unique_ptr<State> clone = game_->NewInitialState();

  // First, deal out cards:
  Action player_chance = history_.at(player_id).action;
  for (int p = 0; p < GetGame()->NumPlayers(); ++p) {
    if (p == player_id) {
      clone->ApplyAction(history_.at(p).action);
    } else {
      Action chosen_action = player_chance;
      while (chosen_action == player_chance || chosen_action == public_card_) {
        chosen_action = SampleAction(clone->ChanceOutcomes(), rng()).first;
      }
      clone->ApplyAction(chosen_action);
    }
  }
  for (int action : round1_sequence_) clone->ApplyAction(action);
  if (public_card_ != kInvalidCard) {
    clone->ApplyAction(public_card_);
    for (int action : round2_sequence_) clone->ApplyAction(action);
  }
  return clone;
}

CoupGame::CoupGame(const GameParameters& params)
    : Game(kGameType, params) {
  default_observer_ = std::make_shared<CoupObserver>(kDefaultObsType);
  info_state_observer_ = std::make_shared<CoupObserver>(kInfoStateObsType);
}

std::unique_ptr<State> CoupGame::NewInitialState() const {
  return absl::make_unique<CoupState>(shared_from_this());
}

int CoupGame::MaxChanceOutcomes() const {
  if (suit_isomorphism_) {
    return total_cards_ / 2;
  } else {
    return total_cards_;
  }
}

std::vector<int> CoupGame::InformationStateTensorShape() const {
  // One-hot encoding for player number (who is to play).
  // 2 slots of cards (total_cards_ bits each): private card, public card
  // Followed by maximum game length * 2 bits each (call / raise)
  if (suit_isomorphism_) {
    return {(num_players_) + (total_cards_) + (MaxGameLength() * 2)};
  } else {
    return {(num_players_) + (total_cards_ * 2) + (MaxGameLength() * 2)};
  }
}

std::vector<int> CoupGame::ObservationTensorShape() const {
  // One-hot encoding for player number (who is to play).
  // 2 slots of cards (total_cards_ bits each): private card, public card
  // Followed by the contribution of each player to the pot
  if (suit_isomorphism_) {
    return {(num_players_) + (total_cards_) + (num_players_)};
  } else {
    return {(num_players_) + (total_cards_ * 2) + (num_players_)};
  }
}

std::shared_ptr<Observer> CoupGame::MakeObserver(
    absl::optional<IIGObservationType> iig_obs_type,
    const GameParameters& params) const {
  if (params.empty()) {
    return std::make_shared<CoupObserver>(
        iig_obs_type.value_or(kDefaultObsType));
  } else {
    return MakeRegisteredObserver(iig_obs_type, params);
  }
}

std::string CoupGame::ActionToString(Player player, Action action) const {
  if (player == kChancePlayerId) {
    return absl::StrCat("Chance outcome:", action);
  } else {
    return StatelessActionToString(action);
  }
}

TabularPolicy GetAlwaysFoldPolicy(const Game& game) {
  SPIEL_CHECK_TRUE(
      dynamic_cast<CoupGame*>(const_cast<Game*>(&game)) != nullptr);
  return GetPrefActionPolicy(game, {ActionType::kFold, ActionType::kCall});
}

TabularPolicy GetAlwaysCallPolicy(const Game& game) {
  SPIEL_CHECK_TRUE(
      dynamic_cast<CoupGame*>(const_cast<Game*>(&game)) != nullptr);
  return GetPrefActionPolicy(game, {ActionType::kCall});
}

TabularPolicy GetAlwaysRaisePolicy(const Game& game) {
  SPIEL_CHECK_TRUE(
      dynamic_cast<CoupGame*>(const_cast<Game*>(&game)) != nullptr);
  return GetPrefActionPolicy(game, {ActionType::kRaise, ActionType::kCall});
}

}  // namespace coup
}  // namespace open_spiel
