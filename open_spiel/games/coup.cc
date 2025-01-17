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
                         {}};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new CoupGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

std::string StatelessCardToString(CardType card) {
  if (card == CardType::kNone) {
    return "-";
  } else if (card == CardType::kAssassin) {
    return "Assassin";
  } else if (card == CardType::kAmbassador) {
    return "Ambassador";
  } else if (card == CardType::kCaptain) {
    return "Captain";
  } else if (card == CardType::kContessa) {
    return "Contessa";
  } else if (card == CardType::kDuke) {
    return "Duke";
  } else {
    SpielFatalError(absl::StrCat("Unknown card: ", card));
    return "Will not return.";
  }
}

std::string StatelessCardStateToString(CardStateType cardState) {
  if (cardState == CardStateType::kNone) {
    return "None";
  } else if (cardState == CardStateType::kFaceDown) {
    return "FaceDown";
  } else if (cardState == CardStateType::kFaceUp) {
    return "FaceUp";
  } else {
    SpielFatalError(absl::StrCat("Unknown card state type: ", cardState));
    return "Will not return.";
  }
}

std::string StatelessActionToString(ActionType action) {
  if (action == ActionType::kNone) {
    return "None";
  } else if (action == ActionType::kIncome) {
    return "Income";
  } else if (action == ActionType::kForeignAid) {
    return "ForeignAid";
  } else if (action == ActionType::kCoup) {
    return "Coup";
  } else if (action == ActionType::kTax) {
    return "Tax";
  } else if (action == ActionType::kAssassinate) {
    return "Assassinate";
  } else if (action == ActionType::kExchange) {
    return "Exchange";
  } else if (action == ActionType::kSteal) {
    return "Steal";
  } else if (action == ActionType::kLoseCard1) {
    return "LoseCard1";
  } else if (action == ActionType::kLoseCard2) {
    return "LoseCard2";
  } else if (action == ActionType::kPass) {
    return "Pass";
  } else if (action == ActionType::kBlock) {
    return "Block";
  } else if (action == ActionType::kChallenge) {
    return "Challenge";
  } else if (action == ActionType::kExchangeReturn12) {
    return "ExchangeReturn12";
  } else if (action == ActionType::kExchangeReturn13) {
    return "ExchangeReturn13";
  } else if (action == ActionType::kExchangeReturn14) {
    return "ExchangeReturn14";
  } else if (action == ActionType::kExchangeReturn23) {
    return "ExchangeReturn23";
  } else if (action == ActionType::kExchangeReturn24) {
    return "ExchangeReturn24";
  } else if (action == ActionType::kExchangeReturn34) {
    return "ExchangeReturn34";
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


class CoupObserver : public Observer {
 public:
  CoupObserver(IIGObservationType iig_obs_type)
      : Observer(/*has_string=*/true, /*has_tensor=*/true),
        iig_obs_type_(iig_obs_type) {}

  // Helper methods to write each piece of the tensor

  // Identity of player. One-hot vector of size num_players.
  // Used both for observing player and player whose move it is.
  static void WritePlayer(const CoupState& state, int player,
                          Allocator* allocator,
                          std::string prefix) {
    auto out = allocator->Get(prefix + "player", {state.num_players_});
    out.at(player) = 1;
  }

  // The following card tensors contain a one-hot vector for each card
  // (for both value and state).
  // This is so that the card value can correspond to the card state by index.
  // We store kMaxCardsInHand regardless of how many cards they have in order
  // to have a constant size tensor.
  // If a card is hidden/private or non-existent the value vector is all 0.
  // If a card is non-existent (e.g. player only has 2 cards in hand,
  // so cards 3,4 are "non-existent") the state vector is all 0.

  // Write the card values for a player depending on whether
  // the observation includes private and/or public
  static void WritePlayerCardsValue(const CoupState& state, int player, 
                                    bool priv, bool pub,
                                    Allocator* allocator) {
    auto out = allocator->Get("p" + std::to_string(player+1) + "_cards",
                              {kMaxCardsInHand, kNumCardTypes});
    for (int i = 0; i < state.players_.at(player).cards.size(); ++i) {
      const CoupCard& card = state.players_.at(player).cards.at(i);
      if (card.value != CardType::kNone &&
          ((priv && card.state == CardStateType::kFaceDown) ||
          (pub && card.state == CardStateType::kFaceUp))) {
        out.at(i, (int)card.value) = 1;
      }
    }
  }

  // Card state (non-existent, face down, face up). Always public for all players.
  static void WriteCardsState(const CoupState& state,
                              Allocator* allocator) {
    auto out = allocator->Get("cards_state",
                              {state.num_players_, kMaxCardsInHand, 2});
    for (int p = 0; p < state.num_players_; ++p) {
      for (int i = 0; i < state.players_.at(p).cards.size(); ++i) {
        const CardStateType& cs = state.players_.at(p).cards.at(i).state;
        if (cs != CardStateType::kNone) out.at(p, i, (int)cs) = 1;
      }
    }
  }

  // Coins for each player. Public.
  static void WriteCoins(const CoupState& state,
                         Allocator* allocator) {
    auto out = allocator->Get("coins", {state.num_players_});
    for (int p = 0; p < state.num_players_; ++p) {
      out.at(p) = state.players_.at(p).coins;
    }
  }

  // Last action for each player. Public.
  // Don't call if perfect recall (will already get the full history)
  static void WriteLastAction(const CoupState& state,
                              Allocator* allocator) {
    auto out = allocator->Get("last_action", {state.num_players_, 
                                              state.num_distinct_actions_});
    for (int p = 0; p < state.num_players_; ++p) {
      const ActionType& a = state.players_.at(p).last_action;
      if (a != ActionType::kNone) out.at(p, (int)a) = 1;
    }
  }

  // Complete action history, except for chance deals to opponent (private).
  // Need all info for perfect recall. Since we don't store old infostates,
  // need all actions to be able to determine what previous infostates were.
  static void WriteActionHistory(const CoupState& state, int player,
                                  Allocator* allocator) {
    auto& game = open_spiel::down_cast<const CoupGame&>(*state.GetGame());

    auto out = allocator->Get("history",
                              {game.MaxMoveNumber(), game.NumDistinctActions()});

    for (int i = 0; i < state.history_.size(); ++i) {
      int p = (int)state.history_.at(i).player;
      if (p >= 0 || (p == kChancePlayerId &&
          state.history_chance_deal_player_.at(i) == player)) {
        const int& a = (int)state.history_.at(i).action;
        if ((ActionType)a != ActionType::kNone) out.at(i, a) = 1;
      }
    }
  }

  // Write the complete observation as tensor
  void WriteTensor(const State& observed_state, int player,
                   Allocator* allocator) const override {
    auto& state = open_spiel::down_cast<const CoupState&>(observed_state);
    SPIEL_CHECK_GE(player, 0);
    SPIEL_CHECK_LT(player, state.num_players_);

    // Observing player
    WritePlayer(state, player, allocator, "");

    // Card value
    bool priv;
    bool pub = iig_obs_type_.public_info;
    for (int p = 0; p < state.num_players_; ++p) {
      priv = iig_obs_type_.private_info == PrivateInfoType::kAllPlayers ||
             (iig_obs_type_.private_info == PrivateInfoType::kSinglePlayer &&
              p == player);
      WritePlayerCardsValue(state, p, priv, pub, allocator);
    }

    // Public information
    if (iig_obs_type_.public_info) {
      if (state.IsTerminal()) {
        // No one's move
        // Leave as all 0
        auto out = allocator->Get("cur_move_player", {state.num_players_});
      } else {
        // Current move player
        WritePlayer(state, state.cur_player_move_, allocator, "cur_move_");
      }

      WriteCardsState(state, allocator);
      WriteCoins(state, allocator);

      if (iig_obs_type_.perfect_recall) {
        WriteActionHistory(state, player, allocator);
      } else {
        WriteLastAction(state, allocator);
      }
    }
  }

  // Write the observation as string, human-readable
  std::string StringFrom(const State& observed_state,
                         int player) const override {
    auto& state = open_spiel::down_cast<const CoupState&>(observed_state);
    SPIEL_CHECK_GE(player, 0);
    SPIEL_CHECK_LT(player, state.num_players_);
    std::string result;

    absl::StrAppend(&result, "Observer: P", player+1, "\n");
    if (iig_obs_type_.public_info) {
      absl::StrAppend(&result, "Turn: ", state.turn_number_, "\n");
      absl::StrAppend(&result, "Move: P", state.cur_player_move_+1, "\n");
    }

    for (int p = 0; p < state.num_players_; ++p) {
      if (iig_obs_type_.public_info || 
          iig_obs_type_.private_info == PrivateInfoType::kAllPlayers ||
          (iig_obs_type_.private_info == PrivateInfoType::kSinglePlayer &&
              player == p)) {
        absl::StrAppend(&result, "P", p+1, "\n");
        absl::StrAppend(&result, "        Card         State\n");

        for (int c = 0; c < state.players_.at(p).cards.size(); ++c) {
          absl::StrAppend(&result, "Card ", c+1, ": ");

          const CoupCard& coupCard = state.players_.at(p).cards.at(c);
          std::string cardVal;
          if ((iig_obs_type_.public_info && coupCard.state == CardStateType::kFaceUp) ||
              (iig_obs_type_.private_info == PrivateInfoType::kSinglePlayer &&
                  p == player &&
                  coupCard.state == CardStateType::kFaceDown) ||
              (iig_obs_type_.private_info == PrivateInfoType::kAllPlayers &&
                  coupCard.state == CardStateType::kFaceDown)) {
            // Show card value
            cardVal = StatelessCardToString(coupCard.value);
          } else {
            // Hidden card
            cardVal = StatelessCardToString(CardType::kNone);
          }
          std::string space(11-cardVal.length(), ' ');
          absl::StrAppend(&result, cardVal, space, "| ");
          
          std::string cardState;
          if (iig_obs_type_.public_info) {
            cardState = StatelessCardStateToString(coupCard.state);
          } else {
            cardState = StatelessCardStateToString(CardStateType::kNone);
          }
          absl::StrAppend(&result, cardState, "\n");
        }
      }

      if (iig_obs_type_.public_info) {
        absl::StrAppend(&result, "Coins: ", state.players_.at(p).coins, "\n");

        if (!iig_obs_type_.perfect_recall) {
          absl::StrAppend(&result, "Last Action: ", 
            StatelessActionToString(state.players_.at(p).last_action), "\n\n");
        } else { absl::StrAppend(&result, "\n"); }
      }
    }

    if (iig_obs_type_.public_info && iig_obs_type_.perfect_recall) {
      absl::StrAppend(&result, "Action Sequence: ");
      for (int i = 0; i < state.history_.size(); ++i) {
        auto& pa = state.history_.at(i);
        if (pa.player == kChancePlayerId) {
          if (state.history_chance_deal_player_.at(i) == player) {
            // Only show card deals if it is for observing player
            absl::StrAppend(&result, "PC-");
            absl::StrAppend(&result, StatelessCardToString((CardType)pa.action));
            if (i < state.history_.size()-1) 
              absl::StrAppend(&result, ", ");
          }
        } else {
          absl::StrAppend(&result, "P", pa.player+1, "-");
          absl::StrAppend(&result, StatelessActionToString((ActionType)pa.action));
          if (i < state.history_.size()-1) 
            absl::StrAppend(&result, ", ");
        }
      }
      absl::StrAppend(&result, "\n");
    }
    return result;
  }

 private:
  IIGObservationType iig_obs_type_;
};

bool CoupPlayer::HasFaceDownCard(CardType card) {
  for (auto &c: cards) {
    if (c.value == card &&
        c.state == CardStateType::kFaceDown) {
      return true;
    }
  }
  return false;
}

void CoupPlayer::SortCards() {
  std::sort(cards.begin(), cards.end());
}

CoupState::CoupState(std::shared_ptr<const Game> game)
    : State(game),
      deck_(kNumCardTypes, kNumEachCardInDeck),
      cur_player_turn_(0),
      cur_player_move_(0),
      opp_player_(1),
      is_turn_begin_(true),
      turn_number_(0),
      is_chance_(true),
      cur_rewards_(num_players_, 0) {
  // Start game
  // Chance node first to deal cards

  // Create 2 players
  players_ = {
    {
      {},//cards
      1,//coins
      ActionType::kNone,//last_action
      false//lost_challenge
    },
    {
      {},//cards
      2,//coins
      ActionType::kNone,//last_action
      false//lost_challenge
    }
  };

  // Queue players to deal cards to
  // These chance nodes will be hit before it is P1's turn
  deal_card_to_.push(0);
  deal_card_to_.push(1);
  deal_card_to_.push(0);
  deal_card_to_.push(1);
}

std::vector<CardType> CoupState::GetCardsValue(Player player) const {
  SPIEL_CHECK_LT(player, NumPlayers());
  std::vector<CardType> cards;
  for (auto &c: players_.at(player).cards) {
    cards.push_back(c.value);
  }
  return cards;
}

std::vector<CardStateType> CoupState::GetCardsState(Player player) const {
  SPIEL_CHECK_LT(player, NumPlayers());
  std::vector<CardStateType> cards;
  for (auto &c: players_.at(player).cards) {
    cards.push_back(c.state);
  }
  return cards;
}

int CoupState::GetCoins(Player player) const {
  SPIEL_CHECK_LT(player, NumPlayers());
  return players_.at(player).coins;
}

Action CoupState::GetLastAction(Player player) const {
  SPIEL_CHECK_LT(player, NumPlayers());
  return (Action)players_.at(player).last_action;
}

Player CoupState::CurrentPlayer() const {
  if (IsTerminal()) {
    return kTerminalPlayerId;
  } else if(is_chance_) {
    return kChancePlayerId;
  } else {
    return cur_player_move_;
  }
}

void CoupState::ChallengeFailReplaceCard(CardType card) {
  CoupPlayer &op = players_.at(opp_player_);
  // Find the face down card to replace
  for (int i = 0; i < op.cards.size(); ++i) {
    CoupCard &c = op.cards.at(i);
    if (c.value == card &&
        c.state == CardStateType::kFaceDown) {
      // Remove from hand and put back in deck
      deck_.at((int)card) += 1;
      op.cards.erase(op.cards.begin()+i);
      // Chance player will deal a random card
      // to replace the card that was just revealed
      deal_card_to_.push(opp_player_);
      is_chance_ = true;
      return;
    }
  }
  SpielFatalError("Error: Tried to replace card which was not found in hand");
}

// In a chance node, `move` should be the card to deal to the player
// On a player node, it should be ActionType
void CoupState::DoApplyAction(Action move) {
  if (IsChanceNode()) {
    SPIEL_CHECK_GE(move, 0);
    SPIEL_CHECK_LT(move, deck_.size());
    SPIEL_CHECK_GT(deck_.at(move), 0);
    SPIEL_CHECK_GT(deal_card_to_.size(), 0);

    // All chance nodes have same action
    // Player gets random card from deck

    int dealToPlayer = deal_card_to_.front();
    deal_card_to_.pop();

    // Track which player is being dealt to
    history_chance_deal_player_.insert({history_.size(), dealToPlayer});

    // Deal the card
    deck_.at(move) -= 1;
    players_.at(dealToPlayer).cards.push_back(
        CoupCard{(CardType)move, CardStateType::kFaceDown});
    players_.at(dealToPlayer).SortCards();

    // 3 situations when chance node hit:
    // - Beginning of game dealing cards
    // - ChallengeFailReplaceCard
    // - Exchange (draw 2 cards from deck)
    //
    // In all cases, it's not the end of the turn,
    // and cur_player_move_ has next move,
    // so don't increment player. Just leave chance node.
    if (deal_card_to_.size() == 0) is_chance_ = false;

  } else {
    CoupPlayer &cp = players_.at(cur_player_move_);
    CoupPlayer &op = players_.at(opp_player_);

    // Reset reward
    cur_rewards_ = {0, 0};

    ActionType action = (ActionType)move;

    if (action == ActionType::kIncome) {
      cp.last_action = action;
      cp.coins += 1;
      NextPlayerTurn();

    } else if (action == ActionType::kForeignAid) {
      if (is_turn_begin_) {
        // Before allowing the action to take effect,
        // the opponent must not block it
        cp.last_action = action;
        NextPlayerMove();
      } else {
        // PASS: Opponent did not block, so complete the action
        cp.coins += 2;
        NextPlayerTurn();
      }

    } else if (action == ActionType::kCoup) {
      SPIEL_CHECK_GE(cp.coins, 7);

      cp.last_action = action;
      cp.coins -= 7;
      NextPlayerMove();

    } else if (action == ActionType::kTax) {
      if (is_turn_begin_) {
        // Before allowing the action to take effect,
        // the opponent must not challenge it
        cp.last_action = action;
        NextPlayerMove();
      } else {
        // PASS: Opponent did not challenge, so complete the action
        cp.coins += 3;
        NextPlayerTurn();
      }

    } else if (action == ActionType::kAssassinate) {
      SPIEL_CHECK_GE(cp.coins, 3);

      cp.last_action = action;
      // Pay the coins whether or not the action is blocked/challenged
      cp.coins -= 3;
      NextPlayerMove();

    } else if (action == ActionType::kExchange) {
      if (is_turn_begin_) {
        // Before allowing the action to take effect,
        // the opponent must not challenge
        cp.last_action = action;
        NextPlayerMove();
      } else {
        // Draw 2 cards (in next chance nodes)
        deal_card_to_.push(cur_player_move_);
        deal_card_to_.push(cur_player_move_);
        is_chance_ = true;
        // Don't increment player. Still their move.
      }

    } else if (action == ActionType::kSteal) {
      SPIEL_CHECK_GE(op.coins, 1);

      if (is_turn_begin_) {
        // Before allowing the action to take effect,
        // the opponent must not block/challenge
        cp.last_action = action;
        NextPlayerMove();
      } else {
        // PASS: Opponent did not block/challenge, so complete the action
        int numSteal = (op.coins > 1) ? 2 : 1;
        cp.coins += numSteal;
        op.coins -= numSteal;
        NextPlayerTurn();
      }

    } else if (action == ActionType::kLoseCard1 ||
               action == ActionType::kLoseCard2) {
      int cardToLose = move - (int)ActionType::kLoseCard1;
      SPIEL_CHECK_EQ((int)cp.cards.at(cardToLose).state, (int)CardStateType::kFaceDown);

      cp.last_action = action;
      cp.cards.at(cardToLose).state = CardStateType::kFaceUp;
      cp.lost_challenge = false;
      cp.SortCards();
      cur_rewards_.at(cur_player_move_) -= 1;
      cur_rewards_.at(opp_player_) += 1;
      NextPlayerTurn();

    } else if (action == ActionType::kPass) {
      cp.last_action = action;
      ActionType n_act = op.last_action;

      if (n_act == ActionType::kBlock) {
        // Block succeeds. Nothing to do.
        NextPlayerTurn();
      } else {
        NextPlayerMove();
        // Pass, so complete their action
        DoApplyAction((Action)n_act);
      }

    } else if (action == ActionType::kBlock) {
      cp.last_action = action;
      NextPlayerMove();

    } else if (action == ActionType::kChallenge) {
      if (op.last_action == ActionType::kBlock) {
        if (cp.last_action == ActionType::kForeignAid) {
          cp.last_action = action;
          if (op.HasFaceDownCard(CardType::kDuke)) {
            cp.lost_challenge = true;
            ChallengeFailReplaceCard(CardType::kDuke);
            // cp must lose a card. Still their move
          } else {
            op.lost_challenge = true;
            // Block failed, so complete the action
            cp.coins += 2;
            // opp must lose a card
            NextPlayerMove();
          }
        } else if (cp.last_action == ActionType::kAssassinate) {
          cp.last_action = action;
          if (op.HasFaceDownCard(CardType::kContessa)) {
            cp.lost_challenge = true;
            ChallengeFailReplaceCard(CardType::kContessa);
            // cp must lose a card. Still their move
          } else {
            // op loses game. Lose 1 for assassination
            // and 1 for losing challenge
            // (if they werent already eliminated)
            if (op.cards.at(0).state == CardStateType::kFaceDown) {
              op.cards.at(0).state = CardStateType::kFaceUp;
              cur_rewards_.at(cur_player_move_) += 1;
              cur_rewards_.at(opp_player_) -= 1;
            }
            if (op.cards.at(1).state == CardStateType::kFaceDown) {
              op.cards.at(1).state = CardStateType::kFaceUp;
              cur_rewards_.at(cur_player_move_) += 1;
              cur_rewards_.at(opp_player_) -= 1;
            }
          }
        } else if (cp.last_action == ActionType::kSteal) {
          cp.last_action = action;
          if (op.HasFaceDownCard(CardType::kCaptain)) {
            cp.lost_challenge = true;
            ChallengeFailReplaceCard(CardType::kCaptain);
            // cp must lose a card. Still their move
          } else if (op.HasFaceDownCard(CardType::kAmbassador)) {
            cp.lost_challenge = true;
            ChallengeFailReplaceCard(CardType::kAmbassador);
            // cp must lose a card. Still their move
          } else {
            op.lost_challenge = true;

            // Block failed. Complete the steal
            int numSteal = (op.coins > 1) ? 2 : 1;
            cp.coins += numSteal;
            op.coins -= numSteal;
            // opp must lose a card
            NextPlayerMove();
          }
        } else {
          SpielFatalError("Invalid player action");
        }
      } else if (op.last_action == ActionType::kTax) {
        cp.last_action = action;
        if (op.HasFaceDownCard(CardType::kDuke)) {
          cp.lost_challenge = true;
          ChallengeFailReplaceCard(CardType::kDuke);
          // Complete the action
          op.coins += 3;
          // cp must lose a card. Still their move
        } else {
          op.lost_challenge = true;
          // opp must lose a card
          NextPlayerMove();
        }

      } else if (op.last_action == ActionType::kExchange) {
        cp.last_action = action;
        if (op.HasFaceDownCard(CardType::kAmbassador)) {
          cp.lost_challenge = true;
          ChallengeFailReplaceCard(CardType::kAmbassador);
          // Turn off is_chance_ momentarily, so that Exchange below works.
          // Exchange will turn it back on anyway.
          is_chance_ = false;
          // Complete the action
          NextPlayerMove();
          DoApplyAction((Action)ActionType::kExchange);
          // cp must lose a card
          // After exchange return, it will switch to their action
        } else {
          op.lost_challenge = true;
          // opp must lose a card
          NextPlayerMove();
        }

      } else if (op.last_action == ActionType::kAssassinate) {
        cp.last_action = action;
        if (op.HasFaceDownCard(CardType::kAssassin)) {
          // cp loses game. Lose 1 for assassination
          // and 1 for losing challenge
          // (if they werent already eliminated)
          if (cp.cards.at(0).state == CardStateType::kFaceDown) {
            cp.cards.at(0).state = CardStateType::kFaceUp;
            cur_rewards_.at(cur_player_move_) -= 1;
            cur_rewards_.at(opp_player_) += 1;
          }
          if (cp.cards.at(1).state == CardStateType::kFaceDown) {
            cp.cards.at(1).state = CardStateType::kFaceUp;
            cur_rewards_.at(cur_player_move_) -= 1;
            cur_rewards_.at(opp_player_) += 1;
          }
        } else {
          op.lost_challenge = true;
          // Coins spent are returned in this one case
          op.coins += 3;
          // opp must lose a card
          NextPlayerMove();
        }

      } else if (op.last_action == ActionType::kSteal) {
        cp.last_action = action;
        if (op.HasFaceDownCard(CardType::kCaptain)) {
          cp.lost_challenge = true;
          ChallengeFailReplaceCard(CardType::kCaptain);

          // Complete the action
          int numSteal = (cp.coins > 1) ? 2 : 1;
          op.coins += numSteal;
          cp.coins -= numSteal;

          // cp must lose a card. Still their move
        } else {
          op.lost_challenge = true;
          // opp must lose a card
          NextPlayerMove();
        }

      } else {
        SpielFatalError("Invalid player action");
      }

    } else if (move >= (int)ActionType::kExchangeReturn12 &&
               move <= (int)ActionType::kExchangeReturn34) {
      cp.last_action = action;
      std::vector<int> cardInd;
      if (move <= (int)ActionType::kExchangeReturn14) cardInd.push_back(0);
      if (action == ActionType::kExchangeReturn12 ||
          action == ActionType::kExchangeReturn23 ||
          action == ActionType::kExchangeReturn24) cardInd.push_back(1);
      if (action == ActionType::kExchangeReturn13 ||
          action == ActionType::kExchangeReturn23 ||
          action == ActionType::kExchangeReturn34) cardInd.push_back(2);
      if (action == ActionType::kExchangeReturn14 ||
          action == ActionType::kExchangeReturn24 ||
          action == ActionType::kExchangeReturn34) cardInd.push_back(3);
      SPIEL_CHECK_EQ(cardInd.size(), 2);

      int c;
      for (int i = 1; i >= 0; --i) {
        c = cardInd.at(i);
        // Remove from hand and put back in deck
        cp.cards.erase(cp.cards.begin()+c);
        deck_.at(c) += 1;
      }
      SPIEL_CHECK_EQ(cp.cards.size(), 2);

      if (op.lost_challenge) {
        // op needs to lose a card
        NextPlayerMove();
      } else {
        NextPlayerTurn();
      }

    } else {
      SpielFatalError("Invalid player action");
    }
  }
}

std::vector<Action> CoupState::LegalLoseCardActions() const {
  // Show which cards the player can lose
  std::vector<Action> legal;
  const CoupPlayer &p = players_.at(cur_player_move_);
  if (p.cards.at(0).state == CardStateType::kFaceDown) {
    legal.push_back((Action)ActionType::kLoseCard1);
  }
  if (p.cards.at(1).state == CardStateType::kFaceDown) {
    legal.push_back((Action)ActionType::kLoseCard2);
  }
  return legal;
}

std::vector<Action> CoupState::LegalActions() const {
  if (IsTerminal()) return {};
  std::vector<Action> legal;

  if (IsChanceNode()) {
    // All chance nodes are the action of drawing a card
    // Only show cards in the deck
    for (int i = 0; i < deck_.size(); ++i) {
      if (deck_.at(i) > 0)
        legal.push_back(i);
    }
    return legal;
  }
  
  // else Decision node
  const CoupPlayer &cp = players_.at(cur_player_move_);
  const CoupPlayer &op = players_.at(opp_player_);
  if (is_turn_begin_) {
    if (cp.coins >= 10) {
      legal.push_back((Action)ActionType::kCoup);
      return legal;
    }

    legal.push_back((Action)ActionType::kIncome);
    legal.push_back((Action)ActionType::kForeignAid);
    if (cp.coins >= 7) legal.push_back((Action)ActionType::kCoup);
    legal.push_back((Action)ActionType::kTax);
    if (cp.coins >= 3) legal.push_back((Action)ActionType::kAssassinate);
    legal.push_back((Action)ActionType::kExchange);
    if (op.coins > 0) legal.push_back((Action)ActionType::kSteal);
    return legal;

  } else if (cp.lost_challenge) {
    // Player lost challenge and needs to lose a card
    return LegalLoseCardActions();

  } else if (cur_player_move_ != cur_player_turn_) {
    // opponent's turn, so cur_player_move_ can
    // choose to block or challenge for certain actions
    if (op.last_action == ActionType::kForeignAid) {
      legal = {(Action)ActionType::kPass, 
               (Action)ActionType::kBlock};
      return legal;
    } else if (op.last_action == ActionType::kTax ||
               op.last_action == ActionType::kExchange) {
      legal = {(Action)ActionType::kPass, 
               (Action)ActionType::kChallenge};
      return legal;
    } else if (op.last_action == ActionType::kSteal) {
      legal = {(Action)ActionType::kPass, 
               (Action)ActionType::kBlock,
               (Action)ActionType::kChallenge};
      return legal;
    } else if (op.last_action == ActionType::kAssassinate) {
      legal = LegalLoseCardActions();
      legal.push_back((Action)ActionType::kBlock);
      legal.push_back((Action)ActionType::kChallenge);
      return legal;
    } else if (op.last_action == ActionType::kCoup) {
      legal = LegalLoseCardActions();
      return legal;
    } else {
      SpielFatalError("Error in LegalActions(): Invalid action progression in cur_player_move_ != cur_player_turn_");
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
      legal = {(Action)ActionType::kExchangeReturn12,
               (Action)ActionType::kExchangeReturn13,
               (Action)ActionType::kExchangeReturn14,
               (Action)ActionType::kExchangeReturn23,
               (Action)ActionType::kExchangeReturn24,
               (Action)ActionType::kExchangeReturn34};
    } else if (faceUpInd == 0) {
      legal = {(Action)ActionType::kExchangeReturn23,
               (Action)ActionType::kExchangeReturn24,
               (Action)ActionType::kExchangeReturn34};
    } else if (faceUpInd == 1) {
      legal = {(Action)ActionType::kExchangeReturn13,
               (Action)ActionType::kExchangeReturn14,
               (Action)ActionType::kExchangeReturn34};
    } else if (faceUpInd == 2) {
      legal = {(Action)ActionType::kExchangeReturn12,
               (Action)ActionType::kExchangeReturn14,
               (Action)ActionType::kExchangeReturn24};
    } else if (faceUpInd == 3) {
      legal = {(Action)ActionType::kExchangeReturn12,
               (Action)ActionType::kExchangeReturn13,
               (Action)ActionType::kExchangeReturn23};
    }
    return legal;

  } else if (op.last_action == ActionType::kBlock) {
    legal = {(Action)ActionType::kPass,
             (Action)ActionType::kChallenge};
    return legal;

  } else {
    SpielFatalError("Error in LegalActions(): Invalid action progression");
  }
}

std::string CoupState::ActionToString(Player player, Action move) const {
  return GetGame()->ActionToString(player, move);
}

// Complete observation including all private info
std::string CoupState::ToString() const {
  std::string result;

  absl::StrAppend(&result, "Turn: ", turn_number_, "\n");
  absl::StrAppend(&result, "Move: P", cur_player_move_+1, "\n");

  for (int p = 0; p < num_players_; ++p) {
    absl::StrAppend(&result, "P", p+1, "\n");
    absl::StrAppend(&result, "        Card         State\n");

    for (int c = 0; c < players_.at(p).cards.size(); ++c) {
      absl::StrAppend(&result, "Card ", c+1, ": ");

      const CoupCard& coupCard = players_.at(p).cards.at(c);
      std::string cardVal = StatelessCardToString(coupCard.value);
      std::string space(11-cardVal.length(), ' ');
      absl::StrAppend(&result, cardVal, space, "| ");
      
      std::string cardState = StatelessCardStateToString(coupCard.state);
      absl::StrAppend(&result, cardState, "\n");
    }
    absl::StrAppend(&result, "Coins: ", players_.at(p).coins, "\n");
    absl::StrAppend(&result, "Last Action: ", 
      StatelessActionToString(players_.at(p).last_action), "\n\n");
  }
  absl::StrAppend(&result, "Action Sequence: ");
  for (int i = 0; i < history_.size(); ++i) {
    auto& pa = history_.at(i);
    if (pa.player == kChancePlayerId) {
      absl::StrAppend(&result, "PC-");
      absl::StrAppend(&result, StatelessCardToString((CardType)pa.action));
      if (i < history_.size()-1) 
        absl::StrAppend(&result, ", ");
    } else {
      absl::StrAppend(&result, "P", pa.player+1, "-");
      absl::StrAppend(&result, StatelessActionToString((ActionType)pa.action));
      if (i < history_.size()-1) 
        absl::StrAppend(&result, ", ");
    }
  }
  absl::StrAppend(&result, "\n");
  return result;
}

bool CoupState::IsTerminal() const {
  if (move_number_ > game_->MaxGameLength()) {
    return true;
  }
  int numPlayersAlive = 0;
  for (auto &p: players_) {
    if (p.cards.size() < 2) {
      // p is being dealt a card, so still alive
      numPlayersAlive += 1;
      continue;
    }
    
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

std::vector<double> CoupState::Rewards() const {
  return cur_rewards_;
}

std::vector<double> CoupState::Returns() const {
  std::vector<double> returns(num_players_, 0);

  // Get count of face up cards
  std::vector<int> faceUp(num_players_, 0);
  for (int i = 0; i < num_players_; ++i) {
    for (auto &c: players_.at(i).cards) {
      if (c.state == CardStateType::kFaceUp) {
        faceUp.at(i) += 1;
      }
    }
  }
  // + reward for opp losing cards, - reward for you losing cards
  returns.at(0) = faceUp.at(1) - faceUp.at(0);
  returns.at(1) = faceUp.at(0) - faceUp.at(1);
  return returns;
}

std::string CoupState::InformationStateString(Player player) const {
  const CoupGame& game = open_spiel::down_cast<const CoupGame&>(*game_);
  return game.info_state_observer_->StringFrom(*this, player);
}

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

  // Num cards in deck
  double deckSize = std::reduce(deck_.begin(), deck_.end());

  double p;
  for (int i = 0; i < deck_.size(); ++i) {
    if (deck_.at(i) > 0) {
      p = deck_.at(i) / deckSize;
      outcomes.push_back({i, p});
    }
  }
  return outcomes;
}

void CoupState::NextPlayerTurn() {
  cur_player_turn_ = 1 - cur_player_turn_;
  // Player always has first move on their turn
  cur_player_move_ = cur_player_turn_;
  opp_player_ = 1 - cur_player_move_;
  ++turn_number_;
  is_turn_begin_ = true;
}

void CoupState::NextPlayerMove() {
  cur_player_move_ = 1 - cur_player_move_;
  opp_player_ = 1 - cur_player_move_;
  is_turn_begin_ = false;
}

CoupGame::CoupGame(const GameParameters& params)
    : Game(kGameType, params) {
  default_observer_ = std::make_shared<CoupObserver>(kDefaultObsType);
  info_state_observer_ = std::make_shared<CoupObserver>(kInfoStateObsType);
}

std::unique_ptr<State> CoupGame::NewInitialState() const {
  return absl::make_unique<CoupState>(shared_from_this());
}

std::vector<int> CoupGame::InformationStateTensorShape() const {
  // Tensor contents (all one-hot):
  // Observing player [NumPlayers]
  // cur_player_move_ [NumPlayers]
  // Card values      [NumPlayers, MaxCardsInHand, NumCardTypes]
  // Card states      [NumPlayers, MaxCardsInHand, 2]
  // Coins            [NumPlayers]
  // Action Sequence  [MaxMoveNum, NumDistinctActions]

  // Card values are hidden if private to opponent
  return {NumPlayers() * (3 + kMaxCardsInHand * (kNumCardTypes + 2))
          + MaxMoveNumber() * NumDistinctActions()};
}

std::vector<int> CoupGame::ObservationTensorShape() const {
  // Tensor contents (all one-hot):
  // Observing player [NumPlayers]
  // cur_player_move_ [NumPlayers]
  // Card values      [NumPlayers, MaxCardsInHand, NumCardTypes]
  // Card states      [NumPlayers, MaxCardsInHand, 2]
  // Coins            [NumPlayers]
  // Last Action      [NumPlayers, NumDistinctActions]

  // Card values are hidden if private to opponent
  return {NumPlayers() * (3 + kMaxCardsInHand * (kNumCardTypes + 2)
          + NumDistinctActions())};
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
    return absl::StrCat("Chance drawn card:", StatelessCardToString((CardType)action));
  } else {
    return StatelessActionToString((ActionType)action);
  }
}
}  // namespace coup
}  // namespace open_spiel
