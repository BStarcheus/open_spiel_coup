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
#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace coup {
namespace {

namespace testing = open_spiel::testing;

void BasicCoupTests() {
  testing::LoadGameTest("coup");
  testing::ChanceOutcomesTest(*LoadGame("coup"));
  testing::RandomSimTest(*LoadGame("coup"), 100);
  auto observer = LoadGame("coup")
                      ->MakeObserver(kDefaultObsType,
                                     GameParametersFromString("single_tensor"));
  testing::RandomSimTestCustomObserver(*LoadGame("coup"), observer);
  observer = LoadGame("coup")
                      ->MakeObserver(kInfoStateObsType,
                                     GameParametersFromString("single_tensor"));
  testing::RandomSimTestCustomObserver(*LoadGame("coup"), observer);
}

// General game and action tests
void CoupGameStartTest() {
  std::shared_ptr<const Game> game = LoadGame("coup");
  std::unique_ptr<State> tempState = game->NewInitialState();
  CoupState* state = static_cast<CoupState*>(tempState.get());

  SPIEL_CHECK_EQ(state->CurrentPlayer(), kChancePlayerId);
  // Deal cards
  state->ApplyAction((Action)CardType::kAmbassador);
  state->ApplyAction((Action)CardType::kAssassin);
  state->ApplyAction((Action)CardType::kContessa);
  state->ApplyAction((Action)CardType::kDuke);

  // Card values and states
  std::vector<CardType> cards;
  std::vector<CardStateType> cardStates;
  for (int p = 0; p < state->NumPlayers(); ++p) {
    cards = state->GetCardsValue(p);
    cardStates = state->GetCardsState(p);
    SPIEL_CHECK_EQ(cards.size(), 2);
    for (int c = 0; c < cards.size(); ++c) {
      SPIEL_CHECK_GE((int)cards.at(c), -1);
      SPIEL_CHECK_EQ((int)cardStates.at(c), (int)CardStateType::kFaceDown);
    }
  }
  // Coins
  SPIEL_CHECK_EQ(state->GetCoins(0), 1);
  SPIEL_CHECK_EQ(state->GetCoins(1), 2);
  // No last action
  SPIEL_CHECK_EQ((int)state->GetLastAction(0), (int)ActionType::kNone);
  SPIEL_CHECK_EQ((int)state->GetLastAction(1), (int)ActionType::kNone);
  // P1 first move
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 0);
}

void CoupIncomeTest() {
  std::shared_ptr<const Game> game = LoadGame("coup");
  std::unique_ptr<State> tempState = game->NewInitialState();
  CoupState* state = static_cast<CoupState*>(tempState.get());

  // Deal cards
  state->ApplyAction((Action)CardType::kAmbassador);
  state->ApplyAction((Action)CardType::kAssassin);
  state->ApplyAction((Action)CardType::kContessa);
  state->ApplyAction((Action)CardType::kDuke);

  state->ApplyAction((Action)ActionType::kIncome);

  SPIEL_CHECK_EQ(state->GetCoins(0), 2);
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 1);
  SPIEL_CHECK_FALSE(state->IsTerminal());
  std::vector<double> rewards = state->Rewards();
  std::vector<double> returns = state->Returns();
  for (int p = 0; p < state->NumPlayers(); ++p) {
    SPIEL_CHECK_EQ(rewards.at(p), 0);
    SPIEL_CHECK_EQ(returns.at(p), 0);
  }
}

void CoupPassForeignAidTest() {
  std::shared_ptr<const Game> game = LoadGame("coup");
  std::unique_ptr<State> tempState = game->NewInitialState();
  CoupState* state = static_cast<CoupState*>(tempState.get());

  // Deal cards
  state->ApplyAction((Action)CardType::kAmbassador);
  state->ApplyAction((Action)CardType::kAssassin);
  state->ApplyAction((Action)CardType::kContessa);
  state->ApplyAction((Action)CardType::kDuke);

  state->ApplyAction((Action)ActionType::kForeignAid);
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 1);
  std::vector<Action> legal = state->LegalActions();
  std::vector<Action> correct = {(Action)ActionType::kPass,
                                 (Action)ActionType::kBlock};
  SPIEL_CHECK_TRUE(legal == correct);
  state->ApplyAction((Action)ActionType::kPass);

  SPIEL_CHECK_EQ(state->GetCoins(0), 3);
  SPIEL_CHECK_FALSE(state->IsTerminal());
  std::vector<double> rewards = state->Rewards();
  std::vector<double> returns = state->Returns();
  for (int p = 0; p < state->NumPlayers(); ++p) {
    SPIEL_CHECK_EQ(rewards.at(p), 0);
    SPIEL_CHECK_EQ(returns.at(p), 0);
  }
}

void CoupBlockForeignAidTest() {
  std::shared_ptr<const Game> game = LoadGame("coup");
  std::unique_ptr<State> tempState = game->NewInitialState();
  CoupState* state = static_cast<CoupState*>(tempState.get());

  // Deal cards
  state->ApplyAction((Action)CardType::kAmbassador);
  state->ApplyAction((Action)CardType::kAssassin);
  state->ApplyAction((Action)CardType::kContessa);
  state->ApplyAction((Action)CardType::kDuke);

  state->ApplyAction((Action)ActionType::kForeignAid);
  state->ApplyAction((Action)ActionType::kBlock);
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 0);
  std::vector<Action> legal = state->LegalActions();
  std::vector<Action> correct = {(Action)ActionType::kPass,
                                 (Action)ActionType::kChallenge};
  SPIEL_CHECK_TRUE(legal == correct);
  state->ApplyAction((Action)ActionType::kPass);

  SPIEL_CHECK_EQ(state->GetCoins(0), 1);
  SPIEL_CHECK_FALSE(state->IsTerminal());
  std::vector<double> rewards = state->Rewards();
  std::vector<double> returns = state->Returns();
  for (int p = 0; p < state->NumPlayers(); ++p) {
    SPIEL_CHECK_EQ(rewards.at(p), 0);
    SPIEL_CHECK_EQ(returns.at(p), 0);
  }
}

void CoupChallengeForeignAidTest() {
  std::shared_ptr<const Game> game = LoadGame("coup");
  std::unique_ptr<State> tempState = game->NewInitialState();
  CoupState* state = static_cast<CoupState*>(tempState.get());

  // Deal cards
  state->ApplyAction((Action)CardType::kDuke);
  state->ApplyAction((Action)CardType::kAssassin);
  state->ApplyAction((Action)CardType::kContessa);
  state->ApplyAction((Action)CardType::kAmbassador);

  state->ApplyAction((Action)ActionType::kForeignAid);
  state->ApplyAction((Action)ActionType::kBlock);
  state->ApplyAction((Action)ActionType::kChallenge);

  // P2 didn't have a Duke, so must lose card
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 1);
  std::vector<Action> legal = state->LegalActions();
  std::vector<Action> correct = {(Action)ActionType::kLoseCard1,
                                 (Action)ActionType::kLoseCard2};
  SPIEL_CHECK_TRUE(legal == correct);

  SPIEL_CHECK_EQ(state->GetCoins(0), 3);
  SPIEL_CHECK_FALSE(state->IsTerminal());
  std::vector<double> rewards = state->Rewards();
  std::vector<double> returns = state->Returns();
  for (int p = 0; p < state->NumPlayers(); ++p) {
    SPIEL_CHECK_EQ(rewards.at(p), 0);
    SPIEL_CHECK_EQ(returns.at(p), 0);
  }
}

void CoupLoseCardTest() {
  std::shared_ptr<const Game> game = LoadGame("coup");
  std::unique_ptr<State> tempState = game->NewInitialState();
  CoupState* state = static_cast<CoupState*>(tempState.get());

  // Deal cards
  state->ApplyAction((Action)CardType::kAmbassador);
  state->ApplyAction((Action)CardType::kAssassin);
  state->ApplyAction((Action)CardType::kContessa);
  state->ApplyAction((Action)CardType::kDuke);

  for (int i = 0; i < 11; ++i)
    state->ApplyAction((Action)ActionType::kIncome);
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 1);
  SPIEL_CHECK_EQ(state->GetCoins(1), 7);

  state->ApplyAction((Action)ActionType::kCoup);
  
  std::vector<Action> legal = state->LegalActions();
  std::vector<Action> correct = {(Action)ActionType::kLoseCard1,
                                 (Action)ActionType::kLoseCard2};
  SPIEL_CHECK_TRUE(legal == correct);
  state->ApplyAction((Action)ActionType::kLoseCard1);

  SPIEL_CHECK_EQ(state->GetCoins(1), 0);
  SPIEL_CHECK_EQ((int)state->GetCardsState(0).at(0), (int)CardStateType::kFaceUp);
  SPIEL_CHECK_FALSE(state->IsTerminal());
  std::vector<double> rewards = state->Rewards();
  std::vector<double> returns = state->Returns();
  SPIEL_CHECK_EQ(rewards.at(0), -1);
  SPIEL_CHECK_EQ(returns.at(0), -1);
  SPIEL_CHECK_EQ(rewards.at(1), 1);
  SPIEL_CHECK_EQ(returns.at(1), 1);
}

// Assassin
void CoupAssassinateTest() {
  std::shared_ptr<const Game> game = LoadGame("coup");
  std::unique_ptr<State> tempState = game->NewInitialState();
  CoupState* state = static_cast<CoupState*>(tempState.get());

  // Deal cards
  state->ApplyAction((Action)CardType::kAssassin);
  state->ApplyAction((Action)CardType::kAssassin);
  state->ApplyAction((Action)CardType::kContessa);
  state->ApplyAction((Action)CardType::kDuke);

  state->ApplyAction((Action)ActionType::kForeignAid);
  state->ApplyAction((Action)ActionType::kPass);
  state->ApplyAction((Action)ActionType::kIncome);
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 0);
  SPIEL_CHECK_EQ(state->GetCoins(0), 3);

  state->ApplyAction((Action)ActionType::kAssassinate);
  
  std::vector<Action> legal = state->LegalActions();
  std::vector<Action> correct = {(Action)ActionType::kLoseCard1,
                                 (Action)ActionType::kLoseCard2,
                                 (Action)ActionType::kBlock,
                                 (Action)ActionType::kChallenge};
  SPIEL_CHECK_TRUE(legal == correct);
  state->ApplyAction((Action)ActionType::kLoseCard1);

  SPIEL_CHECK_EQ(state->GetCoins(0), 0);
  SPIEL_CHECK_EQ((int)state->GetCardsState(1).at(0), (int)CardStateType::kFaceUp);
  SPIEL_CHECK_FALSE(state->IsTerminal());
  std::vector<double> rewards = state->Rewards();
  std::vector<double> returns = state->Returns();
  SPIEL_CHECK_EQ(rewards.at(0), 1);
  SPIEL_CHECK_EQ(returns.at(0), 1);
  SPIEL_CHECK_EQ(rewards.at(1), -1);
  SPIEL_CHECK_EQ(returns.at(1), -1);
}

void CoupDoubleAssassinateTest() {
  std::shared_ptr<const Game> game = LoadGame("coup");
  std::unique_ptr<State> tempState = game->NewInitialState();
  CoupState* state = static_cast<CoupState*>(tempState.get());

  // Deal cards
  state->ApplyAction((Action)CardType::kAssassin);
  state->ApplyAction((Action)CardType::kAssassin);
  state->ApplyAction((Action)CardType::kContessa);
  state->ApplyAction((Action)CardType::kDuke);

  state->ApplyAction((Action)ActionType::kForeignAid);
  state->ApplyAction((Action)ActionType::kPass);
  state->ApplyAction((Action)ActionType::kIncome);
  state->ApplyAction((Action)ActionType::kAssassinate);
  state->ApplyAction((Action)ActionType::kChallenge);
  // P1 had an assassin, so P2 loses the challenge
  // Lose 1 card for assassinate, 1 for lost challenge,
  // therefore lose the game.
  SPIEL_CHECK_EQ(state->GetCoins(0), 0);
  SPIEL_CHECK_TRUE(state->IsTerminal());
  std::vector<double> rewards = state->Rewards();
  std::vector<double> returns = state->Returns();
  SPIEL_CHECK_EQ(rewards.at(0), 2);
  SPIEL_CHECK_EQ(returns.at(0), 2);
  SPIEL_CHECK_EQ(rewards.at(1), -2);
  SPIEL_CHECK_EQ(returns.at(1), -2);
}

// Ambassador
void CoupExchangeTest() {
  std::shared_ptr<const Game> game = LoadGame("coup");
  std::unique_ptr<State> tempState = game->NewInitialState();
  CoupState* state = static_cast<CoupState*>(tempState.get());

  // Deal cards
  state->ApplyAction((Action)CardType::kAmbassador);
  state->ApplyAction((Action)CardType::kAssassin);
  state->ApplyAction((Action)CardType::kContessa);
  state->ApplyAction((Action)CardType::kDuke);

  state->ApplyAction((Action)ActionType::kExchange);
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 1);
  std::vector<Action> legal = state->LegalActions();
  std::vector<Action> correct = {(Action)ActionType::kPass,
                                 (Action)ActionType::kChallenge};
  SPIEL_CHECK_TRUE(legal == correct);
  state->ApplyAction((Action)ActionType::kPass);

  // Chance player deals 2 cards to P1
  SPIEL_CHECK_EQ(state->CurrentPlayer(), kChancePlayerId);
  state->ApplyAction((Action)CardType::kDuke);
  state->ApplyAction((Action)CardType::kDuke);
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 0);
  std::vector<CardType> cards = state->GetCardsValue(0);
  SPIEL_CHECK_EQ(cards.size(), 4);
  legal = state->LegalActions();
  correct = {(Action)ActionType::kExchangeReturn12,
             (Action)ActionType::kExchangeReturn13,
             (Action)ActionType::kExchangeReturn14,
             (Action)ActionType::kExchangeReturn23,
             (Action)ActionType::kExchangeReturn24,
             (Action)ActionType::kExchangeReturn34};
  SPIEL_CHECK_TRUE(legal == correct);
  state->ApplyAction((Action)ActionType::kExchangeReturn12);

  cards = state->GetCardsValue(0);
  std::vector<CardStateType> cardStates = state->GetCardsState(0);
  SPIEL_CHECK_EQ(cards.size(), 2);
  for (int c = 0; c < 2; ++c) {
    SPIEL_CHECK_GE((int)cards.at(c), (int)CardType::kDuke);
    SPIEL_CHECK_EQ((int)cardStates.at(c), (int)CardStateType::kFaceDown);
  }
  
  SPIEL_CHECK_FALSE(state->IsTerminal());
  std::vector<double> rewards = state->Rewards();
  std::vector<double> returns = state->Returns();
  for (int p = 0; p < state->NumPlayers(); ++p) {
    SPIEL_CHECK_EQ(rewards.at(p), 0);
    SPIEL_CHECK_EQ(returns.at(p), 0);
  }
}

// Captain
void CoupStealTest() {
  std::shared_ptr<const Game> game = LoadGame("coup");
  std::unique_ptr<State> tempState = game->NewInitialState();
  CoupState* state = static_cast<CoupState*>(tempState.get());

  // Deal cards
  state->ApplyAction((Action)CardType::kCaptain);
  state->ApplyAction((Action)CardType::kAssassin);
  state->ApplyAction((Action)CardType::kContessa);
  state->ApplyAction((Action)CardType::kDuke);

  state->ApplyAction((Action)ActionType::kSteal);

  SPIEL_CHECK_EQ(state->CurrentPlayer(), 1);
  std::vector<Action> legal = state->LegalActions();
  std::vector<Action> correct = {(Action)ActionType::kPass,
                                 (Action)ActionType::kBlock,
                                 (Action)ActionType::kChallenge};
  SPIEL_CHECK_TRUE(legal == correct);
  state->ApplyAction((Action)ActionType::kPass);

  SPIEL_CHECK_EQ(state->GetCoins(0), 3);
  SPIEL_CHECK_EQ(state->GetCoins(1), 0);
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 1);
  SPIEL_CHECK_FALSE(state->IsTerminal());
  std::vector<double> rewards = state->Rewards();
  std::vector<double> returns = state->Returns();
  for (int p = 0; p < state->NumPlayers(); ++p) {
    SPIEL_CHECK_EQ(rewards.at(p), 0);
    SPIEL_CHECK_EQ(returns.at(p), 0);
  }
}

void CoupBlockStealTest() {
  std::shared_ptr<const Game> game = LoadGame("coup");
  std::unique_ptr<State> tempState = game->NewInitialState();
  CoupState* state = static_cast<CoupState*>(tempState.get());

  // Deal cards
  state->ApplyAction((Action)CardType::kCaptain);
  state->ApplyAction((Action)CardType::kCaptain);
  state->ApplyAction((Action)CardType::kContessa);
  state->ApplyAction((Action)CardType::kDuke);

  state->ApplyAction((Action)ActionType::kSteal);
  state->ApplyAction((Action)ActionType::kBlock);

  SPIEL_CHECK_EQ(state->CurrentPlayer(), 0);
  std::vector<Action> legal = state->LegalActions();
  std::vector<Action> correct = {(Action)ActionType::kPass,
                                 (Action)ActionType::kChallenge};
  SPIEL_CHECK_TRUE(legal == correct);
  state->ApplyAction((Action)ActionType::kChallenge);
  // P2 had a Captain, so P2 showed it to prove it, and gets a new card
  SPIEL_CHECK_EQ(state->CurrentPlayer(), -1);
  state->ApplyAction((Action)CardType::kCaptain);
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 0);
  legal = state->LegalActions();
  correct = {(Action)ActionType::kLoseCard1,
             (Action)ActionType::kLoseCard2};
  SPIEL_CHECK_TRUE(legal == correct);
  state->ApplyAction((Action)ActionType::kLoseCard1);

  SPIEL_CHECK_EQ(state->GetCoins(0), 1);
  SPIEL_CHECK_EQ(state->GetCoins(1), 2);
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 1);
  SPIEL_CHECK_EQ((int)state->GetCardsState(0).at(0), (int)CardStateType::kFaceUp);
  SPIEL_CHECK_FALSE(state->IsTerminal());
  std::vector<double> rewards = state->Rewards();
  std::vector<double> returns = state->Returns();
  SPIEL_CHECK_EQ(rewards.at(0), -1);
  SPIEL_CHECK_EQ(returns.at(0), -1);
  SPIEL_CHECK_EQ(rewards.at(1), 1);
  SPIEL_CHECK_EQ(returns.at(1), 1);
}

// Contessa
void CoupBlockAssassinateTest() {
  std::shared_ptr<const Game> game = LoadGame("coup");
  std::unique_ptr<State> tempState = game->NewInitialState();
  CoupState* state = static_cast<CoupState*>(tempState.get());

  // Deal cards
  state->ApplyAction((Action)CardType::kAssassin);
  state->ApplyAction((Action)CardType::kContessa);
  state->ApplyAction((Action)CardType::kAssassin);
  state->ApplyAction((Action)CardType::kContessa);

  state->ApplyAction((Action)ActionType::kForeignAid);
  state->ApplyAction((Action)ActionType::kPass);
  state->ApplyAction((Action)ActionType::kIncome);
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 0);
  state->ApplyAction((Action)ActionType::kAssassinate);
  state->ApplyAction((Action)ActionType::kBlock);

  SPIEL_CHECK_EQ(state->CurrentPlayer(), 0);
  std::vector<Action> legal = state->LegalActions();
  std::vector<Action> correct = {(Action)ActionType::kPass,
                                 (Action)ActionType::kChallenge};
  SPIEL_CHECK_TRUE(legal == correct);
  state->ApplyAction((Action)ActionType::kChallenge);

  // P2 had a Contessa, so P2 showed it to prove it, and gets a new card
  SPIEL_CHECK_EQ(state->CurrentPlayer(), -1);
  state->ApplyAction((Action)CardType::kContessa);

  SPIEL_CHECK_EQ(state->GetCoins(0), 0);
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 0);
  legal = state->LegalActions();
  correct = {(Action)ActionType::kLoseCard1,
             (Action)ActionType::kLoseCard2};
  SPIEL_CHECK_TRUE(legal == correct);
  SPIEL_CHECK_FALSE(state->IsTerminal());
  std::vector<double> rewards = state->Rewards();
  std::vector<double> returns = state->Returns();
  for (int p = 0; p < state->NumPlayers(); ++p) {
    SPIEL_CHECK_EQ(rewards.at(p), 0);
    SPIEL_CHECK_EQ(returns.at(p), 0);
  }
}

// Duke
void CoupTaxTest() {
  std::shared_ptr<const Game> game = LoadGame("coup");
  std::unique_ptr<State> tempState = game->NewInitialState();
  CoupState* state = static_cast<CoupState*>(tempState.get());

  // Deal cards
  state->ApplyAction((Action)CardType::kAmbassador);
  state->ApplyAction((Action)CardType::kAssassin);
  state->ApplyAction((Action)CardType::kDuke);
  state->ApplyAction((Action)CardType::kDuke);

  state->ApplyAction((Action)ActionType::kTax);

  SPIEL_CHECK_EQ(state->CurrentPlayer(), 1);
  std::vector<Action> legal = state->LegalActions();
  std::vector<Action> correct = {(Action)ActionType::kPass,
                                 (Action)ActionType::kChallenge};
  SPIEL_CHECK_TRUE(legal == correct);
  state->ApplyAction((Action)ActionType::kPass);

  SPIEL_CHECK_EQ(state->GetCoins(0), 4);
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 1);
  SPIEL_CHECK_FALSE(state->IsTerminal());
  std::vector<double> rewards = state->Rewards();
  std::vector<double> returns = state->Returns();
  for (int p = 0; p < state->NumPlayers(); ++p) {
    SPIEL_CHECK_EQ(rewards.at(p), 0);
    SPIEL_CHECK_EQ(returns.at(p), 0);
  }
}

void CoupChallengeBlockForeignAidTest() {
  std::shared_ptr<const Game> game = LoadGame("coup");
  std::unique_ptr<State> tempState = game->NewInitialState();
  CoupState* state = static_cast<CoupState*>(tempState.get());

  // Deal cards
  state->ApplyAction((Action)CardType::kAmbassador);
  state->ApplyAction((Action)CardType::kAssassin);
  state->ApplyAction((Action)CardType::kDuke);
  state->ApplyAction((Action)CardType::kDuke);

  state->ApplyAction((Action)ActionType::kForeignAid);
  state->ApplyAction((Action)ActionType::kBlock);
  
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 0);
  std::vector<Action> legal = state->LegalActions();
  std::vector<Action> correct = {(Action)ActionType::kPass,
                                 (Action)ActionType::kChallenge};
  SPIEL_CHECK_TRUE(legal == correct);
  state->ApplyAction((Action)ActionType::kChallenge);

  // P2 had a Duke, so P2 showed it to prove it, and gets a new card
  SPIEL_CHECK_EQ(state->CurrentPlayer(), -1);
  state->ApplyAction((Action)CardType::kDuke);

  SPIEL_CHECK_EQ(state->GetCoins(0), 1);
  SPIEL_CHECK_EQ(state->CurrentPlayer(), 0);
  legal = state->LegalActions();
  correct = {(Action)ActionType::kLoseCard1,
             (Action)ActionType::kLoseCard2};
  SPIEL_CHECK_TRUE(legal == correct);
  SPIEL_CHECK_FALSE(state->IsTerminal());
  std::vector<double> rewards = state->Rewards();
  std::vector<double> returns = state->Returns();
  for (int p = 0; p < state->NumPlayers(); ++p) {
    SPIEL_CHECK_EQ(rewards.at(p), 0);
    SPIEL_CHECK_EQ(returns.at(p), 0);
  }
}

void CoupGameTests() {
  CoupGameStartTest();
  CoupIncomeTest();
  CoupPassForeignAidTest();
  CoupBlockForeignAidTest();
  CoupChallengeForeignAidTest();
  CoupLoseCardTest();
  CoupAssassinateTest();
  CoupDoubleAssassinateTest();
  CoupExchangeTest();
  CoupStealTest();
  CoupBlockStealTest();
  CoupBlockAssassinateTest();
  CoupTaxTest();
  CoupChallengeBlockForeignAidTest();
}

}  // namespace
}  // namespace coup
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::coup::BasicCoupTests();
  open_spiel::coup::CoupGameTests();
}
