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
  testing::ResampleInfostateTest(*LoadGame("coup"), /*num_sims=*/100);
  auto observer = LoadGame("coup")
                      ->MakeObserver(kDefaultObsType,
                                     GameParametersFromString("single_tensor"));
  testing::RandomSimTestCustomObserver(*LoadGame("coup"), observer);
}

// void PolicyTest() {
//   using PolicyGenerator = std::function<TabularPolicy(const Game& game)>;
//   std::vector<PolicyGenerator> policy_generators = {
//       GetAlwaysFoldPolicy,
//       GetAlwaysCallPolicy,
//       GetAlwaysRaisePolicy
//   };

//   std::shared_ptr<const Game> game = LoadGame("coup");
//   for (const auto& policy_generator : policy_generators) {
//     testing::TestEveryInfostateInPolicy(policy_generator, *game);
//     testing::TestPoliciesCanPlay(policy_generator, *game);
//   }
// }

}  // namespace
}  // namespace coup
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::coup::BasicCoupTests();
  // open_spiel::coup::PolicyTest();
}
