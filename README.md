# Autonomous greenhouse simulation.
## Project goal
The goal of this project is to autonomously control a simulated greenhouse using reinforcement learning techniques. The greenhouse has several factors which can be controlled, this includes:
  1. Vents
  2. Heating
  3. Screens
  4. Lighting
  5. Fogging
  6. Irrigation
  7. CO2 injection.
## Reward function
The reward function includes both the resources used and the average time taken over 5 years per harvest. The best possible value is 0 however in practice this is impossible as it would mean instantanious growth with no resource use.

    **= -mean(harvest_times) - resources_used/amount_of_harvests**
## Policies
### Random policy
Firstly a completely random policy. This controls the greenhouse via a random uniform distribution. It fluctuates a large amount and follows the general outside temperature, it can get both extremely hot and cold in this greenhouse, in reality all of the plants would die.
The reward given to this was -8500.

![Random temperatures](https://github.com/tonzowonzo/greenhouse/blob/master/images/randomtemps.png)

![Random harvest times](https://github.com/tonzowonzo/greenhouse/blob/master/images/randomgdd.png)

### Naive policy
Secondly a naive policy. This opens the vents, closes the screens and turns on fogging when it is too hot and turns on heaters and opens the screens while it is too cold. It performs significantly better than the above with a reward value of -3800.

![Naive temperatures](https://github.com/tonzowonzo/greenhouse/blob/master/images/naivetemps.png)

![Naive harvest times](https://github.com/tonzowonzo/greenhouse/blob/master/images/naivegdd.png)
