# RL_gym

Plan to learn RL by doing some interesting stuff, like playing game provided by OpenAI Gym


## CartPole

Observation can reflect the state of the agent, but is continuous. So we need to parameterize the environment space.

1. CartPole_pg: Simple policy gradient method, using a hand-written naive 2-layer neural network
2. CartPole_q: Try to solve the problem with Q-learning, still working on it.

## Frozen Lake

Easy problem with 16 states and 4 actions, can be solved by implementing Q-learning algorithm.

1. Frozen_Lake: Classical Q-learning with state-action table.
2. Frozen_Lake_nn: Q-learning with trivial vectorized input and weights corresponding to the state-action table.
3. Frozen_Lake_nn_tf: Rewrite 2 using tensorflow. But the performance is quite poor.
