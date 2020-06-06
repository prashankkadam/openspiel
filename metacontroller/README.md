# Metacontroller

This is algorithm implemented in the Agent-57 paper of Deepmeind. This algorithm consists of two parts:
* A non-stationary multiarm bandit algorithm called Sliding window UCB
* Greedy epsilon 

The mathematical formalism of this algorithm can be given as follows:


The following are the results for basic tests with vanilla Alpha-Zero using UCT and vanilla Alpha-Zero using the Metacontroller. For the sake of this test we use only 5 simulations in the MCTS for either of the models. The game that this test has been done on is TicTacToe in OpenSpiel.

* Alpha-Zero (UCT) - [uct_c = 1, simulations = 5]:

* Alpha-Zero (Metacontroller) - [uct_c = 1, simulations = 5, epsilon = 0.7, tao = 50]:


