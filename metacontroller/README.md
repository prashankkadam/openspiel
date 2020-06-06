# Metacontroller

This is algorithm implemented in the Agent-57 paper of Deepmeind. This algorithm consists of two parts:
* A non-stationary multiarm bandit algorithm called Sliding window UCB
* Greedy epsilon 

The mathematical formalism of this algorithm as per the Agent-57 paper and as implemented in this repository can be given as follows:

![Selection_002](https://user-images.githubusercontent.com/17771219/83939645-82451380-a7ac-11ea-9d65-340beb3db109.png)
![Selection_003](https://user-images.githubusercontent.com/17771219/83939646-82451380-a7ac-11ea-8fe9-1e34409018ec.png)

The following are the results for basic tests with vanilla Alpha-Zero using UCT and vanilla Alpha-Zero using the Metacontroller. For the sake of this test we use only 5 simulations in the MCTS for either of the models. The game that this test has been done on is TicTacToe in OpenSpiel.

* Alpha-Zero (UCT) - [uct_c = 1, simulations = 5]:

![Figure_1](https://user-images.githubusercontent.com/17771219/83939685-dbad4280-a7ac-11ea-891c-77bd94920583.png)

* Alpha-Zero (Metacontroller) - [uct_c = 1, simulations = 5, epsilon = 0.7, tao = 50]:

![Figure_1](https://user-images.githubusercontent.com/17771219/83939691-e49e1400-a7ac-11ea-99f0-c588a9309306.png)

We can see a slight drop in the loss for the Meatacontroller over UCT algorithm. The prediction accuracy and absolute value predictions show clear improvements in Metacontroller over UCB. Note that we should be more concerned about the absolute MCTS value prediction accuracy as our multiarm bandit directly affects this parameter, here, Metacontroller clearly outperforms the UCT algorithm.
