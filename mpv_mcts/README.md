# Multiple Policy Value MCTS

Here we have implemented a multiple policy value network which consists of a smaller network that explores various levels of the Monte Carlo tree and provides a lookahead for the larger network to converge faster to the optimum policy.

1. The following is how a vanilla alpha-zero would perform to a (20,3) Nim game:

![alpha zero](https://media.github.ccs.neu.edu/user/7131/files/d70d4200-a592-11ea-87f1-9f867ad3b9e4)

2. Next we run the same (20,3) Nim game on our implementation of mpv-mcts keeping the configuration of the main network same as the vanilla alpha-zero.

![mpv mcts](https://media.github.ccs.neu.edu/user/7131/files/e1c7d700-a592-11ea-8760-2a26e6e23f3e)

We see that the mpv-mcts performs better than the alpha-zero algorithm in just 20 training steps
