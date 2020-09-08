Perfect play: Using the agent as the first player and human as the second. The agent uses optimal strategy to win the game.

~~~
Initial state:
..........
Player 0 sampled action: x(0,1)
Next state:
xx........
Choose an action (empty to print legal actions): 
Legal actions(s):
  2: o(0,2)    3: o(0,3)    4: o(0,4)  
Choose an action (empty to print legal actions): 4
Player 1 sampled action: o(0,4)
Next state:
xxooo.....
Player 0 sampled action: x(0,5)
Next state:
xxooox....
Choose an action (empty to print legal actions): 
Legal actions(s):
  6: o(0,6)    7: o(0,7)    8: o(0,8)  
Choose an action (empty to print legal actions): 8
Player 1 sampled action: o(0,8)
Next state:
xxoooxooo.
Player 0 sampled action: x(0,9)
Next state:
xxoooxooox
Returns: 1.0 -1.0 , Game actions: x(0,1) o(0,4) x(0,5) o(0,8) x(0,9)
Number of games played: 1
Number of distinct games played: 1
Overall wins [1, 0]
Overall returns [1.0, -1.0]
~~~
