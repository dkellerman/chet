# Chess

## TODO
- board setup
- offer draw
- 3-fold
- insufficient material
- check legal position
- randomize position
- is terminal state
- player-based state
- pos score
- lookahead
- enp pin
- requires promotion


- UI/server deployed to vercel w redis
  - manual layout / drag
  - new game / url
  - load game / url
  - make move & respond
  - illegal move
  - resign/draw
  - game over
  - promotions
  + shuffle board
- minimax player -> pos score, order by capture, pruning
  - store as q state table
  - lookahead
  - is_terminal
  - score
  - minimax
  - alpha beta pruning
  - tree ordering
- profile: deep perf & mem/caching
  - cli args
  - profiler
  - timeits
  - flop calc
  - mem reqs
  - htop etc


## Algorithms
- minimax w alpha beta pruning
- mcts
- q/td/dp
- elo
- transformer
  - tokenizer (notation)
- genetic
- dqn
- ppo
- a star

