# Chess

## Sunday

- finish engine basics & tests
  - legal moves
  - king capture
  - board setup
  - offer draw
  - 3-fold
  - insufficient material
  - cache/lru
  - tests
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


## TODO
[1]
  - get legal moves
    - attacked squares
    - pins
    - enp pin
    - checks
    - castles
    - enp
  - return as notation (?)
[1]
  - allow king capture
  - is_ended
  - score (material count)
  - randomize initial position (check legality)
  - 3-fold rep
  - insufficient material
  - offer/accept draw
[1] - lookahead
[1] - perf profile
[1]
  - use q state table
  - cache and pruning
  - notation load test
  - training loop
[1] - parallelize
[1] - tests
[1] - tournament

## UI
[1]
  - make move & respond
  - is legal move


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

