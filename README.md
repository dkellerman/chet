# Chess

## TODO
- insufficient material
- player-based state
- should it check for ending right away?
- lookahead
- enp pin


- UI/server deployed to vercel w redis
  - manual layout / drag
  - new game / url
  - load game / url
  - make move & respond
  - illegal move
  - resign/draw
  - game over
  - promotions
  - shuffle board

- minimax player -> pos score, order by capture, pruning
  - store as q state table
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


## maybe
- offer draw
- allow output of shortened notation
- requires promotion
