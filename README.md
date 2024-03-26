# Chess

## TODO
- rules
  - black not moving out of check?
  - insufficient material
  - enp pin
  - check for ending right away

- lookahead

- refactors
  - player-based state

- UI/server
  - layout & pieces
  - dragging
  - illegal move (client?)
  - new game -> redirect to url
  - load game
  - make move & respond
  - illegal move (server)
  - game over
  - promotions
  - shuffle board
  - setup board?

- minimax player
  - lookahead tree
  - minimax impl
  - alpha beta pruning
  - order by capture
  - tree ordering
  - load/save player

- profile: perf & mem/caching
  - self-play loop
  - archive test
  - cli args
  - profiler
  - timeits
  - flop calc
  - mem reqs, adjust cache
  - htop etc
  - try:
    - typing
    - mojo
    - bitboards


## Algorithms
- minimax w pruning
- mcts
- q
- elo
- bitboards
- transformer
- tokenizer for transformer (notation)
- genetic (player hyperparams)
- dqn
- ppo
- mpc - model predictive control
- jepa
- star/quiet-star
- basic nn
- basic cnn
- ebm?! - energy-based model
- rlhf?
- a star?


## maybe
- offer draw
- allow output of shortened notation
- adorn notation (check, etc.)
- requires promotion
