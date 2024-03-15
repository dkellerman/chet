#!/usr/bin/env python3

import argparse, csv
from tqdm import tqdm
from chess import Game, Computer

if __name__ == "__main__":
    argp = argparse.ArgumentParser()
    argp.add_argument("--play", "-p", type=int, default=0)
    options = argp.parse_args()

    if options.play > 0:
        for _ in tqdm(range(options.play)):
            game = Game(players=(Computer(), Computer()))
            game.play()
    else:
        game_logs = []
        with open('./misc/games.csv', 'r') as file:
            reader = csv.DictReader(file)
            game_logs = [row['moves'].split() for row in reader]
        for i, log in enumerate(game_logs):
            print(f"--Game {i+1}--")
            game = Game()
            for move in log:
                m = game.parse_notation(move)
                if m is None:
                    print(f"[ERR] => [{'W' if game.turn else 'B'}] {move}")
                    game.print_board()
                    break
                game.make_move(m)


