#!/usr/bin/env python3

"""CLI for playing chess"""

import sys, re, os
from . import Game, Player, Computer

PIECE_ICONS = dict(
    p="♙",
    r="♖",
    n="♘",
    b="♗",
    q="♕",
    k="♔",
    P="♟",
    R="♜",
    N="♞",
    B="♝",
    Q="♛",
    K="♚",
)


class Human(Player):
    def get_move(self, game):
        render_board(game.board)

        # print last move if present
        if game.last_move:
            print(f"[{'B' if game.cur_color else 'W'}]", game.last_move)

        move = None
        while not move:
            prompt = f"[{'W' if game.cur_color else 'B'}] => "
            val = input(prompt).lower().strip()
            if val in ["quit", "exit"]:
                sys.exit(0)
            elif val in ["legal", "l"]:
                legal_moves = game.get_legal_moves(color=game.cur_color)
                legal_moves = [game.to_notation(*m) for m in legal_moves]
                # consolidate moves with pawn promotions
                legal_moves = list(
                    set([re.sub(r"[qrbn]$", "", l) for l in legal_moves])
                )
                print("Legal moves:", "|".join(legal_moves))
                continue

            move = game.parse_notation(val)
            if not move:
                print("Please enter a move, e.g. e2e4, or 'quit' to exit.")
            elif not move[2].get("promo") and game.requires_promotion(move):
                promo = input("Promote to [q,r,b,n]: ").lower().strip()
                move = (move[0], move[1], {**move[2], "promo": promo})
            elif not game.is_legal_move(move, game.cur_color):
                print("Illegal move!")
                move = None
        return move


def render_board(board):
    os.system("clear")
    for row in range(7, -1, -1):
        for col in range(8):
            icon = PIECE_ICONS.get(board[(col, row)], "-")
            print(icon, end=" ")
        print()


if __name__ == "__main__":
    game = Game(players=(Human(), Computer()))
    game.play()

    # finished:
    render_board(game.board)
    desc = {
        "WWINS": "White wins",
        "BWINS": "Black wins",
        "DRAW": "Draw",
    }[game.status]
    print(f"\n{desc}: {game.status_desc}!")
