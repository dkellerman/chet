#!/usr/bin/env python3

"""CLI for playing chess"""

import sys, re, random, os
from . import Game

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


class CLI(Game):
    def get_human_move(self):
        self.render_board()

        # print last move if present
        if self.last_move:
            print(f"[{'B' if self.color else 'W'}]", self.last_move)

        move = None
        while not move:
            prompt = f"[{'W' if self.color else 'B'}] => "
            val = input(prompt).lower().strip()
            if val in ["quit", "exit"]:
                sys.exit(0)
            elif val in ["legal", "l"]:
                legal_moves = self.get_legal_moves(color=self.color)
                legal_moves = [self.to_notation(*m) for m in legal_moves]
                # consolidate moves with pawn promotions
                legal_moves = list(
                    set([re.sub(r"[qrbn]$", "", l) for l in legal_moves])
                )
                print("Legal moves:", "|".join(legal_moves))
                continue

            move = self.parse_notation(val)
            if not move:
                print("Please enter a move, e.g. e2e4, or 'quit' to exit.")
            elif not move[2].get("promo") and self.requires_promotion(move):
                promo = input("Promote to [q,r,b,n]: ").lower().strip()
                move = (move[0], move[1], {**move[2], "promo": promo})
            elif not self.is_legal_move(move, self.color):
                print("Illegal move!")
                move = None
        return move

    def get_computer_move(self):
        moves = self.get_legal_moves(self.color)
        return random.choice(moves)

    def render_board(self):
        os.system("clear")
        for row in range(7, -1, -1):
            for col in range(8):
                icon = PIECE_ICONS.get(self.board[(col, row)], "-")
                print(icon, end=" ")
            print()


if __name__ == "__main__":
    game = CLI(players=("H", "C"))
    game.play()

    # finished:
    game.render_board()
    desc = {
        "WWINS": "White wins",
        "BWINS": "Black wins",
        "DRAW": "Draw",
    }[game.status]
    print(f"\n{desc}: {game.ending}!")
