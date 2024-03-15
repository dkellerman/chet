import re

def parse_chess_notation(notation, board, color):
    notation = notation.strip().lower()

    if "draw" in notation:
        return None, None, {"cmd": "draw"}
    elif "resign" in notation:
        return None, None, {"cmd": "resign"}

    if notation in ["o-o", "0-0"]:
        row = 0 if color else 7
        return (4, row), (6, row), {"castle": "K" if color else "k"}
    elif notation in ["o-o-o", "0-0-0"]:
        row = 0 if color else 7
        return (4, row), (2, row), {"castle": "Q" if color else "q"}

    promo_match = re.match(r'^([a-h])([1-8])(?:[-x])?([a-h])([1-8])(=[qrbn])?$', notation)
    if promo_match:
        from_col = ord(promo_match.group(1)) - ord('a')
        from_row = int(promo_match.group(2)) - 1
        to_col = ord(promo_match.group(3)) - ord('a')
        to_row = int(promo_match.group(4)) - 1
        promo = promo_match.group(5)[1] if promo_match.group(5) else promo_match.group(5)
        return (from_col, from_row), (to_col, to_row), {"promo": promo.upper() if promo else None}

    # Regular moves
    move_match = re.match(r'^([pnbrqk])?([a-h])?([1-8])?(?:[-x])?([a-h])([1-8])$', notation)
    if move_match:
        piece = move_match.group(1) if move_match.group(1) else "p"
        from_col = ord(move_match.group(2)) - ord('a') if move_match.group(2) else None
        from_row = int(move_match.group(3)) - 1 if move_match.group(3) else None
        to_col = ord(move_match.group(4)) - ord('a')
        to_row = int(move_match.group(5)) - 1

        # Find the from_square in the board dictionary
        if from_col is None or from_row is None:
            for col in range(8):
                for row in range(8):
                    if board.get((col, row)) == (piece.upper() if color else piece.lower()):
                        from_col, from_row = col, row
                        break
                if from_col is not None:
                    break

        return (from_col, from_row), (to_col, to_row), {"piece": piece}

    return None, None, None

# Example usage
board = {
    (0, 0): 'r', (1, 0): 'n', (2, 0): 'b', (3, 0): 'q', (4, 0): 'k', (5, 0): 'b', (6, 0): 'n', (7, 0): 'r',
    (0, 1): 'p', (1, 1): 'p', (2, 1): 'p', (3, 1): 'p', (4, 1): 'p', (5, 1): 'p', (6, 1): 'p', (7, 1): 'p',
    (0, 6): 'P', (1, 6): 'P', (2, 6): 'P', (3, 6): 'P', (4, 6): 'P', (5, 6): 'P', (6, 6): 'P', (7, 6): 'P',
    (0, 7): 'R', (1, 7): 'N', (2, 7): 'B', (3, 7): 'Q', (4, 7): 'K', (5, 7): 'B', (6, 7): 'N', (7, 7): 'R'
}

# print(parse_chess_notation("e2e4", board, True))
# print(parse_chess_notation("e2-e4", board, True))
# print(parse_chess_notation("e3xe4", board, True))
# print(parse_chess_notation("O-O", board, True))
# print(parse_chess_notation("O-O-O", board, True))
# print(parse_chess_notation("Ne4", board, True))
# print(parse_chess_notation("e4", board, True))
# print(parse_chess_notation("Nfe4", board, True))
# print(parse_chess_notation("B1c3", board, True))
# print(parse_chess_notation("a7a8q", board, True))
# print(parse_chess_notation("a7a8Q", board, True))
# print(parse_chess_notation("b7b8=q", board, True))
# print(parse_chess_notation("I resign", board, True))
# print(parse_chess_notation("Draw?", board, True))
# print(parse_chess_notation("e4+", board, True))
# print(parse_chess_notation("e4#", board, True))
# print(parse_chess_notation("e4++", board, True))
# print(parse_chess_notation("e4 e.p.", board, True))
# print(parse_chess_notation("e4 ep", board, True))
# print(parse_chess_notation("e4 EP", board, True))
# print(parse_chess_notation("e4 E.P", board, True))
# print(parse_chess_notation("e4!", board, True))
# print(parse_chess_notation("e4?", board, True))
# print(parse_chess_notation("e4!!??", board, True))

while True:
    notation = input("> ")
    if notation in ["quit", "exit"]:
        break
    print(parse_chess_notation(notation, board, True))
