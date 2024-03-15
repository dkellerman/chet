#!/usr/bin/env python3

import re
from collections import defaultdict

WHITE, BLACK = True, False
INITIAL_BOARD = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
NOTATION_RE = re.compile(
    r"^([NBRQK])?([a-h])?([1-8])?(x)?([a-h][1-8])(=[NBRQK])?(\+|#)?$|^O-O(-O)?$"
)


class Game:
    def __init__(self, board=INITIAL_BOARD, players=("H", "C")):
        self.set_board_state(board)
        self.players = players
        self.turn = 0
        self.history = []
        self.status = None
        self.check = False
        self.ending = None
        # castles still available: short/long, caps for white
        self.castle = ["S", "L", "s", "l"]
        self.draw_counter = 100
        self.enpassant = []

    @property
    def color(self):
        return self.turn == 0

    @property
    def player_type(self):
        return self.players[self.turn]

    @property
    def last_move(self):
        return self.history[-1] if len(self.history) else None

    def play(self):
        self.status = "PLAYING"
        while self.status == "PLAYING":
            if self.player_type == "H":
                move = self.get_human_move()
            elif self.player_type == "C":
                move = self.get_computer_move()
            self.make_move(move)

    def get_human_move(self):
        raise NotImplementedError

    def get_computer_move(self):
        raise NotImplementedError

    def replay(self, moves):
        for move in moves:
            self.make_move(move)

    def make_move(self, move):
        if type(move) == str:
            move = self.parse_notation(move)
        notation = self.to_notation(*move)
        from_sq, to_sq, props = move
        piece = self.board[from_sq]
        piece_type = piece.lower() if piece else None
        color = piece.isupper() if piece else None
        promo = props.get("promo")
        promo = promo.upper() if promo and self.color else promo

        # move
        self.board[to_sq] = promo or self.board[from_sq]
        self.board[from_sq] = None

        # complete castle
        if piece_type == "k" and from_sq[0] == 4 and to_sq[0] == 6:
            self.board[(5, 0 if color else 7)] = 'R' if color else 'r'
            self.board[(7, 0 if color else 7)] = None
        elif piece_type == "k" and from_sq[0] == 4 and to_sq[0] == 2:
            self.board[(3, 0 if color else 7)] = 'R' if color else 'r'
            self.board[(0, 0 if color else 7)] = None

        # update castles available
        if piece_type == "k":
            self.castle = [c for c in self.castle if c.upper() != color]
        elif piece_type == "r" and from_sq in [(0, 0), (0, 7)]:
            self.castle = [c for c in self.castle if c != ("L" if color else "l")]
        elif piece_type == "r" and from_sq in [(7, 0), (7, 7)]:
            self.castle = [c for c in self.castle if c != ("S" if color else "s")]

        # 50-move rule update
        if self.board[to_sq] is None and piece_type != "p":
            self.draw_counter -= 1
        else:
            self.draw_counter = 100

        self.history.append(notation)

        # is game over
        if self.is_checkmate():
            self.status = "WWINS" if self.color else "BWINS"
            self.ending = "Checkmate"
        elif self.is_stalemate():
            self.status = "DRAW"
            self.ending = "Stalemate"
        elif self.draw_counter <= 0:
            self.status = "DRAW"
            self.ending = "Fifty-move rule"
        else:
            self.turn = 1 - self.turn

    def get_legal_moves(self, color=None, with_board=None):
        # with_board param is a board for a lookahead
        board = with_board or self.board
        legal_moves = []

        # loop through all squares
        for row in range(8):
            for col in range(8):
                piece = board[(col, row)]
                if piece is not None and (color is None or (piece.isupper() == color)):
                    piece_moves = self.get_moves_for_square((col, row), with_board)
                    legal_moves.extend(piece_moves)
        return legal_moves

    def get_moves_for_square(self, from_square, with_board=None):
        board = with_board or self.board
        piece = board[from_square]
        from_col, from_row = from_square
        piece_type = piece.lower()
        piece_color = piece.isupper()
        legal_moves = []

        if piece_type == "p":
            row_dir = 1 if piece_color == WHITE else -1
            row_start = 1 if piece_color == WHITE else 6
            push1 = (from_col, from_row + row_dir)
            push2 = (from_col, from_row + (row_dir * 2))
            cap1 = (from_col - 1, from_row + row_dir)
            cap2 = (from_col + 1, from_row + row_dir)

            if not board[push1]:
                if push1[1] in [0, 7]:
                    for promo in ["q", "r", "b", "n"]:
                        legal_moves.append(
                            (from_square, push1, dict(promo=promo, noattack=True))
                        )
                else:
                    legal_moves.append((from_square, push1, dict(noattack=True)))

            if from_row == row_start and not board[push2]:
                legal_moves.append((from_square, push2, dict(noattack=True, enp=True)))
            if from_col > 0 and board[cap1] and board[cap1].isupper() != piece_color:
                legal_moves.append((from_square, cap1, dict()))
            if from_col < 7 and board[cap2] and board[cap2].isupper() != piece_color:
                legal_moves.append((from_square, cap2, dict()))

        elif piece_type == "r":
            dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            legal_moves.extend(self.get_dirs(from_square, dirs, with_board))

        elif piece_type == "q":
            dirs = [
                (0, 1),
                (0, -1),
                (1, 0),
                (-1, 0),
                (1, 1),
                (-1, 1),
                (1, -1),
                (-1, -1),
            ]
            legal_moves.extend(self.get_dirs(from_square, dirs, with_board))

        elif piece_type == "b":
            dirs = [(1, 1), (-1, 1), (1, -1), (-1, -1)]
            legal_moves.extend(self.get_dirs(from_square, dirs, with_board))

        elif piece_type == "n":
            dirs = [
                (1, 2),
                (2, 1),
                (-1, 2),
                (-2, 1),
                (1, -2),
                (2, -1),
                (-1, -2),
                (-2, -1),
            ]
            for dir in dirs:
                col, row = from_col + dir[0], from_row + dir[1]
                if 0 <= col < 8 and 0 <= row < 8:
                    if (
                        not board[(col, row)]
                        or board[(col, row)].isupper() != piece_color
                    ):
                        legal_moves.append((from_square, (col, row), dict()))

        elif piece_type == "k":
            dirs = [
                (1, 0),
                (1, 1),
                (0, 1),
                (-1, 1),
                (-1, 0),
                (-1, -1),
                (0, -1),
                (1, -1),
            ]
            for dir in dirs:
                col, row = from_col + dir[0], from_row + dir[1]
                if 0 <= col < 8 and 0 <= row < 8:
                    if (
                        not board[(col, row)]
                        or board[(col, row)].isupper() != piece_color
                    ):
                        legal_moves.append((from_square, (col, row), dict()))

            if not with_board:  # check castles, unless we're doing a recursive call
                if self.can_castle(piece_color, short=True):
                    castle = "S" if piece_color else "s"
                    legal_moves.append(
                        (from_square, (from_col + 2, from_row), dict(castle=castle))
                    )
                if self.can_castle(piece_color, short=False):
                    castle = "L" if piece_color else "l"
                    legal_moves.append(
                        (from_square, (from_col - 2, from_row), dict(castle=castle))
                    )

        if not with_board:  # lookahead for check, also don't recurse more than once
            non_checks = []
            for from_square, to_square, props in legal_moves:
                board2 = defaultdict(lambda: None, board.items())
                from_piece = board2[from_square]
                board2[to_square] = from_piece
                board2[from_square] = None
                is_check = self.is_check(piece_color, with_board=board2)
                if not is_check:
                    non_checks.append((from_square, to_square, props))
            legal_moves = non_checks

        return legal_moves

    def can_castle(self, color, short=True):
        board = self.board
        ccode = "s" if short else "l"
        ccode = ccode.upper() if color else ccode
        row = 0 if color else 7

        if (
            (ccode not in self.castle)
            or ((board[(4, row)] or "-").lower() != "k")
            or (short and (board[(7, row)] or "-").lower() != "r")
            or (short and board[(5, row)])
            or (short and board[(6, row)])
            or (not short and (board[(0, row)] or "-").lower() != "r")
            or (not short and board[(1, row)])
            or (not short and board[(2, row)])
            or (not short and board[(3, row)])
        ):
            return False

        attacked = self.get_attacked_squares(not color)
        if (
            (short and (5, row) in attacked)
            or (short and (6, row) in attacked)
            or (not short and (1, row) in attacked)
            or (not short and (2, row) in attacked)
            or (not short and (3, row) in attacked)
            or ((4, row) in attacked)
        ):
            return False

        return True

    def get_attacked_squares(self, by_color=None, with_board=None):
        board = with_board or self.board
        moves = self.get_legal_moves(by_color, with_board=board)
        return [to_sq for _, to_sq, props in moves if not props.get("noattack", False)]

    def get_dirs(self, from_square, dirs, with_board=None):
        """Get moves by directions"""
        moves = []
        for row_dir, col_dir in dirs:
            board = with_board or self.board
            piece_color = board[from_square].isupper()
            col, row = from_square
            while True:
                row += row_dir
                col += col_dir
                if row < 0 or row > 7 or col < 0 or col > 7:
                    break
                if not board[(col, row)]:
                    moves.append((from_square, (col, row), dict()))
                elif board[(col, row)].isupper() != piece_color:
                    moves.append((from_square, (col, row), dict()))
                    break
                else:
                    break
        return moves

    def is_legal_move(self, move, color=None):
        for m in self.get_legal_moves(color):
            if move[0] == m[0] and move[1] == m[1]:
                return True
        return False

    def is_check(self, on_color=None, with_board=None):
        board = with_board or self.board
        if on_color is not None:
            attacked = self.get_attacked_squares(not on_color, with_board=board)
        else:
            attacked = self.get_attacked_squares(with_board=board)
        for to_sq in attacked:
            piece = board[to_sq]
            if piece and piece.lower() == "k":
                return True
        return False

    def requires_promotion(self, move):
        if type(move) == str:
            move = self.parse_notation(move)
        from_sq, to_sq, _ = move
        return ((self.board[from_sq] or "-").lower() == "p") and to_sq[1] in [0, 7]

    def parse_notation(self, val):
        val = val.lower().strip()

        # match = NOTATION_RE.match(val)
        # TODO:
        # r"^([NBRQK])?([a-h])?([1-8])?(x)?([a-h][1-8])(=[NBRQK])?(\+|#)?$|^O-O(-O)?$"
        # 1. Piece
        # 2. From file
        # 3. From rank
        # 4. Capture
        # 5. To square
        # 6. Promotion
        # 7. Check
        # ?? if match and match.group(10):

        match = re.match(r"([a-h])([1-8])([-x]?)([a-h])([1-8])([nbrq]?)", val)
        if not match:
            return None
        col_from = ord(match.group(1)) - ord("a")
        row_from = int(match.group(2)) - 1
        col_to = ord(match.group(4)) - ord("a")
        row_to = int(match.group(5)) - 1
        props = dict()
        promo = match.group(6) or None
        if promo and (promo.lower() in "nbrq"):
            props["promo"] = promo.lower()
        return (col_from, row_from), (col_to, row_to), props

    def to_notation(self, from_sq, to_sq, props=dict()):
        col_from, row_from = from_sq
        col_to, row_to = to_sq
        sep = "" if not self.board[to_sq] else "x"

        return (
            f"{chr(ord('a') + col_from)}{row_from + 1}{sep}"
            f"{chr(ord('a') + col_to)}{row_to + 1}"
            f"{props.get('promo') or ''}"
        )

    def is_checkmate(self, on_color=None):
        on_color = on_color if on_color is not None else not self.color
        return self.is_check(on_color) and not len(self.get_legal_moves(on_color))

    def is_stalemate(self):
        return not self.is_check() and not len(self.get_legal_moves(self.color))

    def set_board_state(self, board_str):
        state = defaultdict(lambda: None)
        rows = board_str.split('/')
        for row_index, row in enumerate(rows):
            col_index = 0
            for char in row:
                if char.isdigit():
                    col_index += int(char)
                else:
                    state[(col_index, 7 - row_index)] = char
                    col_index += 1
        self.board = state

    def get_board_state(self):
        board_str = ''
        for row_index in range(7, -1, -1):
            empty_count = 0
            for col_index in range(8):
                piece = self.board.get((col_index, row_index))
                if piece:
                    if empty_count > 0:
                        board_str += str(empty_count)
                        empty_count = 0
                    board_str += piece
                else:
                    empty_count += 1
            if empty_count > 0:
                board_str += str(empty_count)
            if row_index > 0:
                board_str += '/'
        return board_str
