#!/usr/bin/env python3

import re, random, json
from collections import defaultdict

WHITE, BLACK = True, False
EMPTY_STATE = "8/8/8/8/8/8/8/8"
INITIAL_STATE = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
NOTATION_RE = re.compile(
    r"^([NBRQKP])?([a-h])?([1-8])?(x)?([a-h])([1-8])(=?[nrbqNRBQ])?"
    r"(\+|#|\?|\!|(\s*[eE]\.?[pP]\.?))*$"
)


class Game:
    def __init__(self, state=INITIAL_STATE, players=None):
        self.players = players or [Player(), Player()]
        self.turn = 0
        self.history = []
        self.status = None
        self.check = False
        self.status_desc = None
        self.castle = ["K", "Q", "k", "q"]
        self.draw_counter = 0
        self.half_move_counter = 1
        self.enpassant = None
        self.set_board_state(state)

    @property
    def cur_color(self):
        return WHITE if self.turn == 0 else BLACK

    @property
    def cur_player(self):
        return self.players[self.turn]

    @property
    def last_move(self):
        return self.history[-1] if len(self.history) else None

    def play(self, moves=[]):
        if not self.players or len(self.players) != 2:
            raise ValueError("Two players required")
        self.status = "PLAYING"
        while self.status == "PLAYING":
            move = moves.pop() if len(moves) else self.cur_player.get_move(self)
            self.make_move(move)

    def make_move(self, move):
        if type(move) == str:
            move = self.parse_notation(move)
        elif type(move) == list:
            for m in move:
                self.make_move(m)
            return

        notation = self.to_notation(*move)
        from_sq, to_sq, props = move

        action = props.get("action", None)
        if action:
            if action == "resign":
                self.status = "BWINS" if self.cur_color else "WWINS"
                self.status_desc = "Resignation"
            elif action == "draw":
                self.status = "DRAW"
                self.status_desc = "Agreed to draw"
            self.history.append(notation)
            return

        piece = self.board[from_sq]
        piece_type = piece.lower() if piece else None
        color = piece.isupper() if piece else None
        promo = props.get("promo")
        promo = promo.upper() if promo and self.cur_color else promo

        # move
        self.board[to_sq] = promo or self.board[from_sq]
        self.board[from_sq] = None

        # mark enpassantable pawn
        self.enpassant = (
            to_sq if piece_type == "p" and abs(from_sq[1] - to_sq[1]) == 2 else None
        )

        # complete castle
        if piece_type == "k" and from_sq[0] == 4 and to_sq[0] == 6:
            self.board[(5, 0 if color else 7)] = "R" if color else "r"
            self.board[(7, 0 if color else 7)] = None
        elif piece_type == "k" and from_sq[0] == 4 and to_sq[0] == 2:
            self.board[(3, 0 if color else 7)] = "R" if color else "r"
            self.board[(0, 0 if color else 7)] = None

        # update castles available
        if piece_type == "k":
            self.castle = [c for c in self.castle if c.upper() != color]
        elif piece_type == "r" and from_sq in [(0, 0), (0, 7)]:
            self.castle = [c for c in self.castle if c != ("Q" if color else "q")]
        elif piece_type == "r" and from_sq in [(7, 0), (7, 7)]:
            self.castle = [c for c in self.castle if c != ("K" if color else "k")]

        # half-move and 50-move rule update
        if color == BLACK:
            self.half_move_counter += 1
            self.draw_counter += 1
        if (self.board[to_sq] is None and (piece_type != "p")) or "x" in notation:
            self.draw_counter = 0

        self.history.append(notation)

        # is game over
        if self.is_checkmate():
            self.status = "WWINS" if self.cur_color else "BWINS"
            self.status_desc = "Checkmate"
        elif self.draw_counter >= 50:
            self.status = "DRAW"
            self.status_desc = "Fifty-move rule"
        else:
            self.turn = 1 - self.turn

        if self.is_stalemate():
            self.status = "DRAW"
            self.status_desc = "Stalemate"

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
            push1 = (from_col, from_row + row_dir)
            push2 = (from_col, from_row + (row_dir * 2))
            cap1 = (from_col - 1, from_row + row_dir)
            cap2 = (from_col + 1, from_row + row_dir)

            def _append_with_promos(m, **kwargs):
                if m[1] in [0, 7]:
                    for promo in ["q", "r", "b", "n"]:
                        legal_moves.append(
                            (from_square, m, dict(promo=promo, **kwargs))
                        )
                else:
                    legal_moves.append((from_square, m, dict(**kwargs)))

            if push1[1] <= 7 and push1[1] >= 0 and not board[push1]:
                _append_with_promos(push1, noattack=True)
            if push2[1] <= 7 and push2[1] >= 0 and not board[push2]:
                _append_with_promos(push2, noattack=True)
            if from_col > 0 and board[cap1] and board[cap1].isupper() != piece_color:
                _append_with_promos(cap1)
            if from_col < 7 and board[cap2] and board[cap2].isupper() != piece_color:
                _append_with_promos(cap2)

            if (
                self.enpassant
                and from_row == self.enpassant[1]
                and abs(from_col - self.enpassant[0]) == 1
            ):
                enp_to = (self.enpassant[0], from_row + row_dir)
                legal_moves.append((from_square, enp_to, dict(enp=True)))

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
                if self.can_castle(piece_color, kingside=True):
                    castle = "K" if piece_color else "k"
                    legal_moves.append(
                        (from_square, (from_col + 2, from_row), dict(castle=castle))
                    )
                if self.can_castle(piece_color, kingside=False):
                    castle = "Q" if piece_color else "q"
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

    def can_castle(self, color, kingside=True):
        board = self.board
        ccode = "k" if kingside else "q"
        ccode = ccode.upper() if color else ccode
        row = 0 if color else 7

        if (
            (ccode not in self.castle)
            or ((board[(4, row)] or "-").lower() != "k")
            or (kingside and (board[(7, row)] or "-").lower() != "r")
            or (kingside and board[(5, row)])
            or (kingside and board[(6, row)])
            or (not kingside and (board[(0, row)] or "-").lower() != "r")
            or (not kingside and board[(1, row)])
            or (not kingside and board[(2, row)])
            or (not kingside and board[(3, row)])
        ):
            return False

        attacked = self.get_attacked_squares(not color)
        if (
            (kingside and (5, row) in attacked)
            or (kingside and (6, row) in attacked)
            or (not kingside and (1, row) in attacked)
            or (not kingside and (2, row) in attacked)
            or (not kingside and (3, row) in attacked)
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
        if type(move) == str:
            move = self.parse_notation(move)
        if len(move) > 2 and "action" in move[2]:
            return move[2]["action"] in ["resign", "draw"]
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
        val = val.strip()

        if not val:
            return None
        elif "resign" in val.lower():
            return None, None, dict(action="resign")
        elif "draw" in val.lower():
            return None, None, dict(action="draw")
        elif val.lower().startswith("o-o-o") or val.lower().startswith("0-0-0"):
            row = 0 if self.cur_color else 7
            return (4, row), (2, row), dict(castle="Q" if self.cur_color else "q")
        elif val.lower().startswith("o-o") or val.lower().startswith("0-0"):
            row = 0 if self.cur_color else 7
            return (4, row), (6, row), dict(castle="K" if self.cur_color else "k")

        match = NOTATION_RE.match(val)

        if not match:
            return None

        piece_from = match.group(1) or "p"
        piece_from = piece_from.upper() if self.cur_color else piece_from.lower()
        col_from = ord(match.group(2)) - ord("a") if match.group(2) else None
        row_from = int(match.group(3)) - 1 if match.group(3) else None
        col_to = ord(match.group(5)) - ord("a")
        row_to = int(match.group(6)) - 1

        if not col_from or not row_from:
            legal = self.get_legal_moves(self.cur_color, with_board=self.board)
            for legal_from, legal_to, _ in legal:
                if (
                    (legal_to == (col_to, row_to))
                    and (self.board[legal_from] == piece_from)
                    and (col_from is None or legal_from[0] == col_from)
                    and (row_from is None or legal_from[1] == row_from)
                ):
                    col_from, row_from = legal_from
                    break
            if col_from is None or row_from is None:
                return None

        props = dict()
        promo = match.group(7) or None
        if promo:
            props["promo"] = promo.lstrip("=").lower()
        return (col_from, row_from), (col_to, row_to), props

    def to_notation(self, from_sq, to_sq, props=dict()):
        if "action" in props:
            return props["action"]
        col_from, row_from = from_sq
        col_to, row_to = to_sq
        sep = "" if not self.board[to_sq] and not props.get("enp") else "x"

        return (
            f"{chr(ord('a') + col_from)}{row_from + 1}{sep}"
            f"{chr(ord('a') + col_to)}{row_to + 1}"
            f"{props.get('promo') or ''}"
        )

    def is_checkmate(self, on_color=None):
        on_color = on_color if on_color is not None else not self.cur_color
        return self.is_check(on_color) and not len(self.get_legal_moves(on_color))

    def is_stalemate(self):
        return not self.is_check() and not len(self.get_legal_moves(self.cur_color))

    def set_board_state(self, state_str):
        board_str, turn, castle, enpassant, draw_counter, move_counter = (
            state_str.split() + [None] * 5
        )[:6]
        if turn:
            self.turn = 0 if turn == "w" else 1
        if castle:
            self.castle = list(castle)
        if enpassant and enpassant != "-":
            enpassant = enpassant.lower() if enpassant else None
            self.enpassant = (ord(enpassant[0]) - ord("a"), int(enpassant[1]) - 1)
        if draw_counter:
            self.draw_counter = int(draw_counter)
        if move_counter:
            self.move_counter = int(move_counter)

        state = defaultdict(lambda: None)
        rows = board_str.split("/")
        for row_index, row in enumerate(rows):
            col_index = 0
            for char in row:
                if char.isdigit():
                    col_index += int(char)
                else:
                    state[(col_index, 7 - row_index)] = char
                    col_index += 1
        self.board = state

    def get_board_state(self, full=False, with_board=None):
        board = with_board or self.board
        board_str = ""
        for row_index in range(7, -1, -1):
            empty_count = 0
            for col_index in range(8):
                piece = board.get((col_index, row_index))
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
                board_str += "/"
        if full:
            state_str = (
                f"{board_str} {'w' if self.cur_color == WHITE else 'b'}"
                f" {''.join(self.castle) if self.castle else '-'}"
                f" {self.enpassant or '-'} {self.draw_counter} {self.half_move_counter}"
            )
        else:
            state_str = board_str
        return state_str

    def print_board(self):
        print()
        for row in range(7, -1, -1):
            for col in range(8):
                val = self.board[(col, row)] or "-"
                print(val, end=" ")
            print()


class Player:
    def get_move(self, game):
        raise NotImplementedError


class Computer(Player):
    def get_move(self, game):
        moves = game.get_legal_moves(game.cur_color)
        if len(moves) == 0:
            print("BOARD")
            print(game.get_board_state(full=True))
            game.print_board()
            raise ValueError("No legal moves")
        return random.choice(moves)
