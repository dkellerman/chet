#!/usr/bin/env python3

import re, random, sys, os, functools, collections


WHITE, BLACK = True, False
EMPTY_STATE = "8/8/8/8/8/8/8/8 w KQkq - 0 1"
INITIAL_STATE = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
NOTATION_RE = re.compile(
    r"^([NBRQKP])?([a-h])?([1-8])?(x)?([a-h])([1-8])(=?[nrbqNRBQ])?"
    r"(\+|#|\?|\!|(\s*[eE]\.?[pP]\.?))*$"
)
PIECE_VECTORS = dict(
    n=[(1, 2), (2, 1), (-1, 2), (-2, 1), (1, -2), (2, -1), (-1, -2), (-2, -1)],
    q=[(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, 1), (1, -1), (-1, -1)],
    k=[(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)],
    r=[(0, 1), (0, -1), (1, 0), (-1, 0)],
    b=[(1, 1), (-1, 1), (1, -1), (-1, -1)],
)
PIECE_ICONS = dict(
    [
        ("P", "♟"),
        ("R", "♜"),
        ("N", "♞"),
        ("B", "♝"),
        ("Q", "♛"),
        ("K", "♚"),
        ("p", "♙"),
        ("r", "♖"),
        ("n", "♘"),
        ("b", "♗"),
        ("q", "♕"),
        ("k", "♔"),
    ]
)


def cache_by_game_state(func):
    cache = functools.lru_cache(maxsize=1024)

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        state = self.get_state()
        key = (state, args, frozenset(kwargs.items()))

        @cache
        def cached_func(key):
            return func(self, *args, **kwargs)

        return cached_func(key)

    return wrapper


class Game:
    def __init__(self, state=INITIAL_STATE, players=None):
        self.players = players or [Human(), Computer()]
        self.history = []
        self.half_move_counter = 1
        self.status = None
        self.status_desc = None
        self._allow_king_capture = False
        self.set_state(state)

    def play(self):
        if not self.players or len(self.players) != 2:
            raise ValueError("Two players required")
        self.status = "PLAYING"
        while self.status == "PLAYING":
            move = self.cur_player.get_move(self)
            self.make_move(move)

    def make_moves(self, moves):
        for move in moves:
            self.make_move(move)

    def make_move(self, move):
        if type(move) == str:
            notation = move
            move = self.parse_notation(notation)
        else:
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
        captured = self.board[to_sq]
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
            self.castles = [c for c in self.castles if c.upper() != color]
        elif piece_type == "r" and from_sq in [(0, 0), (0, 7)]:
            self.castles = [c for c in self.castles if c != ("Q" if color else "q")]
        elif piece_type == "r" and from_sq in [(7, 0), (7, 7)]:
            self.castles = [c for c in self.castles if c != ("K" if color else "k")]

        # half-move and 50-move rule update
        if color == BLACK:
            self.half_move_counter += 1
            self.draw_counter += 1
        if (self.board[to_sq] is None and (piece_type != "p")) or "x" in notation:
            self.draw_counter = 0

        self.history.append(notation)
        self.turn = 1 - self.turn

        # is game over
        if self._allow_king_capture and captured == "k":
            self.status = "WWINS"
            self.status_desc = "King captured"
        elif self._allow_king_capture and captured == "K":
            self.status = "BWINS"
            self.status_desc = "King captured"
        elif self.is_checkmate():
            self.status = "BWINS" if self.cur_color else "WWINS"
            self.status_desc = "Checkmate"
        elif self.is_stalemate():
            self.status = "DRAW"
            self.status_desc = "Stalemate"
        elif self.draw_counter >= 50:
            self.status = "DRAW"
            self.status_desc = "Fifty-move rule"

        self.state = self.get_state()

    @cache_by_game_state
    def get_legal_moves(self):
        # first get opponent attacks and pins
        attacks, pin = self.get_attacks()

        # get legal moves for current player
        moves = []
        for from_square, piece in self.board.items():
            if piece is None or piece.isupper() != self.cur_color:
                continue
            from_col, from_row = from_square
            piece_type = piece.lower()
            piece_color = piece.isupper()
            vectors = PIECE_VECTORS.get(piece_type)
            if self._allow_king_capture:
                pinned = None
            else:
                pinned = pin and pin[0] == from_square
                pin_axis = pin[1] if pinned else None

            # adjust vectors based on pin (not relevant for pawn/king)
            if pinned and vectors:
                if pin_axis in [(0, 1), (0, -1)]:  # pinned to col
                    vectors = [v for v in vectors if v[1] == 0]
                elif pin_axis in [(1, 0), (-1, 0)]:  # pinned to row
                    vectors = [v for v in vectors if v[0] == 0]
                else:
                    vectors = set(vectors) & set(
                        [pin_axis, (-pin_axis[0], -pin_axis[1])]
                    )

            if piece_type == "p":
                row_dir = 1 if piece_color == WHITE else -1
                row_start = 1 if piece_color == WHITE else 6
                push1 = (from_col, from_row + row_dir)
                push2 = (from_col, from_row + (row_dir * 2))
                cap1 = (from_col - 1, from_row + row_dir)
                cap2 = (from_col + 1, from_row + row_dir)

                def _append_with_promos(m, **kwargs):
                    if m[1] in [0, 7]:
                        for promo in ["q", "r", "b", "n"]:
                            moves.append((from_square, m, dict(promo=promo, **kwargs)))
                    else:
                        moves.append((from_square, m, dict(**kwargs)))

                if not pinned or pin_axis[0] == 0:
                    if push1[1] <= 7 and push1[1] >= 0 and not self.board.get(push1):
                        _append_with_promos(push1)
                    if (
                        from_square[1] == row_start
                        and push2[1] <= 7
                        and push2[1] >= 0
                        and not self.board.get(push2)
                    ):
                        _append_with_promos(push2)
                if (
                    from_col > 0
                    and self.board.get(cap1)
                    and self.board.get(cap1).isupper() != piece_color
                    and not (pinned and pin_axis[0] != 1)
                ):
                    _append_with_promos(cap1)
                if (
                    from_col < 7
                    and self.board.get(cap2)
                    and self.board.get(cap2).isupper() != piece_color
                    and not (pinned and pin_axis[0] != -1)
                ):
                    _append_with_promos(cap2)

                if (
                    self.enpassant
                    and from_row == self.enpassant[1]
                    and abs(from_col - self.enpassant[0]) == 1
                ):
                    enp_to = (self.enpassant[0], from_row + row_dir)
                    moves.append((from_square, enp_to, dict(enp=True)))

            elif piece_type == "r":
                moves += self.get_moves_by_vectors(from_square, vectors)

            elif piece_type == "q":
                moves += self.get_moves_by_vectors(from_square, vectors)

            elif piece_type == "b":
                moves += self.get_moves_by_vectors(from_square, vectors)

            elif piece_type == "n":
                for col_dir, row_dir in vectors:
                    col, row = from_col + col_dir, from_row + row_dir
                    to_square = (col, row)
                    to_piece = self.board.get(to_square)
                    if 0 <= col < 8 and 0 <= row < 8:
                        if not to_piece or to_piece.isupper() != piece_color:
                            moves.append((from_square, (col, row), dict()))

            elif piece_type == "k":
                for col_dir, row_dir in vectors:
                    col, row = from_col + col_dir, from_row + row_dir
                    if 0 <= col < 8 and 0 <= row < 8:
                        to_square = (col, row)
                        to_piece = self.board.get(to_square)
                        if (not to_piece or to_piece.isupper() != piece_color) and (
                            self._allow_king_capture or (to_square not in attacks)
                        ):
                            moves.append((from_square, (col, row), dict()))

        # add castles
        row = 0 if self.cur_color else 7
        ks = "K" if self.cur_color else "k"
        if (
            (ks in self.castles)
            and (not self.board.get((5, row)))
            and (not self.board.get((6, row)))
            and (self._allow_king_capture or (not (5, row) in attacks))
            and (self._allow_king_capture or (not (6, row) in attacks))
        ):
            moves.append(((4, row), (6, row), dict(castle=ks)))

        qs = "Q" if self.cur_color else "q"
        if (
            (qs in self.castles)
            and (not self.board.get((1, row)))
            and (not self.board.get((2, row)))
            and (not self.board.get((3, row)))
            and (self._allow_king_capture or (not (1, row) in attacks))
            and (self._allow_king_capture or (not (2, row) in attacks))
            and (self._allow_king_capture or (not (3, row) in attacks))
        ):
            moves.append(((4, row), (2, row), dict(castle=qs)))

        return moves

    @cache_by_game_state
    def get_attacks(self):
        attacks, pin = [], None
        color = not self.cur_color

        for from_square, piece in self.board.items():
            if piece is None or piece.isupper() != color:
                continue
            from_col, from_row = from_square
            piece_type = piece.lower()
            vectors = PIECE_VECTORS.get(piece_type)

            if piece_type == "p":
                row_dir = 1 if color == WHITE else -1
                if from_col > 0:
                    cap1 = (from_col - 1, from_row + row_dir)
                    attacks.append(cap1)
                if from_col < 7:
                    cap2 = (from_col + 1, from_row + row_dir)
                    attacks.append(cap2)

            elif piece_type == "r":
                r_attacks, r_pin = self.get_attacks_by_vectors(from_square, vectors)
                attacks += r_attacks
                pin = pin or r_pin

            elif piece_type == "q":
                q_attacks, q_pin = self.get_attacks_by_vectors(from_square, vectors)
                attacks += q_attacks
                pin = pin or q_pin

            elif piece_type == "b":
                b_attacks, b_pin = self.get_attacks_by_vectors(from_square, vectors)
                attacks += b_attacks
                pin = pin or b_pin

            elif piece_type == "n":
                for dcol, drow in vectors:
                    col, row = from_col + dcol, from_row + drow
                    if 0 <= col < 8 and 0 <= row < 8:
                        attacks.append((col, row))

            elif piece_type == "k":
                for dcol, drow in vectors:
                    col, row = from_col + dcol, from_row + drow
                    if 0 <= col < 8 and 0 <= row < 8:
                        attacks.append((col, row))

        return set(attacks), pin

    def get_attacks_by_vectors(self, from_square, vectors):
        attacks = []
        pin = None
        color = self.board.get(from_square).isupper()

        for col_dir, row_dir in vectors:
            looking_for_pin = False
            maybe_pinned = None
            to_col, to_row = from_square
            while True:
                to_col += col_dir
                to_row += row_dir
                to_square = (to_col, to_row)
                if not ((0 <= to_col < 8) and (0 <= to_row < 8)):
                    break
                if not looking_for_pin:
                    attacks.append(to_square)
                attacked_piece = self.board.get(to_square)
                if attacked_piece is None:
                    continue
                if attacked_piece.isupper() == color:
                    # TODO: handle the dreaded enpassant-pin
                    break
                if attacked_piece.lower() == "k":
                    if looking_for_pin:
                        pin = maybe_pinned
                    break
                if looking_for_pin:
                    break
                maybe_pinned = (to_square, (col_dir, row_dir))
                looking_for_pin = True

        return attacks, pin

    def get_moves_by_vectors(self, from_square, vectors):
        moves = []
        for row_dir, col_dir in vectors:
            piece_color = self.board[from_square].isupper()
            col, row = from_square
            while True:
                row += row_dir
                col += col_dir
                to_square = (col, row)
                if row < 0 or row > 7 or col < 0 or col > 7:
                    break
                if not self.board.get(to_square):
                    moves.append((from_square, to_square, dict()))
                elif self.board.get(to_square).isupper() != piece_color:
                    moves.append((from_square, to_square, dict()))
                    break
                else:
                    break

        return moves

    def is_legal_move(self, move):
        if type(move) == str:
            move = self.parse_notation(move)
        if len(move) > 2 and "action" in move[2]:
            return True
        for lm in self.get_legal_moves():
            if move[0] == lm[0] and move[1] == lm[1]:
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
            legal = self.get_legal_moves()
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
        promo = props.get("promo")

        return (
            f"{c2sq((col_from, row_from))}{sep}"
            f"{c2sq((col_to, row_to))}"
            f"{'=' + promo.lower() if promo else ''}"
        )

    @cache_by_game_state
    def is_check(self):
        attacks, _ = self.get_attacks()
        king = "K" if self.cur_color else "k"
        return any([self.board.get(sq) == king for sq in attacks])

    def is_checkmate(self):
        return self.is_check() and not len(self.get_legal_moves())

    def is_stalemate(self):
        return not self.is_check() and not len(self.get_legal_moves())

    def set_state(self, state_str):
        board_str, turn, castles, enpassant, draw_counter, move_counter = (
            state_str.split() + [None] * 5
        )[:6]
        if turn:
            self.turn = 0 if turn == "w" else 1
        if castles and castles != "-":
            self.castles = list(castles)
        else:
            self.castles = []
        if enpassant and (enpassant != "-"):
            enpassant = enpassant.lower() if enpassant else None
            self.enpassant = sq2c(enpassant)
        else:
            self.enpassant = None
        if draw_counter:
            self.draw_counter = int(draw_counter)
        if move_counter:
            self.move_counter = int(move_counter)

        board = collections.defaultdict(lambda: None)
        rows = board_str.split("/")
        for row_index, row in enumerate(rows):
            col_index = 0
            for char in row:
                if char.isdigit():
                    col_index += int(char)
                else:
                    board[(col_index, 7 - row_index)] = char
                    col_index += 1
        self.board = board
        self.state = state_str

    def get_state(self):
        board_str = self.board_to_fen(self.board)
        state_str = (
            f"{board_str} {'w' if self.cur_color == WHITE else 'b'}"
            f" {''.join(self.castles) if self.castles else '-'}"
            f" {c2sq(self.enpassant) if self.enpassant else '-'}"
            f" {self.draw_counter} {self.half_move_counter}"
        )
        return state_str

    def render_board(self, color=None, clear=False, icons=False):
        color = color if color is not None else self.cur_color
        if clear:
            os.system("clear")
        row_range = range(7, -1, -1) if color == WHITE else range(8)
        print()
        for row in row_range:
            for col in range(8):
                piece = self.board.get((col, row), "-")
                piece = PIECE_ICONS.get(piece, "-") if icons else piece
                print(piece, end=" ")
            print()

    def set_board(self, fen):
        self.set_state(f"{fen} {' '.join(self.state.split()[1:])}")

    @cache_by_game_state
    def is_legal_state(self):
        if len([p for p in self.board.values() if p in "kK"]) != 2:
            return False
        # player can't be in check if not their turn
        if not self._allow_king_capture:
            g = Game(state=self.get_state())
            g.turn = 1 - self.turn
            if g.is_check():
                return False
        return True

    def randomize_board(self):
        while True:
            pieces = ["p", "n", "b", "r", "q", "P", "N", "B", "R", "Q"]
            board = ["1"] * 64
            king_positions = random.sample(range(64), 2)
            board[king_positions[0]] = "K"
            board[king_positions[1]] = "k"
            for i in range(64):
                if board[i] == "1" and random.random() < 0.5:
                    board[i] = random.choice(pieces)
            board = {(i % 8, 7 - i // 8): board[i] for i in range(64)}
            fen = self.board_to_fen(board)
            g = Game()
            g.set_board(fen)
            if g.is_legal_state():
                self.set_board(fen)
                break

    def board_to_fen(self, board):
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
        return board_str

    @property
    def cur_color(self):
        return WHITE if self.turn == 0 else BLACK

    @property
    def cur_player(self):
        return self.players[self.turn]

    @property
    def last_move(self):
        return self.history[-1] if len(self.history) else None


class Player:
    def get_move(self, game):
        raise NotImplementedError


class Computer(Player):
    def get_move(self, game):
        moves = game.get_legal_moves()
        return random.choice(moves)


class Human(Player):
    def get_move(self, game):
        game.render_board(clear=True, icons=True)

        # print last move if present
        if game.last_move:
            print(f"[{'B' if game.cur_color else 'W'}]", game.last_move)

        move = None
        while not move:
            prompt = f"[{'W' if game.cur_color else 'B'}] => "
            val = input(prompt).strip()
            if val in ["quit", "exit"]:
                sys.exit(0)
            elif val in ["legal", "l"]:
                legal_moves = game.get_legal_moves()
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
            elif not game.is_legal_move(move):
                print("Illegal move!")
                move = None
        return move


def sq2c(sq):
    """a1 -> (0, 0)"""
    return ord(sq[0]) - ord("a"), int(sq[1]) - 1


def c2sq(coords):
    """(0, 0) -> a1"""
    return chr(ord("a") + coords[0]) + str(coords[1] + 1)


if __name__ == "__main__":
    players = (Human(), Computer())
    if "-b" in sys.argv:
        players = tuple(reversed(players))
    game = Game(players=players)
    game.play()

    # finished:
    game.render_board(clear=True, icons=True)
    desc = dict(WWINS="White wins", BWINS="Black wins", DRAW="Draw")[game.status]
    print(f"\n{desc}: {game.status_desc}!")
