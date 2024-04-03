#!/usr/bin/env python3.11

import random
import sys
import functools
import argparse
import math
import collections
from typing import Optional, Callable

# additional types
Coords = tuple[int, int]  # col, row
Pin = tuple[Coords, Coords]  # sq of pinned piece, axis
Attack = tuple[Coords, Coords]  # sq of attacking piece, sq of attacked piece
Check = tuple[Coords, Coords]  # sq of attacking piece, axis
PieceVectors = list[tuple[int, int]]
Move = tuple[Coords, Coords, str]

WHITE, BLACK = True, False
SCORES = dict(p=10, n=30, b=35, r=50, q=90, k=0)
VECTORS: dict[str, PieceVectors] = dict(
    n=[(1, 2), (2, 1), (-1, 2), (-2, 1), (1, -2), (2, -1), (-1, -2), (-2, -1)],
    q=[(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, 1), (1, -1), (-1, -1)],
    k=[(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)],
    r=[(0, 1), (0, -1), (1, 0), (-1, 0)],
    b=[(1, 1), (-1, 1), (1, -1), (-1, -1)],
)


def cache_by_fen(func: Callable) -> Callable:
    cache = functools.lru_cache(maxsize=10000)

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        fen = self.fen or self.get_fen()
        key = (fen, args, frozenset(kwargs.items()))

        @cache
        def cached_func(key):
            return func(self, *args, **kwargs)

        return cached_func(key)

    return wrapper


class Game:
    EMPTY = "8/8/8/8/8/8/8/8 w KQkq - 0 1"
    STANDARD = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

    board: "Board"
    castles: list[str]
    turn: bool
    enpassant: Optional[Coords]
    draw_counter: int
    status: tuple[str, str]
    allow_king_capture: bool = False
    repititions: dict[str, int]
    history: list[str]

    def __init__(
        self,
        fen=STANDARD,
    ) -> None:
        self.history = []
        self.status = ("playing", "")
        self.set_fen(fen)

    def make_move(self, move: str) -> None:
        from_, to, promo = move2c(move)
        piece = self.board[from_]
        piece_type = piece.lower()
        color = piece.isupper()
        promo = promo.upper() if color else promo.lower()
        is_enp = piece.lower() == "p" and not self.board[to] and from_[0] != to[0]
        is_capture = is_enp or self.board[to]

        # move
        self.board[to] = promo if promo else self.board[from_]
        self.board[from_] = ""

        # complete enpassant
        if is_enp and self.enpassant:
            self.board[self.enpassant] = ""

        # mark enpassantable pawn
        self.enpassant = (
            to if piece_type == "p" and abs(from_[1] - to[1]) == 2 else None
        )

        # complete castle
        if piece_type == "k" and from_[0] == 4 and to[0] == 6:
            self.board[(5, 0 if color else 7)] = "R" if color else "r"
            self.board[(7, 0 if color else 7)] = ""
        elif piece_type == "k" and from_[0] == 4 and to[0] == 2:
            self.board[(3, 0 if color else 7)] = "R" if color else "r"
            self.board[(0, 0 if color else 7)] = ""

        # update castles available
        if piece_type == "k":
            self.castles = [c for c in self.castles if c.isupper() != color]
        elif piece_type == "r" and from_ in [(0, 0), (0, 7)]:
            self.castles = [c for c in self.castles if c != ("Q" if color else "q")]
        elif piece_type == "r" and from_ in [(7, 0), (7, 7)]:
            self.castles = [c for c in self.castles if c != ("K" if color else "k")]

        # flip turn
        self.turn = not self.turn

        # update state
        self.history.append(move)
        self.fen = self.get_fen()

        # 50-move rule update
        if piece_type == "p" or is_capture:
            self.draw_counter = 0
        else:
            self.draw_counter += 1

        # 3-fold repetition update
        fen = self.fen.split()[0]
        self.repititions[fen] = self.repititions.get(fen, 0) + 1
        self.check_game_over()

    def check_game_over(self) -> bool:
        board_fen = self.fen.split()[0]

        if self.allow_king_capture and "k" not in board_fen:
            self.status = ("wwins", "king_captured")
        elif self.allow_king_capture and "K" not in board_fen:
            self.status = ("bwins", "king_captured")
        elif self.is_checkmate():
            self.status = ("bwins" if self.turn else "wwins", "checkmate")
        elif self.is_stalemate():
            self.status = ("draw", "stalemate")
        elif self.draw_counter >= 50:
            self.status = ("draw", "fifty_move")
        elif self.repititions[board_fen] >= 3:
            self.status = ("draw", "threefold_rep")

        # insufficient material
        material = "".join(sorted([str(p) for p in self.board.values() if p]))
        if material in ["Kk", "KNk", "Kkn", "BKk", "Kbk"]:
            self.status = ("draw", "insuff_material")
        if material == "BKbk":
            bishops = [sq for sq, p in self.board.items() if p and p.lower() == "b"]
            if sq_color(bishops[0]) == sq_color(bishops[1]):
                self.status = ("draw", "insuff_material")

        return self.is_ended

    @cache_by_fen
    def get_legal_moves(self) -> list[str]:
        # first get opponent attacks, pins, and checks
        attacks, pin = self.get_attacks()
        attacked_squares = [a[1] for a in attacks]
        checks = self.get_checks()
        is_check = len(checks) > 0
        double_check = len(checks) > 1
        if is_check:
            check_from, check_axis = checks[0]
        else:
            check_from, check_axis = None, None

        # get legal moves for current player
        moves: list[Move] = []

        for from_square, piece in self.board.items():
            piece_type = piece.lower()
            if not piece or (piece.isupper() != self.turn):
                continue
            if double_check and not self.allow_king_capture and piece_type != "k":
                continue

            from_col, from_row = from_square

            # check if piece is pinned (ignore if allowed to capture king)
            if self.allow_king_capture:
                is_pinned, pin_axis = False, None
            else:
                is_pinned = pin and pin[0] == from_square
                pin_axis = pin[1] if is_pinned else None

            # adjust vectors based on pin (not relevant for pawn/king)
            vectors = VECTORS.get(piece_type, [])
            if is_pinned and vectors and pin_axis:
                if pin_axis in [(0, 1), (0, -1)]:  # pinned to col
                    vectors = [v for v in vectors if v[1] == 0]
                elif pin_axis in [(1, 0), (-1, 0)]:  # pinned to row
                    vectors = [v for v in vectors if v[0] == 0]
                else:
                    vectors = list(
                        set(vectors) & set([pin_axis, (-pin_axis[0], -pin_axis[1])])
                    )

            if piece_type == "k":
                for col_dir, row_dir in vectors:
                    if (col_dir, row_dir) == check_axis:
                        continue
                    col, row = from_col + col_dir, from_row + row_dir
                    if 0 <= col < 8 and 0 <= row < 8:
                        to_square = (col, row)
                        to_piece = self.board.get(to_square, "")
                        sq_avail = not to_piece or (
                            to_piece.isupper() != piece.isupper()
                        )
                        sq_attacked = to_square in attacked_squares
                        if sq_avail and (self.allow_king_capture or not sq_attacked):
                            moves.append((from_square, (col, row), ""))

            elif piece_type == "p":
                row_dir = 1 if piece.isupper() else -1
                row_start = 1 if piece.isupper() else 6
                push1 = (from_col, from_row + row_dir)
                push2 = (from_col, from_row + (row_dir * 2))
                cap1 = (from_col - 1, from_row + row_dir)
                cap2 = (from_col + 1, from_row + row_dir)

                def _append_with_promos(to):
                    if to[1] in [0, 7]:
                        for promo in ["q", "r", "b", "n"]:
                            moves.append((from_square, to, promo))
                    else:
                        moves.append((from_square, to, ""))

                if not is_pinned or (pin_axis and pin_axis[0] == 0):
                    if push1[1] <= 7 and push1[1] >= 0 and not self.board.get(push1):
                        _append_with_promos(push1)
                    if (
                        from_square[1] == row_start
                        and push2[1] <= 7
                        and push2[1] >= 0
                        and not self.board.get(push1)
                        and not self.board.get(push2)
                    ):
                        _append_with_promos(push2)
                if (
                    from_col > 0
                    and self.board.get(cap1)
                    and self.board.get(cap1, "").isupper() != piece.isupper()
                    and not (is_pinned and pin_axis and (pin_axis[0] != 1))
                ):
                    _append_with_promos(cap1)
                if (
                    from_col < 7
                    and self.board.get(cap2)
                    and self.board.get(cap2, "").isupper() != piece.isupper()
                    and not (is_pinned and pin_axis and (pin_axis[0] != -1))
                ):
                    _append_with_promos(cap2)

                if (
                    self.enpassant
                    and from_row == self.enpassant[1]
                    and abs(from_col - self.enpassant[0]) == 1
                ):
                    enp_to = (self.enpassant[0], from_row + row_dir)
                    moves.append((from_square, enp_to, ""))

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
                    to_piece = self.board.get(to_square, "")
                    if 0 <= col < 8 and 0 <= row < 8:
                        if not to_piece or to_piece.isupper() != piece.isupper():
                            moves.append((from_square, (col, row), ""))

        # filter out moves that don't resolve check
        if is_check and not self.allow_king_capture:
            intercepts: list[Coords] = []
            if self.board.get(check_from, "").lower() in ["r", "q", "b"]:
                sq = check_from
                while True:
                    sq = sq[0] + check_axis[0], sq[1] + check_axis[1]
                    if self.board.get(sq, "").lower() == "k":
                        break
                    intercepts.append(sq)

            moves = [
                m
                for m in moves
                if (
                    self.board.get(m[0], "").lower() == "k"
                    or m[1] == check_from
                    or m[1] in intercepts
                )
            ]

        # add castles unless check (even with king capture)
        if not is_check:
            row = 0 if self.turn else 7
            kside = "K" if self.turn else "k"
            if (
                (kside in self.castles)
                and (not self.board.get((5, row)))
                and (not self.board.get((6, row)))
                and (self.allow_king_capture or (not (5, row) in attacked_squares))
                and (self.allow_king_capture or (not (6, row) in attacked_squares))
            ):
                moves.append(((4, row), (6, row), ""))

            qside = "Q" if self.turn else "q"
            if (
                (qside in self.castles)
                and (not self.board.get((1, row)))
                and (not self.board.get((2, row)))
                and (not self.board.get((3, row)))
                and (self.allow_king_capture or (not (1, row) in attacked_squares))
                and (self.allow_king_capture or (not (2, row) in attacked_squares))
                and (self.allow_king_capture or (not (3, row) in attacked_squares))
            ):
                moves.append(((4, row), (2, row), ""))

        return [c2move(m) for m in moves]

    @cache_by_fen
    def get_attacks(self) -> tuple[list[Attack], Optional[Pin]]:
        attacks: list[Attack] = []
        pin: Optional[Pin] = None
        color = not self.turn

        for from_square, piece in self.board.items():
            if not piece or (piece.isupper() != color):
                continue
            from_col, from_row = from_square
            piece_type = piece.lower()
            vectors = VECTORS.get(piece_type, [])

            if piece_type == "p":
                row_dir = 1 if color == WHITE else -1
                if from_col > 0:
                    cap1 = (from_col - 1, from_row + row_dir)
                    attacks.append((from_square, cap1))
                if from_col < 7:
                    cap2 = (from_col + 1, from_row + row_dir)
                    attacks.append((from_square, cap2))

            elif piece_type == "r":
                r_attacks, r_pin = self.get_attacks_by_vectors(from_square, vectors)
                attacks.extend(r_attacks)
                pin = pin or r_pin

            elif piece_type == "q":
                q_attacks, q_pin = self.get_attacks_by_vectors(from_square, vectors)
                attacks.extend(q_attacks)
                pin = pin or q_pin

            elif piece_type == "b":
                b_attacks, b_pin = self.get_attacks_by_vectors(from_square, vectors)
                attacks.extend(b_attacks)
                pin = pin or b_pin

            elif piece_type == "n":
                for dcol, drow in vectors:
                    col, row = from_col + dcol, from_row + drow
                    if 0 <= col < 8 and 0 <= row < 8:
                        attacks.append((from_square, (col, row)))

            elif piece_type == "k":
                for dcol, drow in vectors:
                    col, row = from_col + dcol, from_row + drow
                    if 0 <= col < 8 and 0 <= row < 8:
                        attacks.append((from_square, (col, row)))

        return attacks, pin

    def get_attacks_by_vectors(
        self, from_square: Coords, vectors: PieceVectors
    ) -> tuple[list[Attack], Optional[Pin]]:
        attacks: list[Attack] = []
        pin: Optional[Pin] = None
        piece = self.board[from_square]
        color = piece.isupper()

        for col_dir, row_dir in vectors:
            looking_for_pin = False
            maybe_pinned: Optional[Pin] = None
            to_col, to_row = from_square

            while True:
                to_col += col_dir
                to_row += row_dir
                to_square = (to_col, to_row)
                if not ((0 <= to_col < 8) and (0 <= to_row < 8)):
                    break

                if not looking_for_pin:
                    attacks.append((from_square, to_square))
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

    def get_moves_by_vectors(self, from_square, vectors) -> list["Move"]:
        moves: list[Move] = []
        for row_dir, col_dir in vectors:
            piece = self.board[from_square]
            col, row = from_square
            while True:
                row += row_dir
                col += col_dir
                to_square = (col, row)
                if row < 0 or row > 7 or col < 0 or col > 7:
                    break
                if not self.board.get(to_square):
                    moves.append((from_square, to_square, ""))
                elif self.board[to_square].isupper() != piece.isupper():
                    moves.append((from_square, to_square, ""))
                    break
                else:
                    break

        return moves

    def is_legal_move(self, move: str) -> bool:
        move = move.replace(r"x", "").replace(r"-", "").strip()
        return move in self.get_legal_moves()

    def is_check(self) -> bool:
        return len(self.get_checks()) > 0

    @cache_by_fen
    def get_checks(self) -> list[Check]:
        attacks, _ = self.get_attacks()
        checks: list[Check] = []
        for from_square, attacked_square in attacks:
            attacked_piece = self.board.get(attacked_square)
            if (
                attacked_piece
                and attacked_piece.lower() == "k"
                and attacked_piece.isupper() == self.turn
            ):
                a0 = attacked_square[0] - from_square[0]
                a1 = attacked_square[1] - from_square[1]
                axis = (
                    -1 if a0 < 0 else (1 if a0 > 0 else 0),
                    -1 if a1 < 0 else (1 if a1 > 0 else 0),
                )
                checks.append((from_square, axis))
        return checks

    def is_checkmate(self) -> bool:
        return self.is_check() and not len(self.get_legal_moves())

    def is_stalemate(self) -> bool:
        return not self.is_check() and not len(self.get_legal_moves())

    def set_fen(self, fen_str: str) -> None:
        board_str, turn, castles, enpassant, draw_counter, full_move_counter = (
            fen_str.split() + [""] * 5
        )[:6]
        if turn:
            self.turn: bool = WHITE if turn == "w" else BLACK
        if castles and castles != "-":
            self.castles = list(castles)  # type: ignore
        else:
            self.castles = []
        if enpassant and (enpassant != "-"):
            enpassant = enpassant.lower()
            self.enpassant = sq2c(enpassant)
        else:
            self.enpassant = None
        if draw_counter:
            self.draw_counter = int(draw_counter)
        if full_move_counter:
            self.history: list[str] = [""] * ((int(full_move_counter) - 1) * 2)

        self.board: Board = Board.from_fen(board_str)
        self.fen = fen_str
        self.repititions: dict[str, int] = {board_str: 1}
        self.check_game_over()

    def set_board(self, fen: str) -> None:
        self.set_fen(f"{fen} {' '.join(self.fen.split()[1:])}")

    def get_fen(self) -> str:
        board_str = self.board.to_fen()
        fen_str = (
            f"{board_str} {'w' if self.turn == WHITE else 'b'}"
            f" {''.join(self.castles) if self.castles else '-'}"
            f" {c2sq(self.enpassant) if self.enpassant else '-'}"
            f" {self.draw_counter} {(len(self.history) // 2) or 1}"
        )
        return fen_str

    def print_board(self, color: Optional[bool] = None) -> None:
        color = color if color is not None else WHITE
        range_dir = range(7, -1, -1) if color == WHITE else range(8)
        col_dir = range(7, -1, -1) if color == BLACK else range(8)
        print()
        for row in range_dir:
            for col in col_dir:
                piece = self.board.get((col, row)) or "-"
                print(piece, end=" ")
            print()

    @cache_by_fen
    def get_position_score(self) -> float:
        if self.status[0] == "wwins":
            return math.inf
        if self.status[0] == "bwins":
            return -math.inf
        if self.status[0] == "draw":
            return -0.01

        wscore, bscore = 0.0, 0.0
        for piece in self.board.values():
            if not piece:
                continue
            if piece.isupper() == WHITE:
                wscore += SCORES.get(piece.lower(), 0)
            else:
                bscore += SCORES.get(piece.lower(), 0)
        return wscore - bscore

    def lookahead(self, move):
        g = Game(fen=self.fen)
        g.make_move(move)
        return g

    @property
    def last_move(self) -> Optional[str]:
        return self.history[-1] if len(self.history) else None

    @property
    def is_ended(self) -> bool:
        return self.status[0] in ("wwins", "bwins", "draw")

    @property
    def status_str(self):
        return " ".join(self.status)


class Board(collections.defaultdict):
    def __init__(self, board: Optional[dict] = None) -> None:
        super().__init__(lambda: "")
        if board:
            self.update(board)

    def to_fen(self) -> str:
        board_str = ""
        for row_index in range(7, -1, -1):
            empty_count = 0
            for col_index in range(8):
                piece = self.get((col_index, row_index))
                if piece:
                    if empty_count > 0:
                        board_str += str(empty_count)
                        empty_count = 0
                    board_str += str(piece)
                else:
                    empty_count += 1
            if empty_count > 0:
                board_str += str(empty_count)
            if row_index > 0:
                board_str += "/"
        return board_str

    @classmethod
    def from_fen(cls, fen: str) -> "Board":
        board = dict()
        rows = fen.split("/")
        for row_index, row in enumerate(rows):
            col_index = 0
            for char in row:
                if char.isdigit():
                    col_index += int(char)
                else:
                    if char not in "pnbrqkPNBRQK":
                        raise ValueError("Invalid FEN piece")
                    board[(col_index, 7 - row_index)] = char
                    col_index += 1
        return cls(board=board)


class Computer:
    lookahead_moves: int = 1
    random_move: bool = False

    def get_move(self, game: Game) -> str:
        moves = game.get_legal_moves()
        if self.random_move:
            return random.choice(moves)
        scores = [(m, self.evaluate_move(m, game)) for m in moves]
        max_score = max(scores, key=lambda x: x[1])[1]
        best_moves = [m for m, score in scores if score == max_score]
        move = random.choice(best_moves)
        return move

    def evaluate_move(self, move: str, game: Game) -> float:
        maximizing = not game.turn
        score = self.minimax(
            game.lookahead(move), self.lookahead_moves, maximizing=maximizing
        )
        score = -score if maximizing else score
        # print(f"Move: {move} => Score: {score}")
        return score

    def minimax(
        self,
        game: Game,
        depth: int,
        alpha: float = -math.inf,
        beta: float = math.inf,
        maximizing: bool = False,
    ) -> float:
        if depth == 0 or game.is_ended:
            score = game.get_position_score()
            return score

        if maximizing:
            max_eval = -math.inf
            for move in game.get_legal_moves():
                eval = self.minimax(game.lookahead(move), depth - 1, alpha, beta, False)
                if eval > max_eval:
                    max_eval = eval
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = math.inf
            for move in game.get_legal_moves():
                eval = self.minimax(game.lookahead(move), depth - 1, alpha, beta, True)
                if eval < min_eval:
                    min_eval = eval
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval


def sq2c(sq: str) -> Coords:
    """a1 -> (0, 0)"""
    return ord(sq[0]) - ord("a"), int(sq[1]) - 1


def c2sq(coords: Coords) -> str:
    """(0, 0) -> a1"""
    return chr(ord("a") + coords[0]) + str(coords[1] + 1)


def move2c(move: str) -> Move:
    """e2e4=q -> ((4, 1), (4, 3), 'q')"""
    move = move.replace(r"x", "").replace(r"-", "")
    from_, to = sq2c(move[:2]), sq2c(move[2:])
    promo = move[5] if "=" in move else ""
    return from_, to, promo


def c2move(move: Move) -> str:
    """((4, 1), (4, 3), 'q') -> e2e4=q"""
    cfrom, cto, promo = move
    return f"{c2sq(cfrom)}{c2sq(cto)}{'=' + promo if promo else ''}"


def sq_color(c: Coords):
    return WHITE if (c[0] + c[1]) % 2 else BLACK


def self_play(
    n: int,
    allow_king_capture: bool = False,
    lookahead: Optional[int] = None,
    use_random: bool = False,
) -> None:
    from tqdm import tqdm  # type: ignore

    outcomes: dict[str, int] = dict()

    for _ in tqdm(range(n)):
        c1, c2 = Computer(), Computer()
        if use_random:
            c1.random_move = True
            c2.random_move = True
        elif lookahead is not None:
            c1.lookahead_moves = lookahead
            c2.lookahead_moves = lookahead
        game = Game()
        game.allow_king_capture = allow_king_capture
        while not game.is_ended:
            player = c1 if game.turn == WHITE else c2
            move = player.get_move(game)
            game.make_move(move)
        outcomes[game.status_str] = outcomes.get(game.status_str, 0) + 1
    for o, ct in outcomes.items():
        print(f"{o}: {ct}")


def main() -> None:
    argp = argparse.ArgumentParser()
    argp.add_argument("--play", "-p", type=int, default=10)
    argp.add_argument("--lookahead", "-l", type=int, default=None)
    argp.add_argument("--random", "-r", action="store_true")
    argp.add_argument("--allow-king-capture", "-k", action="store_true")
    options = argp.parse_args()

    self_play(
        options.play,
        allow_king_capture=options.allow_king_capture,
        lookahead=options.lookahead,
        use_random=options.random,
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
