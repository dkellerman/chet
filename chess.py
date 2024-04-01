#!/usr/bin/env python3.11

import re
import random
import sys
import os
import functools
import collections
import enum
import argparse
import uuid
import abc
from typing import Union, Optional, Callable


WHITE, BLACK = True, False


class Status(enum.Enum):
    PLAYING = "Playing"
    WWINS = "White wins"
    BWINS = "Black wins"
    DRAW = "Draw"


class Action(enum.Enum):
    RESIGN = "resign"


# additional types
Players = tuple["Player", "Player"]
Promo = str
Coords = tuple[int, int]  # col, row
BoardType = dict[Coords, Optional["Piece"]]
Square = str  # a1
PieceType = str
Castle = str
Pin = tuple[Coords, Coords]  # square of pinned piece, axis
Attack = tuple[Coords, Coords]  # square of attacking piece, square of attacked piece
Check = tuple[Coords, Coords]  # square of attacking piece, axis
PieceVectors = list[tuple[int, int]]


def cache_by_fen(func: Callable):
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

    id: Optional[str]
    players: Players
    history: list[str]
    castles: list[Castle]
    enpassant: Optional[Coords]
    status: Optional[Status]
    status_desc: Optional[str]
    turn: int
    allow_king_capture: bool
    draw_counter: int
    repititions: dict[str, int]

    def __init__(
        self,
        fen=STANDARD,
        players: Optional[Players] = None,
        id: Optional[str] = None,
        rnd_pieces: Optional[int] = None,
    ) -> None:
        self.id = id or uuid.uuid4().hex
        self.players = players or (Human(), Computer())
        self.history = []
        self.status = None
        self.status_desc = None
        self.allow_king_capture = False
        if rnd_pieces:
            self.set_fen(self.EMPTY)
            self.randomize_board(rnd_pieces)
        else:
            self.set_fen(fen)

    def play(self) -> None:
        if not self.players or len(self.players) != 2:
            raise ValueError("Two players required")
        self.status = Status.PLAYING
        while self.status == Status.PLAYING:
            move = self.cur_player.get_move(self)
            self.make_move(move)

    def make_moves(self, moves: list["Move"]) -> None:
        for move in moves:
            self.make_move(move)

    def make_move(self, val: Union["Move", str]):
        move: Optional[Move] = None
        notation: str = ""
        if isinstance(val, str):
            notation = val
            move = Move.parse(notation, self)
        elif isinstance(val, Move):
            move = val
            if move:
                notation = move.to_notation(self)

        if not move:
            raise ValueError("Invalid move")

        if move.action:
            if move.action == Action.RESIGN:
                self.status = Status.BWINS if self.cur_color else Status.WWINS
                self.status_desc = "Resignation"
            self.history.append(notation)
            return

        if not move.from_ or not move.to:
            raise ValueError("Invalid move")

        piece = self.board[move.from_]
        piece_type = piece.type if piece else None
        color = piece.color if piece else None
        promo: Optional[Promo] = (
            move.promo.upper() if move.promo and self.cur_color else move.promo  # type: ignore
        )

        # move
        self.board[move.to] = Piece(promo) if promo else self.board[move.from_]
        self.board[move.from_] = None

        # mark enpassantable pawn
        self.enpassant = (
            move.to
            if piece_type == "p" and abs(move.from_[1] - move.to[1]) == 2
            else None
        )

        # complete castle
        if piece_type == "k" and move.from_[0] == 4 and move.to[0] == 6:
            self.board[(5, 0 if color else 7)] = Piece("R" if color else "r")
            self.board[(7, 0 if color else 7)] = None
        elif piece_type == "k" and move.from_[0] == 4 and move.to[0] == 2:
            self.board[(3, 0 if color else 7)] = Piece("R" if color else "r")
            self.board[(0, 0 if color else 7)] = None

        # update castles available
        if piece_type == "k":
            self.castles = [c for c in self.castles if c.upper() != color]
        elif piece_type == "r" and move.from_ in [(0, 0), (0, 7)]:
            self.castles = [c for c in self.castles if c != ("Q" if color else "q")]
        elif piece_type == "r" and move.from_ in [(7, 0), (7, 7)]:
            self.castles = [c for c in self.castles if c != ("K" if color else "k")]

        # flip turn
        self.history.append(notation)
        self.turn = 1 - self.turn
        self.fen = self.get_fen()

        # 50-move rule update
        self.draw_counter += 1
        if (self.board[move.to] is None and (piece_type != "p")) or "x" in notation:
            self.draw_counter = 0

        # 3-fold repetition update
        fen = self.fen.split()[0]
        self.repititions[fen] = self.repititions.get(fen, 0) + 1
        self.check_game_over()

    def check_game_over(self) -> bool:
        fen = self.fen.split()[0]

        if self.allow_king_capture and "k" not in fen:
            self.status = Status.WWINS
            self.status_desc = "King captured"
        elif self.allow_king_capture and "K" not in fen:
            self.status = Status.BWINS
            self.status_desc = "King captured"
        elif self.is_checkmate():
            self.status = Status.BWINS if self.cur_color == WHITE else Status.WWINS
            self.status_desc = "Checkmate"
        elif self.is_stalemate():
            self.status = Status.DRAW
            self.status_desc = "Stalemate"
        elif self.draw_counter >= 50:
            self.status = Status.DRAW
            self.status_desc = "Fifty-move rule"
        elif self.repititions[fen] >= 3:
            self.status = Status.DRAW
            self.status_desc = "Threefold repetition"

        # insufficient material
        material = "".join(sorted([str(p) for p in self.board.values() if p]))
        if material in ["Kk", "KNk", "Kkn", "BKk", "Kbk"]:
            self.status = Status.DRAW
            self.status_desc = "Insufficient material"
        if material == "BKbk":
            bishops = [sq for sq, p in self.board.items() if p and p.type == "b"]
            if sq_color(bishops[0]) == sq_color(bishops[1]):
                self.status = Status.DRAW
                self.status_desc = "Insufficient material"

        return self.is_ended

    @cache_by_fen
    def get_legal_moves(self) -> list["Move"]:
        # first get opponent attacks and pins
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
            if piece is None or piece.color != self.cur_color:
                continue
            if double_check and not self.allow_king_capture and piece.type != "k":
                continue

            from_col, from_row = from_square

            # check if piece is pinned (ignore if allowed to capture king)
            if self.allow_king_capture:
                is_pinned, pin_axis = False, None
            else:
                is_pinned = pin and pin[0] == from_square
                pin_axis = pin[1] if is_pinned else None

            # adjust vectors based on pin (not relevant for pawn/king)
            vectors = piece.vectors
            if is_pinned and vectors and pin_axis:
                if pin_axis in [(0, 1), (0, -1)]:  # pinned to col
                    vectors = [v for v in vectors if v[1] == 0]
                elif pin_axis in [(1, 0), (-1, 0)]:  # pinned to row
                    vectors = [v for v in vectors if v[0] == 0]
                else:
                    vectors = list(
                        set(vectors) & set([pin_axis, (-pin_axis[0], -pin_axis[1])])
                    )

            if piece.type == "k":
                for col_dir, row_dir in vectors:
                    if (col_dir, row_dir) == check_axis:
                        continue
                    col, row = from_col + col_dir, from_row + row_dir
                    if 0 <= col < 8 and 0 <= row < 8:
                        to_square = (col, row)
                        to_piece = self.board.get(to_square)
                        sq_avail = to_piece is None or to_piece.color != piece.color
                        sq_attacked = to_square in attacked_squares
                        if sq_avail and (self.allow_king_capture or not sq_attacked):
                            moves.append(Move(from_square, (col, row)))

            if piece.type == "p":
                row_dir = 1 if piece.color == WHITE else -1
                row_start = 1 if piece.color == WHITE else 6
                push1 = (from_col, from_row + row_dir)
                push2 = (from_col, from_row + (row_dir * 2))
                cap1 = (from_col - 1, from_row + row_dir)
                cap2 = (from_col + 1, from_row + row_dir)

                def _append_with_promos(to):
                    if to[1] in [0, 7]:
                        for promo in ["q", "r", "b", "n"]:
                            moves.append(Move(from_square, to, promo=promo))
                    else:
                        moves.append(Move(from_square, to))

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
                    and getattr(self.board.get(cap1), 'color', None) != piece.color
                    and not (is_pinned and pin_axis and (pin_axis[0] != 1))
                ):
                    _append_with_promos(cap1)
                if (
                    from_col < 7
                    and self.board.get(cap2)
                    and getattr(self.board.get(cap2), 'color', None) != piece.color
                    and not (is_pinned and pin_axis and (pin_axis[0] != -1))
                ):
                    _append_with_promos(cap2)

                if (
                    self.enpassant
                    and from_row == self.enpassant[1]
                    and abs(from_col - self.enpassant[0]) == 1
                ):
                    enp_to = (self.enpassant[0], from_row + row_dir)
                    moves.append(Move(from_square, enp_to, enpassant=True))

            elif piece.type == "r":
                moves += self.get_moves_by_vectors(from_square, vectors)

            elif piece.type == "q":
                moves += self.get_moves_by_vectors(from_square, vectors)

            elif piece.type == "b":
                moves += self.get_moves_by_vectors(from_square, vectors)

            elif piece.type == "n":
                for col_dir, row_dir in vectors:
                    col, row = from_col + col_dir, from_row + row_dir
                    to_square = (col, row)
                    to_piece = self.board.get(to_square)
                    if 0 <= col < 8 and 0 <= row < 8:
                        if not to_piece or to_piece.color != piece.color:
                            moves.append(Move(from_square, (col, row)))

        # filter out moves that don't resolve check
        if is_check and not self.allow_king_capture:
            intercepts: list[Coords] = []
            if getattr(self.board[check_from], "type", "") in ["r", "q", "b"]:
                sq = check_from
                while True:
                    sq = sq[0] + check_axis[0], sq[1] + check_axis[1]
                    if getattr(self.board[sq], "type", None) == "k":
                        break
                    intercepts.append(sq)

            moves = [
                m
                for m in moves
                if (
                    getattr(self.board[m.from_], "type", None) == "k"
                    or m.to == check_from
                    or m.to in intercepts
                )
            ]

        # add castles unless check (even with king capture)
        if not is_check:
            row = 0 if self.cur_color else 7
            kside: Castle = "K" if self.cur_color else "k"
            if (
                (kside in self.castles)
                and (not self.board.get((5, row)))
                and (not self.board.get((6, row)))
                and (self.allow_king_capture or (not (5, row) in attacked_squares))
                and (self.allow_king_capture or (not (6, row) in attacked_squares))
            ):
                moves.append(Move((4, row), (6, row), castle=kside))

            qside: Castle = "Q" if self.cur_color else "q"
            if (
                (qside in self.castles)
                and (not self.board.get((1, row)))
                and (not self.board.get((2, row)))
                and (not self.board.get((3, row)))
                and (self.allow_king_capture or (not (1, row) in attacked_squares))
                and (self.allow_king_capture or (not (2, row) in attacked_squares))
                and (self.allow_king_capture or (not (3, row) in attacked_squares))
            ):
                moves.append(Move((4, row), (2, row), castle=qside))

        return moves

    @cache_by_fen
    def get_attacks(self) -> tuple[list[Attack], Optional[Pin]]:
        attacks: list[Attack] = []
        pin: Optional[Pin] = None
        color = not self.cur_color

        for from_square, piece in self.board.items():
            if piece is None or (piece.color != color):
                continue
            from_col, from_row = from_square
            vectors = piece.vectors

            if piece.type == "p":
                row_dir = 1 if color == WHITE else -1
                if from_col > 0:
                    cap1 = (from_col - 1, from_row + row_dir)
                    attacks.append((from_square, cap1))
                if from_col < 7:
                    cap2 = (from_col + 1, from_row + row_dir)
                    attacks.append((from_square, cap2))

            elif piece.type == "r":
                r_attacks, r_pin = self.get_attacks_by_vectors(from_square, vectors)
                attacks.extend(r_attacks)
                pin = pin or r_pin

            elif piece.type == "q":
                q_attacks, q_pin = self.get_attacks_by_vectors(from_square, vectors)
                attacks.extend(q_attacks)
                pin = pin or q_pin

            elif piece.type == "b":
                b_attacks, b_pin = self.get_attacks_by_vectors(from_square, vectors)
                attacks.extend(b_attacks)
                pin = pin or b_pin

            elif piece.type == "n":
                for dcol, drow in vectors:
                    col, row = from_col + dcol, from_row + drow
                    if 0 <= col < 8 and 0 <= row < 8:
                        attacks.append((from_square, (col, row)))

            elif piece.type == "k":
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
        piece = self.board.get(from_square)
        if not piece:
            raise ValueError("No piece found on square")
        color = piece.color

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
                if attacked_piece.color == color:
                    # TODO: handle the dreaded enpassant-pin
                    break
                if attacked_piece.type == "k":
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
            if piece is None:
                raise ValueError("No piece found on square")
            col, row = from_square
            while True:
                row += row_dir
                col += col_dir
                to_square = (col, row)
                if row < 0 or row > 7 or col < 0 or col > 7:
                    break
                if not self.board.get(to_square):
                    moves.append(Move(from_square, to_square))
                elif getattr(self.board.get(to_square), "color", None) != piece.color:
                    moves.append(Move(from_square, to_square))
                    break
                else:
                    break

        return moves

    def is_legal_move(self, val: Union["Move", str]) -> bool:
        move: Optional[Move] = None
        if isinstance(val, str):
            move = Move.parse(val, self)
        else:
            move = val
        if not move:
            raise ValueError("Invalid move")
        if move.action:
            return True
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
                and attacked_piece.type == "k"
                and attacked_piece.color == self.cur_color
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
            self.turn: int = 0 if turn == "w" else 1
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
            f"{board_str} {'w' if self.cur_color == WHITE else 'b'}"
            f" {''.join(self.castles) if self.castles else '-'}"
            f" {c2sq(self.enpassant) if self.enpassant else '-'}"
            f" {self.draw_counter} {(len(self.history) // 2) or 1}"
        )
        return fen_str

    def render_board(self, color: Optional[bool] = None, clear: bool = False) -> None:
        color = color if color is not None else WHITE
        if clear:
            os.system("clear")
        range_dir = range(7, -1, -1) if color == WHITE else range(8)
        col_dir = range(7, -1, -1) if color == BLACK else range(8)
        print()
        for row in range_dir:
            for col in col_dir:
                piece = self.board.get((col, row)) or "-"
                print(piece, end=" ")
            print()

    @cache_by_fen
    def is_legal_state(self) -> bool:
        # 2 kings
        if len([p for p in self.board.values() if p and p.type == "k"]) != 2:
            return False
        # no pawns on first/last row
        for col in range(8):
            if (
                getattr(self.board.get((col, 0)), "type", None) == "p"
                or getattr(self.board.get((col, 7)), "type", None) == "p"
            ):
                return False
        # player can't be in check if not their turn
        if not self.allow_king_capture:
            g = Game(fen=self.get_fen())
            g.turn = 1 - self.turn
            if g.is_check():
                return False
        return True

    def randomize_board(self, max_pieces: Optional[int]=None) -> None:
        while True:
            pieces: list[PieceType] = ["p", "n", "b", "r", "q", "P", "N", "B", "R", "Q"]
            board1d: list[Optional[Piece]] = [None] * 64
            king_positions = random.sample(range(64), 2)
            board1d[king_positions[0]] = Piece("K")
            board1d[king_positions[1]] = Piece("k")
            max_ct = max_pieces or 64
            for i in range(64):
                if max_ct > 0 and board1d[i] is None and random.random() < 0.5:
                    board1d[i] = Piece(random.choice(pieces))
                    max_ct -= 1

            board: Board = Board({(i % 8, 7 - i // 8): board1d[i] for i in range(64)})
            fen = board.to_fen()
            g = Game()
            g.set_board(fen)
            if g.is_legal_state() and not g.check_game_over():
                self.set_board(fen)
                break

    @cache_by_fen
    def get_position_score(self) -> float:
        if self.status == Status.WWINS:
            return float("inf")
        if self.status == Status.BWINS:
            return float("-inf")
        if self.status == Status.DRAW:
            return 0

        wscore, bscore = 0.0, 0.0
        for piece in self.board.values():
            if not piece:
                continue
            if piece.color == WHITE:
                wscore += piece.value
            else:
                bscore += piece.value
        return wscore - bscore

    def lookahead(self, move):
        g = Game(fen=self.get_fen(), id="lookahead")
        g.make_move(move)
        return g

    def to_dict(self):
        return {
            "id": self.id,
            "fen": self.fen,
            "status": self.status.value if self.status else None,
            "status_desc": self.status_desc,
            "players": [p.__class__.__qualname__ for p in self.players],
            "history": self.history,
            "legal_moves": [m.to_notation(self) for m in self.get_legal_moves()],
            "last_move": self.last_move if self.last_move else None
        }

    @property
    def cur_color(self) -> bool:
        return WHITE if self.turn == 0 else BLACK

    @property
    def cur_player(self) -> "Player":
        return self.players[self.turn]

    @property
    def last_move(self) -> Optional[str]:
        return self.history[-1] if len(self.history) else None

    @property
    def is_ended(self) -> bool:
        return self.status not in (None, Status.PLAYING)

    @property
    def status_str(self):
        return f"{self.status.value if self.status else '-'}: {self.status_desc}!"


class Player(abc.ABC):
    @abc.abstractmethod
    def get_move(self, game: Game) -> "Move":
        pass


class Computer(Player):
    def get_move(self, game: Game) -> "Move":
        moves = game.get_legal_moves()
        return random.choice(moves)


class Human(Player):
    def get_move(self, game: Game) -> "Move":
        game.render_board(clear=True)

        # print last move if present
        if game.last_move:
            print(f"\n[{'B' if game.cur_color else 'W'}]", game.last_move)

        move = None
        while not move:
            prompt = f"\n[{'W' if game.cur_color else 'B'}] => "
            val = input(prompt).strip()
            if val in ["quit", "exit"]:
                sys.exit(0)
            elif val in ["legal", "l"]:
                legal_moves = game.get_legal_moves()
                legal_moves = [m.to_notation(game) for m in legal_moves]
                # consolidate moves with pawn promotions
                legal_moves = list(
                    set([re.sub(r"[qrbn]$", "", l) for l in legal_moves])
                )
                print("Legal moves:", "|".join(legal_moves))
                continue

            move = Move.parse(val, game)
            if not move:
                print("Please enter a move, e.g. e2e4, or 'quit' to exit.")
            elif not game.is_legal_move(move):
                print("Illegal move!")
                move = None
        return move


class Board:
    def __init__(self, board: Optional[BoardType] = None):
        self.board: BoardType = board or collections.defaultdict(lambda: None)

    def __getitem__(self, coords: Optional[Coords]) -> Optional["Piece"]:
        if not coords:
            return None
        return self.board.get(coords)

    def __setitem__(
        self, coords: Coords, val: Optional[Union["Piece", PieceType]]
    ) -> None:
        piece: Optional[Piece] = None
        if isinstance(val, str):
            piece = Piece(val)
        else:
            piece = val
        self.board[coords] = piece

    def __contains__(self, coords: Coords) -> bool:
        return coords in self.board

    def items(self) -> list[tuple[Coords, Optional["Piece"]]]:
        return list(self.board.items())

    def values(self) -> list[Optional["Piece"]]:
        return list(self.board.values())

    def get(
        self, coords: Coords, default: Optional["Piece"] = None
    ) -> Optional["Piece"]:
        return self.board.get(coords, default)

    def to_fen(self) -> str:
        board_str = ""
        for row_index in range(7, -1, -1):
            empty_count = 0
            for col_index in range(8):
                piece = self.board.get((col_index, row_index))
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
        board: dict[Coords, Optional[Piece]] = collections.defaultdict(lambda: None)
        rows = fen.split("/")
        for row_index, row in enumerate(rows):
            col_index = 0
            for char in row:
                if char.isdigit():
                    col_index += int(char)
                else:
                    if char not in "pnbrqkPNBRQK":
                        raise ValueError("Invalid FEN piece")
                    board[(col_index, 7 - row_index)] = Piece(char)  # type: ignore
                    col_index += 1
        return cls(board)


class Move:
    NOTATION_RE = re.compile(
        r"^([NBRQKP])?([a-h])?([1-8])?([x-])?([a-h])([1-8])(=?[nrbqNRBQ])?"
        r"(\+|#|\?|\!|(\s*[eE]\.?[pP]\.?))*$"
    )

    def __init__(
        self,
        from_: Optional[Coords] = None,
        to: Optional[Coords] = None,
        action: Optional[Action] = None,
        promo: Optional[Promo] = None,
        enpassant: Optional[bool] = False,
        castle: Optional[Castle] = None,
    ):
        self.from_ = from_
        self.to = to
        self.action = action
        self.promo = promo
        self.enpassant = enpassant
        self.castle = castle

    def __eq__(self, obj: object) -> bool:
        if not obj or not type(obj) == Move:
            return False
        if obj.action:
            return obj.action == self.action
        return obj.from_ == self.from_ and obj.to == self.to and obj.promo == self.promo

    @classmethod
    def parse(cls, val: str, game: Game) -> Optional["Move"]:
        val = val.strip()

        if not val:
            return None
        elif "resign" in val.lower():
            return Move(action=Action.RESIGN)
        elif val.lower().startswith("o-o-o") or val.lower().startswith("0-0-0"):
            row = 0 if game.cur_color else 7
            return Move((4, row), (2, row), castle="Q" if game.cur_color else "q")
        elif val.lower().startswith("o-o") or val.lower().startswith("0-0"):
            row = 0 if game.cur_color else 7
            return Move((4, row), (6, row), castle="K" if game.cur_color else "k")

        match = cls.NOTATION_RE.match(val)

        if not match:
            return None

        piece_from_str = (match.group(1) or "p").lower()
        piece_from = Piece(piece_from_str, game.cur_color)
        col_from: Optional[int] = (
            ord(match.group(2)) - ord("a") if match.group(2) else None
        )
        row_from: Optional[int] = int(match.group(3)) - 1 if match.group(3) else None
        col_to: int = ord(match.group(5)) - ord("a")
        row_to: int = int(match.group(6)) - 1

        if not col_from or not row_from:
            for move in game.get_legal_moves():
                if (
                    (move.to == (col_to, row_to))
                    and (game.board[move.from_] == piece_from)
                    and (col_from is None or move.from_[0] == col_from)
                    and (row_from is None or move.from_[1] == row_from)
                ):
                    col_from, row_from = move.from_
                    break
            if col_from is None or row_from is None:
                return None

        promo = match.group(7) or None
        if promo:
            promo = promo.lstrip("=").lower()

        castle = None
        piece = game.board.get((col_from, row_from))
        if col_from == 4 and piece and piece.type == "k":
            if col_to == 2:
                castle = "Q" if game.cur_color else "q"
            elif col_to == 6:
                castle = "K" if game.cur_color else "k"
        return Move((col_from, row_from), (col_to, row_to), promo=promo, castle=castle)

    def to_notation(self, game: Optional[Game] = None) -> str:
        if self.action:
            return self.action.value
        if not self.from_ or not self.to:
            raise ValueError("Invalid move")
        col_from, row_from = self.from_
        col_to, row_to = self.to
        captured = game and game.board.get(self.to, None)
        sep = "" if not captured and not self.enpassant else "x"

        return (
            f"{c2sq((col_from, row_from))}{sep}"
            f"{c2sq((col_to, row_to))}"
            f"{'=' + self.promo.lower() if self.promo else ''}"
        )

    def __repr__(self) -> str:
        return self.to_notation(None)


class Piece:
    SCORES = dict(p=10, n=30, b=35, r=50, q=90, k=0)
    VECTORS: dict[PieceType, PieceVectors] = dict(
        n=[(1, 2), (2, 1), (-1, 2), (-2, 1), (1, -2), (2, -1), (-1, -2), (-2, -1)],
        q=[(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, 1), (1, -1), (-1, -1)],
        k=[(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)],
        r=[(0, 1), (0, -1), (1, 0), (-1, 0)],
        b=[(1, 1), (-1, 1), (1, -1), (-1, -1)],
    )

    def __init__(self, char: PieceType, color: Optional[bool] = None) -> None:
        self.color = color or char.isupper()
        self.type = char.lower()

    def __eq__(self, obj: object) -> bool:
        if not obj or not type(obj) == Piece:
            return False
        return obj.color == self.color and obj.type == self.type

    def __str__(self) -> str:
        return self.type.upper() if self.color == WHITE else self.type

    def __repr__(self) -> str:
        return f"Piece('{self.type}', '{self.color}')"

    def is_white(self) -> bool:
        return self.color == WHITE

    def is_black(self) -> bool:
        return self.color == BLACK

    @property
    def vectors(self) -> PieceVectors:
        return self.VECTORS.get(self.type, [])

    @property
    def value(self) -> int:
        return self.SCORES.get(self.type, 0)


def sq2c(sq: Square) -> Coords:
    """a1 -> (0, 0)"""
    return ord(sq[0]) - ord("a"), int(sq[1]) - 1


def c2sq(coords: Coords) -> Square:
    """(0, 0) -> a1"""
    return chr(ord("a") + coords[0]) + str(coords[1] + 1)


def sq_color(c: Coords):
    return WHITE if (c[0] + c[1]) % 2 else BLACK


if __name__ == "__main__":
    argp = argparse.ArgumentParser()
    argp.add_argument("--play", "-p", type=int, default=0)
    argp.add_argument("--allow-king-capture", "-k", action="store_true")
    argp.add_argument("--black", "-b", action="store_true")
    options = argp.parse_args()

    # self-play
    if options.play > 0:
        from tqdm import tqdm  # type: ignore

        outcomes: dict[str, int] = dict()

        for _ in tqdm(range(options.play)):
            game = Game(players=(Computer(), Computer()))
            game.allow_king_capture = options.allow_king_capture
            game.play()
            outcomes[game.status_str] = outcomes.get(game.status_str, 0) + 1
            if game.is_stalemate():
                game.render_board()
        for o, ct in outcomes.items():
            print(f"{o}: {ct}")
        sys.exit(0)

    # cli human vs computer
    players: Players = (
        (Human(), Computer()) if not options.black else (Computer(), Human())
    )
    game = Game(players=players)
    game.allow_king_capture = options.allow_king_capture
    game.play()

    # finished:
    game.render_board(clear=True)
    print("\n", game.status_str)
