#!/usr/bin/env python3.11

import unittest, sys, io
from chess import *


class TestChess(unittest.TestCase):
    def test_fen(self):
        st1 = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        g = Game(st1)
        st2 = g.fen
        self.assertEqual(
            st2, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        )
        self.assertEqual(g.turn, WHITE)
        self.assertEqual(g.draw_counter, 0)
        self.assertEqual(g.enpassant, None)
        self.assertEqual(g.castles, ["K", "Q", "k", "q"])

    def test_get_attacks(self):
        g = Game()
        g.turn = BLACK
        attacks, pin = g.get_attacks()
        attacked_squares = list(set([c2sq(a[1]) for a in attacks]))
        self.assertEqual(
            " ".join(sorted(attacked_squares)),
            " ".join(
                sorted(
                    "b1 c1 d1 e1 f1 g1 a2 b2 c2 d2 e2 f2 g2 h2 a3 "
                    "b3 c3 d3 e3 f3 g3 h3".split()
                )
            ),
        )
        self.assertIsNone(pin)

        g = Game("rnbqkbnr/8/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        attacks, pin = g.get_attacks()
        attacked_squares = list(set([c2sq(a[1]) for a in attacks]))
        self.assertEqual(
            " ".join(sorted(attacked_squares)),
            " ".join(
                sorted(
                    set(
                        "a7 a6 a5 a4 a3 a2 b8 "  # Ra8
                        "a6 c6 d7 "  # Nb8
                        "b7 a6 d7 e6 f5 g4 h3 "  # Bc8
                        "c7 b6 a5 e7 f6 g5 h4 d7 d6 d5 d4 d3 d2 e8 c8 "  # Qd8
                        "d8 f8 d7 e7 f7 "  # Ke8
                        "e7 d6 c5 b4 a3 g7 h6 "  # Bf8
                        "h6 f6 d7 "  # Ng8
                        "h7 h6 h5 h4 h3 h2 g8".split()  # Rh8
                    )
                )
            ),
        )
        self.assertIsNone(pin)

        g = Game("rnb1kbnr/4q3/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        attacks, pin = g.get_attacks()
        self.assertEqual(pin, (sq2c("e2"), (0, -1)))

        g = Game("8/8/8/8/6b1/5B2/8/3K4 w KQkq - 0 1")
        attacks, pin = g.get_attacks()
        self.assertEqual(pin, (sq2c("f3"), (-1, -1)))

    def test_make_move(self):
        g = Game()
        g.make_move("e2e4")
        self.assertEqual(
            g.fen,
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e4 0 1",
        )
        self.assertEqual(g.last_move, "e2e4")
        self.assertEqual(g.turn, BLACK)
        self.assertEqual(g.status[0], "playing")
        self.assertEqual(g.status[1], "")
        g.make_move("e2e4")
        self.assertEqual(g.turn, WHITE)
        self.assertEqual(len(g.history), 2)

    def test_pawn_moves(self):
        # basic moves
        g = Game("8/8/8/8/8/8/PPPPPPPP/8 w - - 0 1")
        self.assertLegalMoves(
            g,
            "a2a3|a2a4|b2b3|b2b4|c2c3|c2c4|d2d3|d2d4|e2e3|e2e4|f2f3|f2f4|"
            "g2g3|g2g4|h2h3|h2h4",
        )
        # 1-push
        g = Game("8/8/P7/8/8/8/8/8 w - - 0 1")
        self.assertLegalMoves(g, "a6a7")
        # promos
        g = Game("1q6/P7/8/8/8/8/8/8 w - - 0 1")
        self.assertLegalMoves(
            g, "a7a8=q|a7a8=r|a7a8=b|a7a8=n|a7xb8=q|a7xb8=r|a7xb8=b|a7xb8=n"
        )
        # pin diagonal
        g = Game("4k3/5p2/4N1B1/8/8/8/8/8 b - - 0 1")
        self.assertLegalMoves(g, "f7xg6|e8e7|e8d7")
        # pin on col
        g = Game("4k3/4p3/3Q1B2/8/8/8/8/4R3 b - - 0 1")
        self.assertLegalMoves(g, "e7e6|e7e5|e8f8|e8f7")
        # pin on row
        g = Game("8/k3p2R/8/8/8/8/8/8 b - - 0 1")
        self.assertLegalMoves(g, "a7a8|a7a6|a7b8|a7b7|a7b6")
        # enpassant
        g = Game("8/8/8/2Pp4/8/8/8/8 w - d5 0 1")
        self.assertLegalMoves(g, "c5c6|c5xd6")

        # TODO: enpassant w pin!
        # g = Game("8/8/8/K1Pp3r/8/8/8/8 w - d5 0 1")
        # self.assertLegalMoves(moves, "c5c6")

    def test_knight_moves(self):
        g = Game("8/2B1b3/8/3N4/8/8/8/N7 w - - 0 1")
        self.assertLegalMoves(
            g,
            "d5b6|d5b4|d5c3|d5e3|d5f4|d5f6|d5xe7|a1b3|a1c2|"
            "c7b8|c7d6|c7e5|c7f4|c7g3|c7h2|c7d8|c7b6|c7a5",
        )

    def test_king_moves(self):
        # basic
        g = Game("8/8/8/8/8/8/8/4K3 w - - 0 1")
        self.assertLegalMoves(g, "e1d1|e1d2|e1e2|e1f2|e1f1")

        # capture
        g = Game("8/8/8/8/8/8/4pP2/4K3 w - - 0 1")
        self.assertLegalMoves(g, "e1xe2|e1d2|f2f3|f2f4")

        # can't move into check
        g = Game("3r4/8/8/8/8/8/r7/4K3 w - - 0 1")
        self.assertLegalMoves(g, "e1f1")

        # must move out of check
        g = Game("8/4r3/8/8/8/8/P7/4K3 w - - 0 1")
        self.assertLegalMoves(g, "e1d1|e1d2|e1f1|e1f2")

        # castles: basic
        g = Game("r3kbnr/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
        moves = sorted(g.get_legal_moves())
        self.assertTrue("e1g1" in moves)
        self.assertTrue("e1c1" in moves)
        g = Game("r3kbnr/8/8/8/8/8/8/R3K2R b KQkq - 0 1")
        moves = sorted(g.get_legal_moves())
        self.assertTrue("e8g8" not in moves)
        self.assertTrue("e8c8" in moves)

        # castles: r/k has moved or otherwise unavailable
        g = Game("r3kbnr/8/8/8/8/8/8/R3K2R w - - 0 1")
        moves = sorted(g.get_legal_moves())
        self.assertTrue("e1g1" not in moves)
        self.assertTrue("e1c1" not in moves)
        g = Game("r3kbnr/8/8/8/8/8/8/R3K2R b - - 0 1")
        moves = sorted(g.get_legal_moves())
        self.assertTrue("e8g8" not in moves)
        self.assertTrue("e8c8" not in moves)

        # castles: squares attacked
        g = Game("r3kbrn/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
        moves = sorted(g.get_legal_moves())
        self.assertTrue("e1g1" not in moves)
        self.assertTrue("e1c1" in moves)

        # test castling with sequential king/rook moves
        g = Game("r3kbnr/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
        self.assertTrue(g.is_legal_move("e1g1"))
        self.assertTrue(g.is_legal_move("e1c1"))
        g.make_move("a1a2")  # move A rook
        self.assertTrue(g.is_legal_move("e8c8"))
        self.assertFalse(g.is_legal_move("e8g8"))
        g.make_move("e8c8")
        self.assertTrue(g.is_legal_move("e1g1"))
        self.assertFalse(g.is_legal_move("e1c1"))
        make_moves(g, ["a8a7", "h1h2"])  # move H rook
        self.assertFalse(g.is_legal_move("e1g1"))
        self.assertFalse(g.is_legal_move("e1c1"))
        g = Game("r3kbnr/8/8/8/8/8/8/R2K3R w KQkq - 0 1")
        g.make_move("e1e2")  # move king
        self.assertFalse(g.is_legal_move("e1g1"))
        self.assertFalse(g.is_legal_move("e1c1"))

    def test_rook_moves(self):
        # basic
        g = Game("8/4p3/8/4R3/8/4P3/8/8 w - - 0 1")
        self.assertLegalMoves(
            g, "e5d5|e5c5|e5b5|e5a5|e5f5|e5g5|e5h5|e5e4|e5e6|e5xe7|e3e4"
        )
        # pinned diagonal
        g = Game("K7/1R6/2b5/8/8/8/8/8 w - - 0 1")
        self.assertLegalMoves(g, "a8a7|a8b8")
        # pinned to row
        g = Game("KR1r4/8/8/8/8/8/8/8 w - - 0 1")
        self.assertLegalMoves(g, "a8a7|a8b7|b8c8|b8xd8")
        # pinned to col
        g = Game("K8/R7/8/8/8/8/r7/8 w - - 0 1")
        self.assertLegalMoves(g, "a8b8|a8b7|a7a6|a7a5|a7a4|a7a3|a7xa2")

    def test_bishop_moves(self):
        # basic
        g = Game("8/2p5/8/4B3/8/6P1/8/8 w - - 0 1")
        moves = sorted(g.get_legal_moves())
        self.assertLegalMoves(
            g, "e5d6|e5xc7|e5f4|e5h8|e5g7|e5f6|e5d4|e5c3|e5b2|e5a1|g3g4"
        )
        # pinned to row/col
        g = Game("8/4K2/8/4B3/8/4r2/8/8/8 w - - 0 1")
        moves = sorted(g.get_legal_moves())
        self.assertTrue(len([m for m in moves if m[0:1] == "e5"]) == 0)
        g = Game("8/8/8/r3B2k/8/8/8/8/8 w - - 0 1")
        moves = sorted(g.get_legal_moves())
        self.assertTrue(len([m for m in moves if m[0:1] == "e5"]) == 0)
        # pinned to diags
        g = Game("8/6K1/8/4B3/3b4/8/8/8 w - - 0 1")
        moves = sorted(g.get_legal_moves())
        self.assertTrue("e5d4" in moves)
        self.assertTrue("e5f6" in moves)
        self.assertTrue("e5d6" not in moves)
        self.assertTrue("e5f4" not in moves)
        g = Game("8/6b1/8/4B3/3K4/8/8/8 w - - 0 1")
        moves = sorted(g.get_legal_moves())
        self.assertTrue("e5g7" in moves)
        self.assertTrue("e5f6" in moves)
        self.assertTrue("e5d6" not in moves)
        self.assertTrue("e5f4" not in moves)

    def test_queen_moves(self):
        # basic
        g = Game("8/4p3/8/4Q3/8/4P3/8/8 w - - 0 1")
        self.assertLegalMoves(
            g,
            "e5e6|e5xe7|e5e4|e5d5|e5c5|e5b5|e5a5|e5f5|e5g5|e5h5|"
            "e5a1|e5b2|e5c3|e5d4|e5f6|e5g7|e5h8|"
            "e5d6|e5c7|e5b8|e5f4|e5g3|e5h2|e3e4",
        )
        # pinned diagonal
        g = Game("K7/1Q6/2b5/8/8/8/8/8 w - - 0 1")
        self.assertLegalMoves(g, "a8a7|a8b8|b7xc6")
        # pinned to row
        g = Game("KQ1r4/8/8/8/8/8/8/8 w - - 0 1")
        self.assertLegalMoves(g, "a8a7|a8b7|b8c8|b8xd8")
        # pinned to col
        g = Game("K8/Q7/8/8/8/8/r7/8 w - - 0 1")
        self.assertLegalMoves(g, "a8b8|a8b7|a7a6|a7a5|a7a4|a7a3|a7xa2")

    def test_in_check_legal_moves(self):
        # double check must move king out of check
        g = Game("N7/8/8/8/8/2nr4/7P/QRNK2R1 w - - 0 1")
        self.assertLegalMoves(g, "d1c2|d1e1")
        # capture attacking piece
        g = Game("N7/8/8/8/8/3r4/7P/QRNK2R1 w - - 0 1")
        self.assertLegalMoves(g, "d1c2|d1e1|d1e2|c1xd3")
        # can't castle
        g = Game("4r3/8/8/8/8/8/8/R3K2R w - - 0 1")
        self.assertLegalMoves(g, "e1d1|e1f1|e1d2|e1f2")
        # intercept check
        g = Game("7R/8/4r3/8/8/8/Q6R/R2BK1NR w - - 0 1")
        self.assertLegalMoves(g, "e1f1|e1d2|e1f2|g1e2|a2e2|h2e2|d1e2|a2xe6")
        # intercept check diagonally
        g = Game("q5B1/4N3/8/7Q/3PK3/8/8/3R1R1Q w - - 0 1")
        self.assertLegalMoves(g, "h5d5|e7d5|e7c6|d4d5|g8d5|e4e3|e4e5|e4f5|e4f4|e4d3")
        # king can't move along axis of check (diagonal)
        g = Game("8/8/5k2/8/3Q4/8/8/8 b - - 0 1")
        self.assertLegalMoves(g, "f6e6|f6e7|f6f5|f6f7|f6g6|f6g5")
        # king can't move along axis of check (flank)
        g = Game("8/8/5k2/8/5R2/8/8/8 b - - 0 1")
        self.assertLegalMoves(g, "f6e6|f6e7|f6e5|f6g7|f6g6|f6g5")

    def test_discovered_check(self):
        g = Game("8/p7/5k2/8/8/8/5N2/5R2 b - - 0 1")
        self.assertLegalMoves(g, "f6e6|f6e7|f6e5|f6g7|f6g6|f6g5|f6f5|f6f7|a7a6|a7a5")
        make_moves(g, ["f6f5", "f2d1"])
        g.print_board()
        self.assertLegalMoves(g, "f5e4|f5e5|f5e6|f5g4|f5g5|f5g6")

    def test_is_check(self):
        g = Game()
        self.assertFalse(g.is_check())
        g.turn = BLACK
        self.assertFalse(g.is_check())
        g.board[(4, 6)] = "Q"
        g.board[(4, 1)] = None
        self.assertTrue(g.is_check())
        g.turn = WHITE
        self.assertFalse(g.is_check())

    def test_enpassant(self):
        g = Game()
        self.assertEqual(g.enpassant, None)
        g.make_move("a2a4")
        self.assertEqual(g.enpassant, (0, 3))
        g.make_move("b7b6")
        self.assertEqual(g.enpassant, None)
        g.make_move("e2e4")
        self.assertEqual(g.enpassant, (4, 3))
        g.make_move("a7a5")
        self.assertEqual(g.enpassant, (0, 4))
        g.make_move("e4e5")
        g.make_move("d7d5")
        self.assertEqual(g.enpassant, (3, 4))
        self.assertTrue(g.is_legal_move("e5xd6"))
        self.assertTrue(g.is_legal_move("e5d6"))
        g.make_move("e5xd6")
        self.assertEqual(g.enpassant, None)
        self.assertEqual("", g.board[sq2c("d5")])

    def test_checkmate(self):
        g = Game("7k/RR6/8/8/8/8/8/K7 w - - 0 1")
        self.assertFalse(g.is_checkmate())
        g.turn = BLACK
        self.assertFalse(g.is_checkmate())
        g.turn = WHITE
        self.assertEqual(g.status[0], "playing")
        g.make_move("a7a8")
        self.assertTrue(g.is_checkmate())
        self.assertEqual(g.status[0], "wwins")

    def test_stalemate(self):
        g = Game("7k/R7/8/8/8/8/8/K4R2 w - - 0 1")
        self.assertFalse(g.is_stalemate())
        g.make_move("f1g1")
        g.turn = WHITE
        self.assertFalse(g.is_stalemate())  # not black's turn yet
        g.turn = BLACK
        self.assertTrue(g.is_stalemate())

    def test_insufficient_material(self):
        # k v k
        g = Game("7k/8/8/8/8/8/8/K7 w - - 0 1")
        self.assertTrue(g.is_ended)
        self.assertEqual(g.status[0], "draw")
        self.assertEqual(g.status[1], "insuff_material")
        # k/b v k
        g = Game("7k/8/8/8/8/8/8/KB6 w - - 0 1")
        self.assertEqual(g.status[1], "insuff_material")
        g = Game("6bk/8/8/8/8/8/8/K7 w - - 0 1")
        self.assertEqual(g.status[1], "insuff_material")
        # k/n v k
        g = Game("7k/8/8/8/8/8/8/KN6 w - - 0 1")
        self.assertEqual(g.status[1], "insuff_material")
        g = Game("6nk/8/8/8/8/8/8/K7 w - - 0 1")
        self.assertEqual(g.status[1], "insuff_material")
        # k/b v k/b - same color
        g = Game("6kb/8/8/8/8/8/8/BK6 w - - 0 1")
        self.assertEqual(g.status[1], "insuff_material")
        # k/b v k/b - opp color
        g = Game("6k1/7b/8/8/8/8/8/BK6 w - - 0 1")
        self.assertNotEqual(g.status[1], "insuff_material")

    def test_50_move_rule(self):
        g = Game()
        self.assertEqual(g.draw_counter, 0)
        for _ in range(12):
            g.make_move("g1f3")
            g.make_move("g8f6")
            g.make_move("f3g1")
            g.make_move("f6g8")
            g.repititions = dict()
        self.assertEqual(g.status[0], "playing")
        self.assertEqual(g.draw_counter, 48)

        g.make_move("g1f3")
        g.make_move("g8f6")
        self.assertEqual(g.status[0], "draw")
        self.assertEqual(g.status[1], "fifty_move")
        self.assertEqual(g.draw_counter, 50)

    def test_print_board(self):
        g = Game()
        val = _capture_stdout(g.print_board)
        self.assertTrue(val.startswith("\nr n b q k b n r \n"))
        val = _capture_stdout(g.print_board, color=BLACK)
        self.assertTrue(val.startswith("\nR N B K Q B N R \n"))

    def test_allow_king_capture(self):
        g = Game("rnbqkbnr/8/8/8/8/8/8/RNBQKBNR w KQkq - 0 1")
        g.allow_king_capture = True
        self.assertTrue(g.is_legal_move("e1d2"))
        g.make_move("e1d2")
        self.assertTrue(g.is_legal_move("d8xd2"))
        g.make_move("d8xd2")
        self.assertEqual(g.status[0], "bwins")
        self.assertEqual(g.status[1], "king_captured")

    def test_threefold_rep(self):
        g = Game()
        make_moves(g, ["g1f3", "g8f6", "f3g1", "f6g8"])
        self.assertEqual(g.status[0], "playing")
        make_moves(g, ["g1f3", "g8f6", "f3g1", "f6g8"])
        self.assertEqual(g.status[0], "draw")
        self.assertEqual(g.status[1], "threefold_rep")

    def test_position_score(self):
        g = Game()
        self.assertEqual(g.get_position_score(), 0.0)
        make_moves(g, ["e2e4", "d7d5", "e4xd5"])
        self.assertEqual(g.get_position_score(), 10.0)
        g.board[(3, 0)] = None
        self.assertEqual(g.get_position_score(), -80.0)
        g = Game("7k/RR6/8/8/8/8/8/K7 w - - 0 1")
        g.make_move("a7a8#")
        self.assertEqual(g.get_position_score(), float("inf"))

    def test_minimax(self):
        g = Game(fen="8/p7/8/3qk1n1/8/8/P2Q4/2K4N w - - 0 1")
        p = Computer()
        p.lookahead_moves = 0
        move = p.get_move(g)
        self.assertEqual(move, "d2d5")
        p.lookahead_moves = 1
        move = p.get_move(g)
        self.assertEqual(move, "d2g5")

        # black moves
        g.make_move("c1d1")
        p.lookahead_moves = 0
        move = p.get_move(g)
        self.assertEqual(move, "d5d2")
        p.lookahead_moves = 1
        move = p.get_move(g)
        self.assertEqual(move, "d5h1")

    def assertLegalMoves(self, g: Game, moves_str: str):
        moves = sorted(g.get_legal_moves())
        self.assertEqual(
            moves, sorted([m.replace("x", "") for m in moves_str.split("|")])
        )


def _capture_stdout(func, *args, **kwargs):
    out = io.StringIO()
    sys.stdout = out
    func(*args, **kwargs)
    sys.stdout = sys.__stdout__
    return out.getvalue()


def make_moves(game, moves):
    for move in moves:
        game.make_move(move)


if __name__ == "__main__":
    unittest.main()
