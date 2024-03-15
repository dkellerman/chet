#!/usr/bin/env python3

import unittest
import chess as C


class TestChess(unittest.TestCase):
    def test_board_state(self):
        st1 = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
        g = C.Game(board=st1)
        st2 = g.get_board_state()
        self.assertEqual(st1, st2)

    def test_notation(self):
        g = C.Game()
        self.assertEqual(g.to_notation((0, 1), (0, 3), {}), "a2a4")
        self.assertEqual(g.parse_notation("a2a4"), ((0, 1), (0, 3), {}))
        self.assertEqual(g.parse_notation("e2e4"), ((4, 1), (4, 3), {}))
        self.assertEqual(g.to_notation((0, 6), (0, 7), dict(promo="q")), "a7xa8q")
        self.assertEqual(g.parse_notation("a2a4q"), ((0, 1), (0, 3), dict(promo="q")))

    def test_make_move(self):
        g = C.Game()
        g.make_move("e2e4")
        self.assertEqual(
            g.get_board_state(), "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR"
        )
        self.assertEqual(g.color, C.BLACK)
        self.assertEqual(g.last_move, "e2e4")
        self.assertEqual(g.turn, 1)
        self.assertEqual(g.status, None)
        self.assertEqual(g.ending, None)
        g.make_move("e2e4")
        self.assertEqual(g.color, C.WHITE)
        self.assertEqual(g.turn, 0)
        self.assertEqual(len(g.history), 2)

    def test_get_legal_moves(self):
        g = C.Game()
        moves = sorted([g.to_notation(*m) for m in g.get_legal_moves()])
        self.assertEqual(
            moves,
            "a2a3|a2a4|a7a5|a7a6|b1a3|b1c3|b2b3|b2b4|b7b5|b7b6|"
            "b8a6|b8c6|c2c3|c2c4|c7c5|c7c6|d2d3|d2d4|d7d5|d7d6|"
            "e2e3|e2e4|e7e5|e7e6|f2f3|f2f4|f7f5|f7f6|g1f3|g1h3|"
            "g2g3|g2g4|g7g5|g7g6|g8f6|g8h6|h2h3|h2h4|h7h5|h7h6".split("|"),
        )
        moves = sorted([g.to_notation(*m) for m in g.get_legal_moves(color=C.BLACK)])
        self.assertEqual(
            moves,
            "a7a5|a7a6|b7b5|b7b6|b8a6|b8c6|c7c5|c7c6|d7d5|d7d6|"
            "e7e5|e7e6|f7f5|f7f6|g7g5|g7g6|g8f6|g8h6|h7h5|h7h6".split("|"),
        )

        g = C.Game(board="rnbqkbnr/8/8/8/8/8/8/RNBQKBNR")
        moves = sorted([g.to_notation(*m) for m in g.get_legal_moves(color=C.WHITE)])
        self.assertEqual(
            moves,
            "a1a2|a1a3|a1a4|a1a5|a1a6|a1a7|a1xa8|b1a3|b1c3|b1d2|"
            "c1a3|c1b2|c1d2|c1e3|c1f4|c1g5|c1h6|d1a4|d1b3|d1c2|d1d2|"
            "d1d3|d1d4|d1d5|d1d6|d1d7|d1e2|d1f3|d1g4|d1h5|d1xd8|"
            "e1e2|e1f2|f1a6|f1b5|f1c4|f1d3|f1e2|f1g2|f1h3|g1e2|g1f3|g1h3|"
            "h1h2|h1h3|h1h4|h1h5|h1h6|h1h7|h1xh8".split("|"),
        )

    def test_can_castle(self):
        g = C.Game(board="r3kbnr/8/8/8/8/8/8/R3K2R")
        moves = sorted([g.to_notation(*m) for m in g.get_legal_moves()])
        self.assertTrue("e1g1" in moves)
        self.assertTrue(g.can_castle(C.WHITE, True))
        self.assertTrue("e1c1" in moves)
        self.assertTrue(g.can_castle(C.WHITE, False))
        self.assertTrue("e8g8" not in moves)
        self.assertFalse(g.can_castle(C.BLACK, True))
        self.assertTrue("e8c8" in moves)
        self.assertTrue(g.can_castle(C.BLACK, False))

        g.castle = []
        moves = sorted([g.to_notation(*m) for m in g.get_legal_moves()])
        self.assertTrue("e1g1" not in moves)
        self.assertTrue("e1c1" not in moves)
        self.assertTrue("e8g8" not in moves)
        self.assertTrue("e8c8" not in moves)

        g = C.Game(board="r3kbrn/8/8/8/8/8/8/R3K2R") # attacked
        self.assertFalse(g.can_castle(C.WHITE, True))
        self.assertTrue(g.can_castle(C.WHITE, False))
        g.make_move("e1d1") # move king
        self.assertFalse(g.can_castle(C.WHITE, False))

        g = C.Game(board="r3kbnr/8/8/8/8/8/8/R3K2R")
        self.assertTrue(g.can_castle(C.WHITE, True))
        self.assertTrue(g.can_castle(C.WHITE, False))
        g.make_move("a1a2") # move A rook
        self.assertTrue(g.can_castle(C.WHITE, True))
        self.assertFalse(g.can_castle(C.WHITE, False))
        g.make_move("h1h2") # move H rook
        self.assertFalse(g.can_castle(C.WHITE, True))

        g = C.Game(board="r3kbnr/8/8/8/8/8/8/R2K3R") # king in wrong position
        self.assertFalse(g.can_castle(C.WHITE, True))
        self.assertFalse(g.can_castle(C.WHITE, False))

    def test_check(self):
        g = C.Game()
        self.assertFalse(g.is_check(C.WHITE))
        self.assertFalse(g.is_check(C.BLACK))
        g.board[(4, 6)] = "Q"
        g.board[(4, 1)] = None
        self.assertTrue(g.is_check(C.BLACK))
        self.assertFalse(g.is_check(C.WHITE))

    def test_promotion(self):
        g = C.Game(board="7k/P7/8/8/8/8/8/K7")
        moves = g.get_legal_moves(C.WHITE)
        self.assertTrue(((0, 6), (0, 7), dict()) not in moves)
        self.assertTrue(((0, 6), (0, 7), dict(promo="q", noattack=True)) in moves)
        g.make_move("a7a8q")
        self.assertEqual(g.get_board_state(), "Q6k/8/8/8/8/8/8/K7")

    def test_checkmate(self):
        g = C.Game(board="7k/RR6/8/8/8/8/8/K7")
        self.assertFalse(g.is_checkmate(C.WHITE))
        self.assertFalse(g.is_checkmate(C.BLACK))
        self.assertEqual(g.status, None)
        g.make_move("a7a8")
        self.assertEqual(g.status, "WWINS")
        self.assertTrue(g.is_checkmate(C.BLACK))
        self.assertFalse(g.is_checkmate(C.WHITE))

    def test_stalemate(self):
        g = C.Game(board="7k/R7/8/8/8/8/8/K4R2")
        self.assertFalse(g.is_stalemate())
        g.make_move("f1g1")
        self.assertTrue(g.is_stalemate())

    def test_replay(self):
        g = C.Game()
        g.replay(["e2e4", "a7a6", "f1c4", "a6a5", "d1h5", "a5a4", "h5f7"])
        self.assertTrue(g.is_checkmate(C.BLACK))


def print_board(board):
    print()
    for row in range(7, -1, -1):
        for col in range(8):
            val = board[(col, row)] or "-"
            print(val, end=" ")
        print()


if __name__ == "__main__":
    unittest.main()
