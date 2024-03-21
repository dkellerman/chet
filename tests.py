#!/usr/bin/env python3

import unittest
import chess as C


class TestChess(unittest.TestCase):
    def test_board_state(self):
        st1 = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
        g = C.Game(state=st1)
        st2 = g.get_board_state()
        self.assertEqual(
            st2, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        )
        self.assertEqual(g.turn, 0)
        self.assertEqual(g.cur_color, C.WHITE)
        self.assertEqual(g.draw_counter, 0)
        self.assertEqual(g.half_move_counter, 1)
        self.assertEqual(g.enpassant, None)
        self.assertEqual(g.castles, ["K", "Q", "k", "q"])

    def test_notation(self):
        g = C.Game()

        self.assertEqual(g.to_notation((0, 1), (0, 3), {}), "a2a4")
        self.assertEqual(g.parse_notation("a2a4"), ((0, 1), (0, 3), {}))
        self.assertEqual(g.parse_notation("e2e4"), ((4, 1), (4, 3), {}))
        self.assertEqual(g.to_notation((0, 6), (0, 7), dict(promo="q")), "a7xa8q")
        self.assertEqual(g.parse_notation("a2a4q"), ((0, 1), (0, 3), dict(promo="q")))
        self.assertEqual(g.parse_notation("a2a4=q"), ((0, 1), (0, 3), dict(promo="q")))
        self.assertEqual(g.parse_notation("a2a4=Q"), ((0, 1), (0, 3), dict(promo="q")))
        self.assertEqual(g.parse_notation("a2a4=p"), None)
        self.assertEqual(g.parse_notation("a2a4p"), None)

        self.assertEqual(g.parse_notation("e4"), ((4, 1), (4, 3), dict()))
        self.assertEqual(g.parse_notation("Pe4"), ((4, 1), (4, 3), dict()))
        self.assertEqual(g.parse_notation("pe4"), None)

        self.assertEqual(g.parse_notation("Nf3"), ((6, 0), (5, 2), dict()))
        self.assertEqual(g.parse_notation("Nxf3"), ((6, 0), (5, 2), dict()))
        self.assertEqual(g.parse_notation("Ngf3"), ((6, 0), (5, 2), dict()))
        self.assertEqual(g.parse_notation("N1f3"), ((6, 0), (5, 2), dict()))
        # illegal moves are actually ok if it's good form
        self.assertEqual(g.parse_notation("h1f3"), ((7, 0), (5, 2), dict()))
        # but you can't implicitly move a piece that doesn't exist (or wrong color)
        self.assertEqual(g.parse_notation("Ncf3"), None)
        self.assertEqual(g.parse_notation("N2f3"), None)
        self.assertEqual(g.parse_notation("e1"), None)
        self.assertEqual(g.parse_notation("N8f3"), None)

        # ignores annotations
        self.assertEqual(g.parse_notation("Nf3+"), ((6, 0), (5, 2), dict()))
        self.assertEqual(g.parse_notation("Nf3++"), ((6, 0), (5, 2), dict()))
        self.assertEqual(g.parse_notation("Nf3#"), ((6, 0), (5, 2), dict()))
        self.assertEqual(g.parse_notation("Nf3?"), ((6, 0), (5, 2), dict()))
        self.assertEqual(g.parse_notation("Nf3!"), ((6, 0), (5, 2), dict()))
        self.assertEqual(g.parse_notation("Nf3!!"), ((6, 0), (5, 2), dict()))
        self.assertEqual(g.parse_notation("Nf3??"), ((6, 0), (5, 2), dict()))
        self.assertEqual(g.parse_notation("Nf3?!"), ((6, 0), (5, 2), dict()))
        self.assertEqual(g.parse_notation("Nf3+?!"), ((6, 0), (5, 2), dict()))
        self.assertEqual(g.parse_notation("Nf3?!+"), ((6, 0), (5, 2), dict()))
        self.assertEqual(g.parse_notation("e4xd3ep"), ((4, 3), (3, 2), dict()))
        self.assertEqual(g.parse_notation("e4xd3e.p"), ((4, 3), (3, 2), dict()))
        self.assertEqual(g.parse_notation("e4xd3e.p."), ((4, 3), (3, 2), dict()))
        self.assertEqual(g.parse_notation("e4xd3EP"), ((4, 3), (3, 2), dict()))
        self.assertEqual(g.parse_notation("e4xd3 eP"), ((4, 3), (3, 2), dict()))
        self.assertEqual(g.parse_notation("e4xd3 EP"), ((4, 3), (3, 2), dict()))

        # castles
        self.assertEqual(g.parse_notation("O-O"), ((4, 0), (6, 0), dict(castle="K")))
        self.assertEqual(g.parse_notation("O-O-O+"), ((4, 0), (2, 0), dict(castle="Q")))
        self.assertEqual(g.parse_notation("o-o"), ((4, 0), (6, 0), dict(castle="K")))
        self.assertEqual(g.parse_notation("o-o-o+"), ((4, 0), (2, 0), dict(castle="Q")))
        self.assertEqual(g.parse_notation("O-O"), ((4, 0), (6, 0), dict(castle="K")))
        self.assertEqual(g.parse_notation("0-0-0#"), ((4, 0), (2, 0), dict(castle="Q")))
        g.turn = 1
        self.assertEqual(g.parse_notation("O-O"), ((4, 7), (6, 7), dict(castle="k")))
        self.assertEqual(g.parse_notation("O-O-o!"), ((4, 7), (2, 7), dict(castle="q")))
        g.turn = 0

        g.make_move(["e2e4", "e7e5", "d2d4", "d7d5"])
        self.assertEqual(g.parse_notation("Bc4"), ((5, 0), (2, 3), dict()))
        self.assertEqual(g.parse_notation("Bxc4"), ((5, 0), (2, 3), dict()))
        self.assertEqual(g.parse_notation("Bc5"), None)
        self.assertEqual(g.parse_notation("Bxc5"), None)

        g.make_move("a1a3")
        self.assertEqual(g.parse_notation("Bc5"), ((5, 7), (2, 4), dict()))
        self.assertEqual(g.parse_notation("Bxc5"), ((5, 7), (2, 4), dict()))
        self.turn = 0

        # actions
        self.assertEqual(
            g.parse_notation("resign"), (None, None, dict(action="resign"))
        )
        self.assertEqual(
            g.parse_notation("White Resigns"), (None, None, dict(action="resign"))
        )
        self.assertEqual(g.parse_notation("draw"), (None, None, dict(action="draw")))
        self.assertEqual(
            g.parse_notation("White offers a draw"), (None, None, dict(action="draw"))
        )


    def test_make_move(self):
        g = C.Game()
        g.make_move("e2e4")
        self.assertEqual(
            g.get_board_state(),
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e4 0 1",
        )
        self.assertEqual(g.cur_color, C.BLACK)
        self.assertEqual(g.last_move, "e2e4")
        self.assertEqual(g.turn, 1)
        self.assertEqual(g.status, None)
        self.assertEqual(g.status_desc, None)
        g.make_move("e2e4")
        self.assertEqual(g.cur_color, C.WHITE)
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

        g = C.Game(state="rnbqkbnr/8/8/8/8/8/8/RNBQKBNR")
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
        g = C.Game(state="r3kbnr/8/8/8/8/8/8/R3K2R")
        moves = sorted([g.to_notation(*m) for m in g.get_legal_moves()])
        self.assertTrue("e1g1" in moves)
        self.assertTrue(g.can_castle(C.WHITE, True))
        self.assertTrue("e1c1" in moves)
        self.assertTrue(g.can_castle(C.WHITE, False))
        self.assertTrue("e8g8" not in moves)
        self.assertFalse(g.can_castle(C.BLACK, True))
        self.assertTrue("e8c8" in moves)
        self.assertTrue(g.can_castle(C.BLACK, False))

        g.castles = []
        moves = sorted([g.to_notation(*m) for m in g.get_legal_moves()])
        self.assertTrue("e1g1" not in moves)
        self.assertTrue("e1c1" not in moves)
        self.assertTrue("e8g8" not in moves)
        self.assertTrue("e8c8" not in moves)

        g = C.Game(state="r3kbrn/8/8/8/8/8/8/R3K2R")  # attacked
        self.assertFalse(g.can_castle(C.WHITE, True))
        self.assertTrue(g.can_castle(C.WHITE, False))
        g.make_move("e1d1")  # move king
        self.assertFalse(g.can_castle(C.WHITE, False))

        g = C.Game(state="r3kbnr/8/8/8/8/8/8/R3K2R")
        self.assertTrue(g.can_castle(C.WHITE, True))
        self.assertTrue(g.can_castle(C.WHITE, False))
        g.make_move("a1a2")  # move A rook
        self.assertTrue(g.can_castle(C.WHITE, True))
        self.assertFalse(g.can_castle(C.WHITE, False))
        g.make_move("h1h2")  # move H rook
        self.assertFalse(g.can_castle(C.WHITE, True))

        g = C.Game(state="r3kbnr/8/8/8/8/8/8/R2K3R")  # king in wrong position
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

    def test_enpassant(self):
        g = C.Game()
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
        self.assertTrue(g.is_legal_move("e5xd6", C.WHITE))
        self.assertTrue(g.is_legal_move("e5d6", C.WHITE))
        g.make_move("e5xd6")
        self.assertEqual(g.enpassant, None)

    def test_promotion(self):
        g = C.Game(state="7k/P7/8/8/8/8/8/K7")
        moves = g.get_legal_moves(C.WHITE)
        self.assertTrue(((0, 6), (0, 7), dict()) not in moves)
        self.assertTrue(((0, 6), (0, 7), dict(promo="q", noattack=True)) in moves)
        g.make_move("a7a8q")
        self.assertRegex(g.get_board_state(), r"^Q6k/8/8/8/8/8/8/K7 b ")

        # with capture
        g = C.Game(state="r6k/1P6/8/8/8/8/8/K7")
        moves = g.get_legal_moves(C.WHITE)
        self.assertTrue(((1, 6), (0, 7), dict()) not in moves)
        self.assertTrue(((1, 6), (0, 7), dict(promo="q")) in moves)
        g.make_move("b7xa8q")
        self.assertRegex(g.get_board_state(), r"^Q6k/8/8/8/8/8/8/K7")

    def test_checkmate(self):
        g = C.Game(state="7k/RR6/8/8/8/8/8/K7")
        self.assertFalse(g.is_checkmate(C.WHITE))
        self.assertFalse(g.is_checkmate(C.BLACK))
        self.assertEqual(g.status, None)
        g.make_move("a7a8")
        self.assertEqual(g.status, "WWINS")
        self.assertTrue(g.is_checkmate(C.BLACK))
        self.assertFalse(g.is_checkmate(C.WHITE))

    def test_stalemate(self):
        g = C.Game(state="7k/R7/8/8/8/8/8/K4R2")
        self.assertFalse(g.is_stalemate())
        g.make_move("f1g1")
        g.turn = 0
        self.assertFalse(g.is_stalemate())  # not black's turn yet
        g.turn = 1
        self.assertTrue(g.is_stalemate())

    def test_resign(self):
        g = C.Game()
        g.make_move(["e2e4", "a7a6", "resign"])
        self.assertEqual(g.status, "BWINS")

    def test_50_move_rule(self):
        g = C.Game()
        for _ in range(24):
            g.make_move("g1f3")
            g.make_move("g8f6")
            g.make_move("f3g1")
            g.make_move("f6g8")
        self.assertEqual(g.status, None)
        self.assertEqual(g.half_move_counter, 49)
        self.assertEqual(g.draw_counter, 48)

        g.make_move("g1f3")
        g.make_move("g8f6")
        g.make_move("f3g1")
        g.make_move("f6g8")
        self.assertEqual(g.status, "DRAW")
        self.assertEqual(g.status_desc, "Fifty-move rule")
        self.assertEqual(g.half_move_counter, 51)
        self.assertEqual(g.draw_counter, 50)


if __name__ == "__main__":
    unittest.main()
