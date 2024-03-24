#!/usr/bin/env python3

import unittest, sys, io
import chess as C


class TestChess(unittest.TestCase):
    def test_state(self):
        st1 = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        g = C.Game(state=st1)
        st2 = g.state
        self.assertEqual(
            st2, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        )
        self.assertEqual(g.turn, 0)
        self.assertEqual(g.cur_color, C.WHITE)
        self.assertEqual(g.draw_counter, 0)
        self.assertEqual(g.enpassant, None)
        self.assertEqual(g.castles, ["K", "Q", "k", "q"])

    def test_get_attacks(self):
        g = C.Game()
        g.turn = 1
        attacks, pin = g.get_attacks()
        self.assertEqual(
            " ".join([C.c2sq(sq) for sq in sorted(attacks)]),
            " ".join(
                sorted(
                    "b1 c1 d1 e1 f1 g1 a2 b2 c2 d2 e2 f2 g2 h2 a3 "
                    "b3 c3 d3 e3 f3 g3 h3".split()
                )
            ),
        )
        self.assertIsNone(pin)

        g = C.Game("rnbqkbnr/8/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        attacks, pin = g.get_attacks()
        self.assertEqual(
            " ".join([C.c2sq(sq) for sq in sorted(attacks)]),
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

        g = C.Game("rnb1kbnr/4q3/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        attacks, pin = g.get_attacks()
        self.assertEqual(pin, (C.sq2c("e2"), (0, -1)))

        g = C.Game("8/8/8/8/6b1/5B2/8/3K4 w KQkq - 0 1")
        attacks, pin = g.get_attacks()
        self.assertEqual(pin, (C.sq2c("f3"), (-1, -1)))

    def test_notation(self):
        g = C.Game()

        self.assertEqual(g.to_notation((0, 1), (0, 3), {}), "a2a4")
        self.assertEqual(g.parse_notation("a2a4"), ((0, 1), (0, 3), {}))
        self.assertEqual(g.parse_notation("e2e4"), ((4, 1), (4, 3), {}))
        self.assertEqual(g.to_notation((0, 6), (0, 7), dict(promo="q")), "a7xa8=q")
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

        g.make_moves(["e2e4", "e7e5", "d2d4", "d7d5"])
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
            g.state,
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

    def test_pawn_moves(self):
        # basic moves
        g = C.Game(state="8/8/8/8/8/8/PPPPPPPP/8 w - - 0 1")
        moves = sorted([g.to_notation(*m) for m in g.get_legal_moves()])
        self.assertEqual(
            moves,
            sorted(
                "a2a3|a2a4|b2b3|b2b4|c2c3|c2c4|d2d3|d2d4|e2e3|e2e4|f2f3|f2f4|"
                "g2g3|g2g4|h2h3|h2h4".split("|")
            ),
        )
        # 1-push
        g = C.Game(state="8/8/P7/8/8/8/8/8 w - - 0 1")
        moves = sorted([g.to_notation(*m) for m in g.get_legal_moves()])
        self.assertEqual(moves, ["a6a7"])
        # promos
        g = C.Game(state="1q6/P7/8/8/8/8/8/8 w - - 0 1")
        moves = sorted([g.to_notation(*m) for m in g.get_legal_moves()])
        self.assertEqual(
            moves,
            sorted(
                "a7a8=q|a7a8=r|a7a8=b|a7a8=n|a7xb8=q|a7xb8=r|a7xb8=b|a7xb8=n".split("|")
            ),
        )
        # pin diagonal
        g = C.Game(state="4k3/5p2/4N1B1/8/8/8/8/8 b - - 0 1")
        moves = sorted([g.to_notation(*m) for m in g.get_legal_moves()])
        self.assertEqual(moves, sorted("f7xg6|e8e7|e8d7".split("|")))
        # pin on col
        g = C.Game(state="4k3/4p3/3Q1B2/8/8/8/8/4R3 b - - 0 1")
        moves = sorted([g.to_notation(*m) for m in g.get_legal_moves()])
        self.assertEqual(moves, sorted("e7e6|e7e5|e8f8|e8f7".split("|")))
        # pin on row
        g = C.Game(state="8/k3p2R/8/8/8/8/8/8 b - - 0 1")
        moves = sorted([g.to_notation(*m) for m in g.get_legal_moves()])
        self.assertEqual(moves, sorted("a7a8|a7a6|a7b8|a7b7|a7b6".split("|")))
        # enpassant
        g = C.Game(state="8/8/8/2Pp4/8/8/8/8 w - d5 0 1")
        moves = sorted([g.to_notation(*m) for m in g.get_legal_moves()])
        self.assertEqual(moves, ["c5c6", "c5xd6"])

        # TODO: enpassant w pin!
        # g = C.Game(state="8/8/8/K1Pp3r/8/8/8/8 w - d5 0 1")
        # moves = sorted([g.to_notation(*m) for m in g.get_legal_moves()])
        # self.assertEqual(moves, ["c5c6"])

    def test_knight_moves(self):
        g = C.Game(state="8/2B1b3/8/3N4/8/8/8/N7 w - - 0 1")
        moves = sorted([g.to_notation(*m) for m in g.get_legal_moves()])
        self.assertEqual(
            moves,
            sorted(
                "d5b6|d5b4|d5c3|d5e3|d5f4|d5f6|d5xe7|a1b3|a1c2|"
                "c7b8|c7d6|c7e5|c7f4|c7g3|c7h2|c7d8|c7b6|c7a5".split("|")
            ),
        )

    def test_king_moves(self):
        # basic
        g = C.Game(state="8/8/8/8/8/8/8/4K3 w - - 0 1")
        moves = sorted([g.to_notation(*m) for m in g.get_legal_moves()])
        self.assertEqual(moves, sorted("e1d1|e1d2|e1e2|e1f2|e1f1".split("|")))

        # capture
        g = C.Game(state="8/8/8/8/8/8/4pP2/4K3 w - - 0 1")
        moves = sorted([g.to_notation(*m) for m in g.get_legal_moves()])
        self.assertEqual(moves, sorted("e1xe2|e1d2|f2f3|f2f4".split("|")))

        # can't move into check
        g = C.Game(state="3r4/8/8/8/8/8/r7/4K3 w - - 0 1")
        moves = sorted([g.to_notation(*m) for m in g.get_legal_moves()])
        self.assertEqual(moves, sorted("e1f1".split("|")))

        # castles: basic
        g = C.Game(state="r3kbnr/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
        moves = sorted([g.to_notation(*m) for m in g.get_legal_moves()])
        self.assertTrue("e1g1" in moves)
        self.assertTrue("e1c1" in moves)
        g = C.Game(state="r3kbnr/8/8/8/8/8/8/R3K2R b KQkq - 0 1")
        moves = sorted([g.to_notation(*m) for m in g.get_legal_moves()])
        self.assertTrue("e8g8" not in moves)
        self.assertTrue("e8c8" in moves)

        # castles: r/k has moved or otherwise unavailable
        g = C.Game(state="r3kbnr/8/8/8/8/8/8/R3K2R w - - 0 1")
        moves = sorted([g.to_notation(*m) for m in g.get_legal_moves()])
        self.assertTrue("e1g1" not in moves)
        self.assertTrue("e1c1" not in moves)
        g = C.Game(state="r3kbnr/8/8/8/8/8/8/R3K2R b - - 0 1")
        moves = sorted([g.to_notation(*m) for m in g.get_legal_moves()])
        self.assertTrue("e8g8" not in moves)
        self.assertTrue("e8c8" not in moves)

        # castles: squares attacked
        g = C.Game(state="r3kbrn/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
        moves = sorted([g.to_notation(*m) for m in g.get_legal_moves()])
        self.assertTrue("e1g1" not in moves)
        self.assertTrue("e1c1" in moves)

        # test castling with sequential king/rook moves
        g = C.Game(state="r3kbnr/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
        self.assertTrue(g.is_legal_move("e1g1"))
        self.assertTrue(g.is_legal_move("o-o"))
        self.assertTrue(g.is_legal_move("e1c1"))
        self.assertTrue(g.is_legal_move("0-0-0"))
        g.make_move("a1a2")  # move A rook
        self.assertTrue(g.is_legal_move("e8c8"))
        self.assertFalse(g.is_legal_move("e8g8"))
        g.make_move("o-o-o")
        self.assertTrue(g.is_legal_move("e1g1"))
        self.assertFalse(g.is_legal_move("e1c1"))
        g.make_moves(["a8a7", "h1h2"])  # move H rook
        self.assertFalse(g.is_legal_move("e1g1"))
        self.assertFalse(g.is_legal_move("e1c1"))
        g = C.Game(state="r3kbnr/8/8/8/8/8/8/R2K3R w KQkq - 0 1")
        g.make_move("e1e2")  # move king
        self.assertFalse(g.is_legal_move("e1g1"))
        self.assertFalse(g.is_legal_move("e1c1"))

    def test_rook_moves(self):
        # basic
        g = C.Game(state="8/4p3/8/4R3/8/4P3/8/8 w - - 0 1")
        moves = sorted([g.to_notation(*m) for m in g.get_legal_moves()])
        self.assertEqual(
            moves,
            sorted(
                "e5d5|e5c5|e5b5|e5a5|e5f5|e5g5|e5h5|e5e4|e5e6|e5xe7|e3e4".split("|")
            ),
        )
        # pinned diagonal
        g = C.Game(state="K7/1R6/2b5/8/8/8/8/8 w - - 0 1")
        moves = sorted([g.to_notation(*m) for m in g.get_legal_moves()])
        self.assertEqual(moves, sorted("a8a7|a8b8".split("|")))
        # pinned to row
        g = C.Game(state="KR1r4/8/8/8/8/8/8/8 w - - 0 1")
        moves = sorted([g.to_notation(*m) for m in g.get_legal_moves()])
        self.assertEqual(moves, sorted("a8a7|a8b7|b8c8|b8xd8".split("|")))
        # pinned to col
        g = C.Game(state="K8/R7/8/8/8/8/r7/8 w - - 0 1")
        moves = sorted([g.to_notation(*m) for m in g.get_legal_moves()])
        self.assertEqual(
            moves, sorted("a8b8|a8b7|a7a6|a7a5|a7a4|a7a3|a7xa2".split("|"))
        )

    def test_bishop_moves(self):
        # basic
        g = C.Game(state="8/2p5/8/4B3/8/6P1/8/8 w - - 0 1")
        moves = sorted([g.to_notation(*m) for m in g.get_legal_moves()])
        self.assertEqual(
            moves,
            sorted(
                "e5d6|e5xc7|e5f4|e5h8|e5g7|e5f6|e5d4|e5c3|e5b2|e5a1|g3g4".split("|")
            ),
        )
        # pinned to row/col
        g = C.Game(state="8/4K2/8/4B3/8/4r2/8/8/8 w - - 0 1")
        moves = sorted([g.to_notation(*m) for m in g.get_legal_moves()])
        self.assertTrue(len([m for m in moves if m[0:1] == "e5"]) == 0)
        g = C.Game(state="8/8/8/r3B2k/8/8/8/8/8 w - - 0 1")
        moves = sorted([g.to_notation(*m) for m in g.get_legal_moves()])
        self.assertTrue(len([m for m in moves if m[0:1] == "e5"]) == 0)
        # pinned to diags
        g = C.Game(state="8/6K1/8/4B3/3b4/8/8/8 w - - 0 1")
        moves = sorted([g.to_notation(*m) for m in g.get_legal_moves()])
        self.assertTrue("e5xd4" in moves)
        self.assertTrue("e5f6" in moves)
        self.assertTrue("e5d6" not in moves)
        self.assertTrue("e5f4" not in moves)
        g = C.Game(state="8/6b1/8/4B3/3K4/8/8/8 w - - 0 1")
        moves = sorted([g.to_notation(*m) for m in g.get_legal_moves()])
        self.assertTrue("e5xg7" in moves)
        self.assertTrue("e5f6" in moves)
        self.assertTrue("e5d6" not in moves)
        self.assertTrue("e5f4" not in moves)

    def test_queen_moves(self):
        # basic
        g = C.Game(state="8/4p3/8/4Q3/8/4P3/8/8 w - - 0 1")
        moves = sorted([g.to_notation(*m) for m in g.get_legal_moves()])
        self.assertEqual(
            moves,
            sorted(
                "e5e6|e5xe7|e5e4|e5d5|e5c5|e5b5|e5a5|e5f5|e5g5|e5h5|"
                "e5a1|e5b2|e5c3|e5d4|e5f6|e5g7|e5h8|"
                "e5d6|e5c7|e5b8|e5f4|e5g3|e5h2|e3e4".split("|")
            ),
        )
        # pinned diagonal
        g = C.Game(state="K7/1Q6/2b5/8/8/8/8/8 w - - 0 1")
        moves = sorted([g.to_notation(*m) for m in g.get_legal_moves()])
        self.assertEqual(moves, sorted("a8a7|a8b8|b7xc6".split("|")))
        # pinned to row
        g = C.Game(state="KQ1r4/8/8/8/8/8/8/8 w - - 0 1")
        moves = sorted([g.to_notation(*m) for m in g.get_legal_moves()])
        self.assertEqual(moves, sorted("a8a7|a8b7|b8c8|b8xd8".split("|")))
        # pinned to col
        g = C.Game(state="K8/Q7/8/8/8/8/r7/8 w - - 0 1")
        moves = sorted([g.to_notation(*m) for m in g.get_legal_moves()])
        self.assertEqual(
            moves, sorted("a8b8|a8b7|a7a6|a7a5|a7a4|a7a3|a7xa2".split("|"))
        )

    def test_check(self):
        g = C.Game()
        self.assertFalse(g.is_check())
        g.turn = 1
        self.assertFalse(g.is_check())
        g.board[(4, 6)] = "Q"
        g.board[(4, 1)] = None
        self.assertTrue(g.is_check())
        g.turn = 0
        self.assertFalse(g.is_check())

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
        self.assertTrue(g.is_legal_move("e5xd6"))
        self.assertTrue(g.is_legal_move("e5d6"))
        g.make_move("e5xd6")
        self.assertEqual(g.enpassant, None)

    def test_checkmate(self):
        g = C.Game(state="7k/RR6/8/8/8/8/8/K7 w - - 0 1")
        self.assertFalse(g.is_checkmate())
        g.turn = 1
        self.assertFalse(g.is_checkmate())
        g.turn = 0
        self.assertEqual(g.status, None)
        g.make_move("a7a8")
        self.assertTrue(g.is_checkmate())
        self.assertEqual(g.status, C.Status.WWINS)

    def test_stalemate(self):
        g = C.Game(state="7k/R7/8/8/8/8/8/K4R2 w - - 0 1")
        self.assertFalse(g.is_stalemate())
        g.make_move("f1g1")
        g.turn = 0
        self.assertFalse(g.is_stalemate())  # not black's turn yet
        g.turn = 1
        self.assertTrue(g.is_stalemate())

    def test_resign(self):
        g = C.Game()
        g.make_moves(["e2e4", "a7a6", "resign"])
        self.assertEqual(g.status, C.Status.BWINS)

    def test_50_move_rule(self):
        g = C.Game()
        self.assertEqual(g.draw_counter, 0)
        for _ in range(12):
            g.make_move("g1f3")
            g.make_move("g8f6")
            g.make_move("f3g1")
            g.make_move("f6g8")
            g.repititions = dict()
        self.assertEqual(g.status, None)
        self.assertEqual(g.draw_counter, 48)

        g.make_move("g1f3")
        g.make_move("g8f6")
        self.assertEqual(g.status, C.Status.DRAW)
        self.assertEqual(g.status_desc, "Fifty-move rule")
        self.assertEqual(g.draw_counter, 50)

    def test_render_board(self):
        g = C.Game()
        val = _capture_stdout(g.render_board)
        self.assertTrue(val.startswith("\nr n b q k b n r \n"))
        val = _capture_stdout(g.render_board, color=C.BLACK)
        self.assertTrue(val.startswith("\nR N B Q K B N R \n"))

    def test_allow_king_capture(self):
        g = C.Game("rnbqkbnr/8/8/8/8/8/8/RNBQKBNR w KQkq - 0 1")
        g._allow_king_capture = True
        self.assertTrue(g.is_legal_move("e1d2"))
        g.make_move("e1d2")
        self.assertTrue(g.is_legal_move("d8xd2"))
        g.make_move("d8xd2")
        self.assertEqual(g.status, C.Status.BWINS)
        self.assertEqual(g.status_desc, "King captured")

    def test_is_legal_state(self):
        g = C.Game("rnbqkbnr/8/8/8/8/8/8/RNBQKBNR w KQkq - 0 1")
        self.assertTrue(g.is_legal_state())
        g = C.Game("rnbqkbnr/8/8/8/8/8/8/RNBKBQNR w KQkq - 0 1")
        self.assertTrue(g.is_legal_state())
        g = C.Game("rnbkqbnr/8/8/8/8/8/8/RNBQKBNR w KQkq - 0 1")
        self.assertFalse(g.is_legal_state())

    def test_random_boards(self):
        for _ in range(10):
            g = C.Game()
            g.randomize_board()
            self.assertTrue(g.is_legal_state())

    def test_threefold_rep(self):
        g = C.Game()
        g.make_moves(["g1f3", "g8f6", "f3g1", "f6g8"])
        self.assertEqual(g.status, None)
        g.make_moves(["g1f3", "g8f6", "f3g1", "f6g8"])
        self.assertEqual(g.status, C.Status.DRAW)
        self.assertEqual(g.status_desc, "Threefold repetition")

    def test_position_score(self):
        g = C.Game()
        self.assertEqual(g.get_position_score(), 0)
        g.make_moves(["e2e4", "d7d5", "e2xd5"])
        self.assertEqual(g.get_position_score(), 1)
        g.board[(3, 0)] = None
        self.assertEqual(g.get_position_score(), -8)
        g = C.Game(state="7k/RR6/8/8/8/8/8/K7 w - - 0 1")
        g.make_move("a7a8#")
        self.assertEqual(g.get_position_score(), float("inf"))


def _capture_stdout(func, *args, **kwargs):
    out = io.StringIO()
    sys.stdout = out
    func(*args, **kwargs)
    sys.stdout = sys.__stdout__
    return out.getvalue()


if __name__ == "__main__":
    unittest.main()
