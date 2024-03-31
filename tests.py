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
        self.assertEqual(g.turn, 0)
        self.assertEqual(g.cur_color, WHITE)
        self.assertEqual(g.draw_counter, 0)
        self.assertEqual(g.enpassant, None)
        self.assertEqual(g.castles, ["K", "Q", "k", "q"])

    def test_get_attacks(self):
        g = Game()
        g.turn = 1
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

    def test_notation(self):
        g = Game()
        self.assertEqual(Move((0, 1), (0, 3)).to_notation(g), "a2a4")
        self.assertEqual(Move.parse("a2a4", g), Move((0, 1), (0, 3)))
        self.assertEqual(Move.parse("e2e4", g), Move((4, 1), (4, 3)))
        self.assertEqual(Move((0, 6), (0, 7), promo="q").to_notation(g), "a7xa8=q")
        self.assertEqual(Move.parse("a2a4q", g), Move((0, 1), (0, 3), promo="q"))
        self.assertEqual(Move.parse("a2a4=q", g), Move((0, 1), (0, 3), promo="q"))
        self.assertEqual(Move.parse("a2a4=Q", g), Move((0, 1), (0, 3), promo="q"))
        self.assertEqual(Move.parse("a2a4=p", g), None)
        self.assertEqual(Move.parse("a2a4p", g), None)
        self.assertEqual(Move.parse("e4", g), Move((4, 1), (4, 3)))
        self.assertEqual(Move.parse("Pe4", g), Move((4, 1), (4, 3)))
        self.assertEqual(Move.parse("pe4", g), None)
        self.assertEqual(Move.parse("Nf3", g), Move((6, 0), (5, 2)))
        self.assertEqual(Move.parse("Nxf3", g), Move((6, 0), (5, 2)))
        self.assertEqual(Move.parse("Ngf3", g), Move((6, 0), (5, 2)))
        self.assertEqual(Move.parse("N1f3", g), Move((6, 0), (5, 2)))
        # illegal moves are actually ok if it's good form
        self.assertEqual(Move.parse("h1f3", g), Move((7, 0), (5, 2)))
        # but you can't implicitly move a piece that doesn't exist (or wrong color)
        self.assertEqual(Move.parse("Ncf3", g), None)
        self.assertEqual(Move.parse("N2f3", g), None)
        self.assertEqual(Move.parse("e1", g), None)
        self.assertEqual(Move.parse("N8f3", g), None)
        # ignores annotations
        self.assertEqual(Move.parse("Nf3+", g), Move((6, 0), (5, 2)))
        self.assertEqual(Move.parse("Nf3++", g), Move((6, 0), (5, 2)))
        self.assertEqual(Move.parse("Nf3#", g), Move((6, 0), (5, 2)))
        self.assertEqual(Move.parse("Nf3?", g), Move((6, 0), (5, 2)))
        self.assertEqual(Move.parse("Nf3!", g), Move((6, 0), (5, 2)))
        self.assertEqual(Move.parse("Nf3!!", g), Move((6, 0), (5, 2)))
        self.assertEqual(Move.parse("Nf3??", g), Move((6, 0), (5, 2)))
        self.assertEqual(Move.parse("Nf3?!", g), Move((6, 0), (5, 2)))
        self.assertEqual(Move.parse("Nf3+?!", g), Move((6, 0), (5, 2)))
        self.assertEqual(Move.parse("Nf3?!+", g), Move((6, 0), (5, 2)))
        self.assertEqual(Move.parse("e4xd3ep", g), Move((4, 3), (3, 2)))
        self.assertEqual(Move.parse("e4xd3e.p", g), Move((4, 3), (3, 2)))
        self.assertEqual(Move.parse("e4xd3e.p.", g), Move((4, 3), (3, 2)))
        self.assertEqual(Move.parse("e4xd3EP", g), Move((4, 3), (3, 2)))
        self.assertEqual(Move.parse("e4xd3 eP", g), Move((4, 3), (3, 2)))
        self.assertEqual(Move.parse("e4xd3 EP", g), Move((4, 3), (3, 2)))
        # castles
        self.assertEqual(Move.parse("O-O", g), Move((4, 0), (6, 0), castle="K"))
        self.assertEqual(Move.parse("O-O-O+", g), Move((4, 0), (2, 0), castle="Q"))
        self.assertEqual(Move.parse("o-o", g), Move((4, 0), (6, 0), castle="K"))
        self.assertEqual(Move.parse("o-o-o+", g), Move((4, 0), (2, 0), castle="Q"))
        self.assertEqual(Move.parse("O-O", g), Move((4, 0), (6, 0), castle="K"))
        self.assertEqual(Move.parse("0-0-0#", g), Move((4, 0), (2, 0), castle="Q"))
        g.turn = 1
        self.assertEqual(Move.parse("O-O", g), Move((4, 7), (6, 7), castle="k"))
        self.assertEqual(Move.parse("O-O-o!", g), Move((4, 7), (2, 7), castle="q"))
        g.turn = 0
        g.make_moves(["e2e4", "e7e5", "d2d4", "d7d5"])
        self.assertEqual(Move.parse("Bc4", g), Move((5, 0), (2, 3)))
        self.assertEqual(Move.parse("Bxc4", g), Move((5, 0), (2, 3)))
        self.assertEqual(Move.parse("Bc5", g), None)
        self.assertEqual(Move.parse("Bxc5", g), None)
        g.make_move("a1a3")
        self.assertEqual(Move.parse("Bc5", g), Move((5, 7), (2, 4)))
        self.assertEqual(Move.parse("Bxc5", g), Move((5, 7), (2, 4)))
        self.turn = 0
        g = Game("rnbqkbnr/pppppppp/8/8/8/5n2/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        self.assertEqual(Move.parse("gxf3", g), Move.parse("g2xf3", g))
        self.assertEqual(Move.parse("gxf3", g), Move.parse("Pg2xf3", g))

        # actions
        self.assertEqual(
            Move.parse("resign", g), Move(None, None, action=Action.RESIGN)
        )
        self.assertEqual(
            Move.parse("White Resigns", g), Move(None, None, action=Action.RESIGN)
        )

    def test_make_move(self):
        g = Game()
        g.make_move("e2e4")
        self.assertEqual(
            g.fen,
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e4 0 1",
        )
        self.assertEqual(g.cur_color, BLACK)
        self.assertEqual(g.last_move, "e2e4")
        self.assertEqual(g.turn, 1)
        self.assertEqual(g.status, None)
        self.assertEqual(g.status_desc, None)
        g.make_move("e2e4")
        self.assertEqual(g.cur_color, WHITE)
        self.assertEqual(g.turn, 0)
        self.assertEqual(len(g.history), 2)

    def test_pawn_moves(self):
        # basic moves
        g = Game("8/8/8/8/8/8/PPPPPPPP/8 w - - 0 1")
        moves = sorted([m.to_notation(g) for m in g.get_legal_moves()])
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
        moves = sorted([m.to_notation(g) for m in g.get_legal_moves()])
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

        # TODO: must move out of check
        # g = Game("8/4r3/8/8/8/8/P7/4K3 w - - 0 1")
        # g.render_board()
        # self.assertLegalMoves(g, "e1d1|e1d2|e1f1|e1f2")

        # castles: basic
        g = Game("r3kbnr/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
        moves = sorted([m.to_notation(g) for m in g.get_legal_moves()])
        self.assertTrue("e1g1" in moves)
        self.assertTrue("e1c1" in moves)
        g = Game("r3kbnr/8/8/8/8/8/8/R3K2R b KQkq - 0 1")
        moves = sorted([m.to_notation(g) for m in g.get_legal_moves()])
        self.assertTrue("e8g8" not in moves)
        self.assertTrue("e8c8" in moves)

        # castles: r/k has moved or otherwise unavailable
        g = Game("r3kbnr/8/8/8/8/8/8/R3K2R w - - 0 1")
        moves = sorted([m.to_notation(g) for m in g.get_legal_moves()])
        self.assertTrue("e1g1" not in moves)
        self.assertTrue("e1c1" not in moves)
        g = Game("r3kbnr/8/8/8/8/8/8/R3K2R b - - 0 1")
        moves = sorted([m.to_notation(g) for m in g.get_legal_moves()])
        self.assertTrue("e8g8" not in moves)
        self.assertTrue("e8c8" not in moves)

        # castles: squares attacked
        g = Game("r3kbrn/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
        moves = sorted([m.to_notation(g) for m in g.get_legal_moves()])
        self.assertTrue("e1g1" not in moves)
        self.assertTrue("e1c1" in moves)

        # test castling with sequential king/rook moves
        g = Game("r3kbnr/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
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
        moves = sorted([m.to_notation(g) for m in g.get_legal_moves()])
        self.assertLegalMoves(
            g, "e5d6|e5xc7|e5f4|e5h8|e5g7|e5f6|e5d4|e5c3|e5b2|e5a1|g3g4"
        )
        # pinned to row/col
        g = Game("8/4K2/8/4B3/8/4r2/8/8/8 w - - 0 1")
        moves = sorted([m.to_notation(g) for m in g.get_legal_moves()])
        self.assertTrue(len([m for m in moves if m[0:1] == "e5"]) == 0)
        g = Game("8/8/8/r3B2k/8/8/8/8/8 w - - 0 1")
        moves = sorted([m.to_notation(g) for m in g.get_legal_moves()])
        self.assertTrue(len([m for m in moves if m[0:1] == "e5"]) == 0)
        # pinned to diags
        g = Game("8/6K1/8/4B3/3b4/8/8/8 w - - 0 1")
        moves = sorted([m.to_notation(g) for m in g.get_legal_moves()])
        self.assertTrue("e5xd4" in moves)
        self.assertTrue("e5f6" in moves)
        self.assertTrue("e5d6" not in moves)
        self.assertTrue("e5f4" not in moves)
        g = Game("8/6b1/8/4B3/3K4/8/8/8 w - - 0 1")
        moves = sorted([m.to_notation(g) for m in g.get_legal_moves()])
        self.assertTrue("e5xg7" in moves)
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
        self.assertLegalMoves(g, "h5d5|e7d5|e7c6|d4d5|g8d5|e4e3|e4e5|e4f5|e4f4|e4f3|e4d3")

    def test_is_check(self):
        g = Game()
        self.assertFalse(g.is_check())
        g.turn = 1
        self.assertFalse(g.is_check())
        g.board[(4, 6)] = "Q"
        g.board[(4, 1)] = None
        self.assertTrue(g.is_check())
        g.turn = 0
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

    def test_checkmate(self):
        g = Game("7k/RR6/8/8/8/8/8/K7 w - - 0 1")
        self.assertFalse(g.is_checkmate())
        g.turn = 1
        self.assertFalse(g.is_checkmate())
        g.turn = 0
        self.assertEqual(g.status, None)
        g.make_move("a7a8")
        self.assertTrue(g.is_checkmate())
        self.assertEqual(g.status, Status.WWINS)

    def test_stalemate(self):
        g = Game("7k/R7/8/8/8/8/8/K4R2 w - - 0 1")
        self.assertFalse(g.is_stalemate())
        g.make_move("f1g1")
        g.turn = 0
        self.assertFalse(g.is_stalemate())  # not black's turn yet
        g.turn = 1
        self.assertTrue(g.is_stalemate())

    def test_insufficient_material(self):
        # k v k
        g = Game("7k/8/8/8/8/8/8/K7 w - - 0 1")
        self.assertTrue(g.is_ended)
        self.assertEqual(g.status, Status.DRAW)
        self.assertEqual(g.status_desc, 'Insufficient material')
        # k/b v k
        g = Game("7k/8/8/8/8/8/8/KB6 w - - 0 1")
        self.assertEqual(g.status_desc, 'Insufficient material')
        g = Game("6bk/8/8/8/8/8/8/K7 w - - 0 1")
        self.assertEqual(g.status_desc, 'Insufficient material')
        # k/n v k
        g = Game("7k/8/8/8/8/8/8/KN6 w - - 0 1")
        self.assertEqual(g.status_desc, 'Insufficient material')
        g = Game("6nk/8/8/8/8/8/8/K7 w - - 0 1")
        self.assertEqual(g.status_desc, 'Insufficient material')
        # k/b v k/b - same color
        g = Game("6kb/8/8/8/8/8/8/BK6 w - - 0 1")
        self.assertEqual(g.status_desc, 'Insufficient material')
        # k/b v k/b - opp color
        g = Game("6k1/7b/8/8/8/8/8/BK6 w - - 0 1")
        self.assertNotEqual(g.status_desc, 'Insufficient material')

    def test_resign(self):
        g = Game()
        g.make_moves(["e2e4", "a7a6", "resign"])
        self.assertEqual(g.status, Status.BWINS)

    def test_50_move_rule(self):
        g = Game()
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
        self.assertEqual(g.status, Status.DRAW)
        self.assertEqual(g.status_desc, "Fifty-move rule")
        self.assertEqual(g.draw_counter, 50)

    def test_render_board(self):
        g = Game()
        val = _capture_stdout(g.render_board)
        self.assertTrue(val.startswith("\nr n b q k b n r \n"))
        val = _capture_stdout(g.render_board, color=BLACK)
        self.assertTrue(val.startswith("\nR N B K Q B N R \n"))

    def test_allow_king_capture(self):
        g = Game("rnbqkbnr/8/8/8/8/8/8/RNBQKBNR w KQkq - 0 1")
        g.allow_king_capture = True
        self.assertTrue(g.is_legal_move("e1d2"))
        g.make_move("e1d2")
        self.assertTrue(g.is_legal_move("d8xd2"))
        g.make_move("d8xd2")
        self.assertEqual(g.status, Status.BWINS)
        self.assertEqual(g.status_desc, "King captured")

    def test_is_legal_state(self):
        g = Game("rnbqkbnr/8/8/8/8/8/8/RNBQKBNR w KQkq - 0 1")
        self.assertTrue(g.is_legal_state())
        g = Game("rnbqkbnr/8/8/8/8/8/8/RNBKBQNR w KQkq - 0 1")
        self.assertTrue(g.is_legal_state())
        g = Game("rnbkqbnr/8/8/8/8/8/8/RNBQKBNR w KQkq - 0 1")
        self.assertFalse(g.is_legal_state())

    def test_random_boards(self):
        for _ in range(10):
            g = Game()
            g.randomize_board()
            self.assertTrue(g.is_legal_state())

    def test_threefold_rep(self):
        g = Game()
        g.make_moves(["g1f3", "g8f6", "f3g1", "f6g8"])
        self.assertEqual(g.status, None)
        g.make_moves(["g1f3", "g8f6", "f3g1", "f6g8"])
        self.assertEqual(g.status, Status.DRAW)
        self.assertEqual(g.status_desc, "Threefold repetition")

    def test_position_score(self):
        g = Game()
        self.assertEqual(g.get_position_score(), 0.0)
        g.make_moves(["e2e4", "d7d5", "e2xd5"])
        self.assertEqual(g.get_position_score(), 10.0)
        g.board[(3, 0)] = None
        self.assertEqual(g.get_position_score(), -80.0)
        g = Game("7k/RR6/8/8/8/8/8/K7 w - - 0 1")
        g.make_move("a7a8#")
        self.assertEqual(g.get_position_score(), float("inf"))

    def assertLegalMoves(self, g: Game, moves_str: str):
        moves = sorted([m.to_notation(g) for m in g.get_legal_moves()])
        self.assertEqual(moves, sorted(moves_str.split("|")))


def _capture_stdout(func, *args, **kwargs):
    out = io.StringIO()
    sys.stdout = out
    func(*args, **kwargs)
    sys.stdout = sys.__stdout__
    return out.getvalue()


def archive_test():
    with open("./misc/archive.pgn", "r") as file:
        lines = file.readlines()
        game_ct = 0
        for l in lines:
            if l.strip().startswith("1."):
                l = re.sub(r"\d+\.", "", l)
                l = re.sub(r"\{[^\}]*\}", "", l)
                l = re.sub(r"0-1|1-0|1/2-1/2", "", l)
                moves = [m.strip() for m in l.split()]
                game = Game()
                good = True
                for move in moves:
                    if game_ct == 10:
                        print(move)
                    m = Move.parse(move, game)
                    if m is None:
                        print(move, end=" ")
                        good = False
                        break
                    else:
                        game.make_move(m)
                game_ct += 1
                if game_ct > 10:
                    break


if __name__ == "__main__":
    if "-a" in sys.argv:
        archive_test()
    else:
        unittest.main()
