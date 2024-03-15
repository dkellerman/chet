import json
from chess import Game
from http.server import BaseHTTPRequestHandler


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        game = Game()
        board = game.get_board_state()
        data = {
            "id": "game_id_123",
            "board": board,
        }

        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))
        return
