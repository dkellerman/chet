import urllib.parse
import http.client
import json
import os
from chess import Game, Computer
from http.server import BaseHTTPRequestHandler


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        """Make a move"""
        data = self.read_json()
        id, fen, move = data["id"], data["fen"], data["move"]
        game = Game(id=id, fen=fen)
        if not game.is_legal_move(move):
            return self.write_json({"error": "Illegal move"}, status=400)
        game.make_move(move)
        if not game.is_ended:
            player = Computer()
            move = player.get_move(game)
            game.make_move(move)
        return self.write_json(game.to_dict())

    def read_json(self):
        content_length = int(self.headers["Content-Length"])
        post_data = self.rfile.read(content_length)
        return json.loads(post_data)

    def write_json(self, data, status=200):
        self.send_response(status)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))
