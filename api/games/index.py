import json
from http.server import BaseHTTPRequestHandler
from chess import Game


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        """Create a new game"""
        game = Game()
        self.send_response(201)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(game.to_dict()).encode("utf-8"))
