import json, os
import http.client
import urllib.parse
from http.server import BaseHTTPRequestHandler
from chess import Game, Status


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        """Create a new game"""
        game = Game()
        game.status = Status.PLAYING
        self.save_game(game)
        self.send_response(201)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(game.to_dict()).encode("utf-8"))

    def save_game(self, game):
        conn = http.client.HTTPSConnection(os.environ["KV_REST_API_URL"].replace("https://", ""))
        headers = {
            "Authorization": f"Bearer " + os.environ["KV_REST_API_TOKEN"],
            "Content-Type": "application/json",
        }
        id = urllib.parse.quote(game.id, safe='')
        fen = urllib.parse.quote(game.fen, safe='')
        print("*id", id, 'fen', fen)
        conn.request("GET", f"/set/{id}/{fen}", None, headers)
        res = conn.getresponse()
        data = json.loads(res.read().decode("utf-8"))
        print("*SET*", data)
