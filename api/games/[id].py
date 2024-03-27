import json
from chess import Game, Computer
from http.server import BaseHTTPRequestHandler


def with_game(func):
    def wrapper(self, *args, **kwargs):
        game_id = self.path.split("/")[-1]
        if game_id:
            games = {}
            game = games.get(game_id)
            return func(self, game, *args, **kwargs)
        else:
            return None

    return wrapper


class handler(BaseHTTPRequestHandler):
    @with_game
    def do_GET(self, game):
        """Get game by ID"""
        if game:
            self.write_json(game.to_dict())
        else:
            self.write_json({"error": "Game not found"}, status=404)

    @with_game
    def do_POST(self, game):
        """Make a move"""
        data = self.read_json()
        move = data["move"]
        if not game.is_legal_move():
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
