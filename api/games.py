import json, uuid
from chess import Game
from http.server import BaseHTTPRequestHandler

class handler(BaseHTTPRequestHandler):
    def load_games(self):
        with open("./games.json", "r") as file:
            return json.load(file)

    def save_games(self, games):
        with open("games.json", "w") as file:
            json.dump(games, file)

    def send_json(self, data, status=200):
        self.send_response(status)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))

    def do_POST(self):
        game_id = uuid.uuid4().hex[:8]
        game = Game()
        state = game.get_board_state(full=True)
        data = {"id": game_id, "state": state}
        games = self.load_games()
        games[game_id] = data
        self.save_games(games)
        self.send_json(data)

    def do_GET(self):
        game_id = self.path.split("/")[-1]
        games = self.load_games()
        if game_id in games:
            self.send_json(games[game_id])
        else:
            self.send_json({"error": "Game not found"}, status=404)
