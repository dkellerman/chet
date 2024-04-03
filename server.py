#!/usr/bin/env python3.11

import flask
import uuid
import chess

app = flask.Flask(__name__)


@app.route("/")
def home():
    return flask.send_from_directory("./", "index.html")


@app.route(f"/games", methods=["POST"])
def create_game():
    game = chess.Game()
    id = uuid.uuid4().hex
    return flask.make_response(flask.jsonify(game2dict(id, game)), 201)


@app.route(f"/games/<id>", methods=["POST"])
def make_move(id):
    data = flask.request.get_json()
    fen, move = data["fen"], data["move"]
    game = chess.Game(fen=fen)
    if not game.is_legal_move(move):
        return flask.jsonify({"error": "Illegal move"}), 400
    game.make_move(move)
    if not game.is_ended:
        player = chess.Computer()
        move = player.get_move(game)
        game.make_move(move)
    return flask.jsonify(game2dict(id, game))


def game2dict(id, game):
    return {
        "id": id,
        "fen": game.fen,
        "status": game.status[0],
        "status_desc": game.status[1],
        "players": [p.__class__.__qualname__ for p in game.players],
        "history": game.history,
        "legal_moves": [m.to_notation(game) for m in game.get_legal_moves()],
        "last_move": game.last_move if game.last_move else None,
    }


if __name__ == "__main__":
    app.run(port=3000, debug=True)
