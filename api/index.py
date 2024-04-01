#!/usr/bin/env python

import flask
import chess

app = flask.Flask(__name__)


@app.route("/")
def home():
    return flask.send_from_directory("../", "index.html")


@app.route(f"/games", methods=["POST"])
def create_game():
    game = chess.Game()
    game.status = chess.Status.PLAYING
    return flask.make_response(flask.jsonify(game.to_dict()), 201)


@app.route(f"/games/<id>", methods=["POST"])
def make_move(id):
    data = flask.request.get_json()
    fen, move = data["fen"], data["move"]
    game = chess.Game(id=id, fen=fen)
    if not game.is_legal_move(move):
        return flask.jsonify({"error": "Illegal move"}), 400
    game.make_move(move)
    if not game.is_ended:
        player = chess.Computer()
        move = player.get_move(game)
        game.make_move(move)
    return flask.jsonify(game.to_dict())


if __name__ == "__main__":
    app.run(port=3000, debug=True)
