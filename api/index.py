#!/usr/bin/env python3.11

import os
import flask

app = flask.Flask(__name__)
is_local = not os.getenv("VERCEL")
url_prefix = "/api" if is_local else ""

if is_local:
    import sys, pathlib

    sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
    import chess

    @app.route("/")
    def home():
        return flask.send_from_directory("../", "index.html")

else:
    import chess


@app.route(f"{url_prefix}/games", methods=["POST"])
def create_game():
    game = chess.Game()
    game.status = chess.Status.PLAYING
    return flask.make_response(flask.jsonify(game.to_dict()), 201)


@app.route(f"{url_prefix}/games/<id>", methods=["POST"])
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


def main(environ, start_response):  # vercel
    return app(environ, start_response)


if __name__ == "__main__":  # local dev
    app.run(port=3000, debug=True)
