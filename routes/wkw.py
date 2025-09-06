# ---- add/imports at top of your server file ----
from flask import Flask, request, jsonify
from flask_cors import CORS
import random
import os
import re
from typing import List, Tuple
import logging
from flask import request, jsonify
from routes import app
from flask import Flask, request, jsonify, abort


logger = logging.getLogger(__name__)

# --- Config / logging ---
logging.basicConfig(level=logging.INFO)


from flask import Flask, request, jsonify
import re
import logging

app.url_map.strict_slashes = False   # /2048 and /2048/ behave the same
CORS(app)

N = 4

def rotate_clockwise(grid):
    return [[grid[N-1-r][c] for r in range(N)] for c in range(N)]

def rotate_counterclockwise(grid):
    return [[grid[r][N-1-c] for r in range(N)] for c in range(N-1, -1, -1)]

def slide_left_row(row):
    nums = [x for x in row if x is not None]
    out, i, score_inc = [], 0, 0
    while i < len(nums):
        if i + 1 < len(nums) and nums[i] == nums[i + 1]:
            merged = nums[i] * 2
            out.append(merged)
            score_inc += merged
            i += 2
        else:
            out.append(nums[i])
            i += 1
    while len(out) < N:
        out.append(None)
    return out, score_inc

def move_grid(grid, direction):
    g = [row[:] for row in grid]
    # normalize to LEFT
    if direction == "UP":
        g = rotate_counterclockwise(g)
    elif direction == "DOWN":
        g = rotate_clockwise(rotate_clockwise(rotate_clockwise(g)))
    elif direction == "RIGHT":
        g = rotate_clockwise(rotate_clockwise(g))

    moved = False
    score_gain = 0
    new_g = []
    for row in g:
        new_row, inc = slide_left_row(row)
        if new_row != row:
            moved = True
        score_gain += inc
        new_g.append(new_row)

    # rotate back
    if direction == "UP":
        new_g = rotate_clockwise(new_g)
    elif direction == "DOWN":
        new_g = rotate_counterclockwise(rotate_counterclockwise(rotate_counterclockwise(new_g)))
    elif direction == "RIGHT":
        new_g = rotate_clockwise(rotate_clockwise(new_g))

    return new_g, moved, score_gain

def empty_cells(grid):
    return [(r, c) for r in range(N) for c in range(N) if grid[r][c] is None]

def add_random_tile(grid):
    cells = empty_cells(grid)
    if not cells:
        return
    r, c = random.choice(cells)
    grid[r][c] = 2 if random.random() < 0.9 else 4

def has_2048(grid):
    return any(cell == 2048 for row in grid for cell in row if cell)

def can_move(grid):
    if empty_cells(grid):
        return True
    for r in range(N):
        for c in range(N):
            if r + 1 < N and grid[r][c] == grid[r + 1][c]:
                return True
            if c + 1 < N and grid[r][c] == grid[r][c + 1]:
                return True
    return False

@app.route("/2048", methods=["POST", "OPTIONS"])
@app.route("/2048/", methods=["POST", "OPTIONS"])
def play_2048():
    # Handle preflight cleanly (some clients send it)
    if request.method == "OPTIONS":
        return ("", 204)

    # Be tolerant about JSON (in case Content-Type is missing)
    data = request.get_json(silent=True)
    if data is None:
        try:
            data = json.loads(request.data.decode("utf-8") or "{}")
        except Exception:
            return jsonify({"error": "invalid JSON"}), 400

    grid = data.get("grid")
    direction = str(data.get("mergeDirection", "")).upper()

    if not (isinstance(grid, list) and len(grid) == 4 and all(isinstance(r, list) and len(r) == 4 for r in grid)):
        return jsonify({"error": "grid must be 4x4"}), 400
    if direction not in {"UP", "DOWN", "LEFT", "RIGHT"}:
        return jsonify({"error": "mergeDirection must be LEFT|RIGHT|UP|DOWN"}), 400

    # ensure ints or None
    grid = [[(int(v) if v is not None else None) for v in row] for row in grid]

    new_grid, moved, _ = move_grid(grid, direction)
    if moved:
        add_random_tile(new_grid)

    if has_2048(new_grid):
        end_status = "win"
    elif not can_move(new_grid):
        end_status = "lose"
    else:
        end_status = None

    # log to Render logs so you can see judge inputs
    print({"in": grid, "dir": direction, "out": new_grid, "end": end_status}, file=sys.stdout, flush=True)

    return jsonify({"nextGrid": new_grid, "endGame": end_status})

