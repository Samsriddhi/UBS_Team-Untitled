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

    # if you already have this, don't duplicate
CORS(app)                     # allow the UI origin

N = 4

def rotate_clockwise(grid):
    return [[grid[N-1-r][c] for r in range(N)] for c in range(N)]

def slide_right_row(row):
    # row: list of length 4 with None or numbers
    nums = [v for v in row if v is not None]
    out = [None]*N
    write = N-1
    moved = False
    for i in range(len(nums)-1, -1, -1):
        cur = nums[i]
        if i-1 >= 0 and nums[i-1] == cur:
            cur = cur*2
            i -= 1
        if out[write] != cur:
            moved = True
        out[write] = cur
        write -= 1
    if out != row:
        moved = True
    return out, moved

def empty_cells(g):
    cells = []
    for r in range(N):
        for c in range(N):
            if g[r][c] is None:
                cells.append((r,c))
    return cells

def add_random_tile(g):
    cells = empty_cells(g)
    if not cells:
        return
    r,c = random.choice(cells)
    g[r][c] = 2 if random.random() < 0.9 else 4

def apply_move(grid, direction):
    turns = {"RIGHT":0, "DOWN":1, "LEFT":2, "UP":3}.get(direction.upper(), 0)
    g = [row[:] for row in grid]
    for _ in range(turns):
        g = rotate_clockwise(g)

    moved_any = False
    after = []
    for row in g:
        new_row, moved = slide_right_row(row)
        moved_any = moved_any or moved
        after.append(new_row)

    # rotate back
    for _ in range((4 - turns) % 4):
        after = rotate_clockwise(after)

    return after, moved_any

# ---- EXACT endpoint the UI calls ----
@app.post("/2048")
def play_2048():
    data = request.get_json(force=True)
    grid = data.get("grid")
    merge_dir = str(data.get("mergeDirection", "")).upper()

    if not isinstance(grid, list) or len(grid) != 4 or any(len(r)!=4 for r in grid):
        return jsonify({"error": "grid must be 4x4"}), 400
    if merge_dir not in {"LEFT","RIGHT","UP","DOWN"}:
        return jsonify({"error": "mergeDirection must be LEFT|RIGHT|UP|DOWN"}), 400

    new_grid, moved = apply_move(grid, merge_dir)
    if moved:
        add_random_tile(new_grid)

    return jsonify({"grid": new_grid})
