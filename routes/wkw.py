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

def rotate_counterclockwise(grid):
    return [[grid[r][N-1-c] for r in range(N)] for c in range(N-1, -1, -1)]

def slide_left_row(row):
    """Slide one row left, merging equal neighbors once."""
    nums = [x for x in row if x is not None]
    out, i, score_inc = [], 0, 0
    while i < len(nums):
        if i+1 < len(nums) and nums[i] == nums[i+1]:
            merged = nums[i]*2
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
    """Return new grid after move + whether moved + score increment."""
    g = [row[:] for row in grid]

    # Normalize: always move LEFT, rotate as needed
    if direction == "UP":
        g = rotate_counterclockwise(g)
    elif direction == "DOWN":
        g = rotate_clockwise(g)
        g = rotate_clockwise(g)
        g = rotate_clockwise(g)
    elif direction == "RIGHT":
        g = rotate_clockwise(g)
        g = rotate_clockwise(g)

    moved = False
    score_gain = 0
    new_g = []
    for row in g:
        new_row, inc = slide_left_row(row)
        if new_row != row:
            moved = True
        score_gain += inc
        new_g.append(new_row)

    # Rotate back to original orientation
    if direction == "UP":
        new_g = rotate_clockwise(new_g)
    elif direction == "DOWN":
        new_g = rotate_counterclockwise(new_g)
        new_g = rotate_counterclockwise(new_g)
        new_g = rotate_counterclockwise(new_g)
    elif direction == "RIGHT":
        new_g = rotate_clockwise(new_g)
        new_g = rotate_clockwise(new_g)

    return new_g, moved, score_gain

def empty_cells(grid):
    return [(r,c) for r in range(N) for c in range(N) if grid[r][c] is None]

def add_random_tile(grid):
    cells = empty_cells(grid)
    if not cells:
        return
    r,c = random.choice(cells)
    grid[r][c] = 2 if random.random() < 0.9 else 4

def has_2048(grid):
    return any(cell == 2048 for row in grid for cell in row if cell)

def can_move(grid):
    if empty_cells(grid):
        return True
    for r in range(N):
        for c in range(N):
            if r+1 < N and grid[r][c] == grid[r+1][c]:
                return True
            if c+1 < N and grid[r][c] == grid[r][c+1]:
                return True
    return False

@app.post("/2048")
def play_2048():
    data = request.get_json(force=True)
    grid = data.get("grid")
    direction = str(data.get("mergeDirection", "")).upper()

    if not isinstance(grid, list) or len(grid) != 4:
        return jsonify({"error": "grid must be 4x4"}), 400
    if direction not in {"UP","DOWN","LEFT","RIGHT"}:
        return jsonify({"error": "bad direction"}), 400

    new_grid, moved, _ = move_grid(grid, direction)
    if moved:
        add_random_tile(new_grid)

    # End game detection
    if has_2048(new_grid):
        end_status = "win"
    elif not can_move(new_grid):
        end_status = "lose"
    else:
        end_status = None

    return jsonify({
        "nextGrid": new_grid,
        "endGame": end_status
    })
