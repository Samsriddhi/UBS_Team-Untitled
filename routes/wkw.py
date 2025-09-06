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

# ----- GFG Helper functions -----
def add_new(grid):
    r, c = random.choice([(i, j) for i in range(N) for j in range(N) if grid[i][j] is None])
    grid[r][c] = 2 if random.random() < 0.9 else 4
    return grid

def compress(grid):
    new_grid = [[None]*N for _ in range(N)]
    changed = False
    for i in range(N):
        pos = 0
        for j in range(N):
            if grid[i][j] is not None:
                new_grid[i][pos] = grid[i][j]
                if j != pos:
                    changed = True
                pos += 1
    return new_grid, changed

def merge(grid):
    changed = False
    for i in range(N):
        for j in range(N-1):
            if grid[i][j] is not None and grid[i][j] == grid[i][j+1]:
                grid[i][j] *= 2
                grid[i][j+1] = None
                changed = True
    return grid, changed

def reverse(grid):
    return [row[::-1] for row in grid]

def transpose(grid):
    return [[grid[j][i] for j in range(N)] for i in range(N)]

# ----- Move operations -----
def move_left(grid):
    grid, c1 = compress(grid)
    grid, c2 = merge(grid)
    grid, _ = compress(grid)
    return grid, c1 or c2

def move_right(grid):
    grid = reverse(grid)
    grid, changed = move_left(grid)
    grid = reverse(grid)
    return grid, changed

def move_up(grid):
    grid = transpose(grid)
    grid, changed = move_left(grid)
    grid = transpose(grid)
    return grid, changed

def move_down(grid):
    grid = transpose(grid)
    grid, changed = move_right(grid)
    grid = transpose(grid)
    return grid, changed

# ----- Game state -----
def get_end_status(grid):
    # win?
    if any(cell == 2048 for row in grid for cell in row if cell):
        return "win"
    # any empty?
    if any(cell is None for row in grid for cell in row):
        return None
    # any moves possible?
    for r in range(N):
        for c in range(N-1):
            if grid[r][c] == grid[r][c+1]:
                return None
    for r in range(N-1):
        for c in range(N):
            if grid[r][c] == grid[r+1][c]:
                return None
    return "lose"

# ----- Flask endpoint -----
@app.post("/2048")
def play_2048():
    data = request.get_json(force=True)
    grid = data.get("grid")
    direction = str(data.get("mergeDirection", "")).upper()

    if direction == "LEFT":
        new_grid, moved = move_left(grid)
    elif direction == "RIGHT":
        new_grid, moved = move_right(grid)
    elif direction == "UP":
        new_grid, moved = move_up(grid)
    elif direction == "DOWN":
        new_grid, moved = move_down(grid)
    else:
        return jsonify({"error": "bad direction"}), 400

    if moved:
        new_grid = add_new(new_grid)

    end_status = get_end_status(new_grid)

    return jsonify({
        "nextGrid": new_grid,
        "endGame": end_status
    })

