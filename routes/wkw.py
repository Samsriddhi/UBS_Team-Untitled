"""
2048 Game Server Implementation
Handles the game logic for the 2048 puzzle game.
"""

import json
import random
import logging
from typing import List, Optional, Tuple, Dict, Any
from flask import request, jsonify

from routes import app

logger = logging.getLogger(__name__)


class Game2048:
    """2048 game logic implementation"""

    def __init__(self):
        # 20% chance to use bigger grid (5x5), otherwise classic 4x4
        self.grid_size = 5 if random.random() < 0.2 else 4

    # ---------------- Validation ----------------
    def is_valid_cell(self, cell) -> bool:
        """Check if a cell value is valid"""
        if cell is None:
            return True
        if cell == 0:  # special immovable tile
            return True
        return isinstance(cell, int) and cell >= 2 and (cell & (cell - 1)) == 0

    def is_valid_grid(self, grid: List[List]) -> bool:
        """Validate that the grid is N×N with valid values"""
        if not isinstance(grid, list) or len(grid) != self.grid_size:
            return False
        for row in grid:
            if not isinstance(row, list) or len(row) != self.grid_size:
                return False
            for cell in row:
                if not self.is_valid_cell(cell):
                    return False
        return True

    # ---------------- Tile Spawning ----------------
    def add_random_tile(self, grid: List[List[Optional[int]]]) -> List[List[Optional[int]]]:
        """Add a random tile (2, 4, or sometimes '0') to an empty cell"""
        empty_cells = [(r, c) for r in range(self.grid_size) for c in range(self.grid_size) if grid[r][c] is None]

        if not empty_cells:
            return grid

        r, c = random.choice(empty_cells)

        roll = random.random()
        if roll < 0.2:
            grid[r][c] = 0   # 20% chance of spawning a blocking '0'
        elif roll < 0.9:
            grid[r][c] = 2
        else:
            grid[r][c] = 4

        return grid

    # ---------------- Movement ----------------
    def process_row_left(self, row: List[Optional[int]]) -> List[Optional[int]]:
        """Process a single row moving left, respecting '0' barriers"""
        size = self.grid_size
        result = [None] * size

        i = 0
        pos = 0
        while i < size:
            if row[i] == 0:
                # keep '0' in place, reset position pointer
                result[i] = 0
                pos = i + 1
                i += 1
            elif row[i] is None:
                i += 1
            else:
                # collect a run until next barrier
                run = []
                j = i
                while j < size and row[j] not in (0, None):
                    run.append(row[j])
                    j += 1

                # merge in this run
                merged = []
                k = 0
                while k < len(run):
                    if k + 1 < len(run) and run[k] == run[k + 1]:
                        merged.append(run[k] * 2)
                        k += 2
                    else:
                        merged.append(run[k])
                        k += 1

                for val in merged:
                    if pos < size:
                        result[pos] = val
                        pos += 1

                i = j

        return result

    def move_left(self, grid: List[List[Optional[int]]]) -> Tuple[List[List[Optional[int]]], bool]:
        size = self.grid_size
        new_grid = []
        changed = False

        for r in range(size):
            new_row = self.process_row_left(grid[r])
            if new_row != grid[r]:
                changed = True
            new_grid.append(new_row)

        return new_grid, changed

    def move_right(self, grid: List[List[Optional[int]]]) -> Tuple[List[List[Optional[int]]], bool]:
        reversed_grid = [row[::-1] for row in grid]
        moved_grid, changed = self.move_left(reversed_grid)
        result_grid = [row[::-1] for row in moved_grid]
        return result_grid, changed

    def move_up(self, grid: List[List[Optional[int]]]) -> Tuple[List[List[Optional[int]]], bool]:
        size = self.grid_size
        transposed = [[grid[r][c] for r in range(size)] for c in range(size)]
        moved_transposed, changed = self.move_left(transposed)
        result_grid = [[moved_transposed[c][r] for c in range(size)] for r in range(size)]
        return result_grid, changed

    def move_down(self, grid: List[List[Optional[int]]]) -> Tuple[List[List[Optional[int]]], bool]:
        size = self.grid_size
        transposed = [[grid[r][c] for r in range(size)] for c in range(size)]
        moved_transposed, changed = self.move_right(transposed)
        result_grid = [[moved_transposed[c][r] for c in range(size)] for r in range(size)]
        return result_grid, changed

    def make_move(self, grid: List[List[Optional[int]]], direction: str) -> Tuple[List[List[Optional[int]]], bool]:
        direction = direction.upper()
        if direction == "LEFT":
            return self.move_left(grid)
        elif direction == "RIGHT":
            return self.move_right(grid)
        elif direction == "UP":
            return self.move_up(grid)
        elif direction == "DOWN":
            return self.move_down(grid)
        else:
            raise ValueError(f"Invalid direction: {direction}")

    # ---------------- Game State ----------------
    def check_win(self, grid: List[List[Optional[int]]]) -> bool:
        return any(cell is not None and cell >= 2048 for row in grid for cell in row if isinstance(cell, int))

    def can_move(self, grid: List[List[Optional[int]]]) -> bool:
        size = self.grid_size

        # empty cells
        if any(None in row for row in grid):
            return True

        # horizontal
        for r in range(size):
            for c in range(size - 1):
                if grid[r][c] == grid[r][c + 1] and grid[r][c] != 0:
                    return True

        # vertical
        for r in range(size - 1):
            for c in range(size):
                if grid[r][c] == grid[r + 1][c] and grid[r][c] != 0:
                    return True

        return False

    def process_move(self, grid: List[List[Optional[int]]], direction: str) -> Dict[str, Any]:
        try:
            if not self.is_valid_grid(grid):
                return {"nextGrid": grid, "endGame": None, "error": "Invalid grid format"}

            new_grid, changed = self.make_move(grid, direction)
            if not changed:
                return {"nextGrid": grid, "endGame": None}

            new_grid = self.add_random_tile(new_grid)

            if self.check_win(new_grid):
                return {"nextGrid": new_grid, "endGame": "win"}
            if not self.can_move(new_grid):
                return {"nextGrid": new_grid, "endGame": "lose"}

            return {"nextGrid": new_grid, "endGame": None}

        except Exception as e:
            logger.error(f"Error processing move: {e}")
            return {"nextGrid": grid, "endGame": None, "error": str(e)}


# Global game instance
game_2048 = Game2048()


@app.route('/2048', methods=['POST', 'OPTIONS'])
def game_2048_endpoint():
    """
    2048 game endpoint
    Supports 4x4 (80%) and 5x5 (20%) boards.
    Includes special '0' immovable barrier tiles.
    """
    if request.method == 'OPTIONS':
        response = jsonify()
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        return response

    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        grid = data.get('grid')
        merge_direction = data.get('mergeDirection')

        if not grid:
            return jsonify({"error": "Missing 'grid' field"}), 400
        if not merge_direction:
            return jsonify({"error": "Missing 'mergeDirection' field"}), 400

        logger.info(f"2048 move: direction={merge_direction}")
        logger.debug(f"Current grid: {grid}")

        result = game_2048.process_move(grid, merge_direction)

        logger.debug(f"Result grid: {result.get('nextGrid')}")
        logger.info(f"End game status: {result.get('endGame')}")

        response = jsonify(result)
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response

    except Exception as e:
        logger.error(f"Error in 2048 endpoint: {e}")
        response = jsonify({"error": "Internal server error"})
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response, 500





# """
# 2048 Game Server Implementation
# Handles the game logic for the 2048 puzzle game.
# """

# import json
# import random
# import logging
# from typing import List, Optional, Tuple, Dict, Any
# from flask import request, jsonify

# from routes import app

# logger = logging.getLogger(__name__)


# class Game2048:
#     """2048 game logic implementation"""

#     def __init__(self):
#         self.grid_size = 4

#     def is_valid_grid(self, grid: List[List]) -> bool:
#         """Validate that the grid is 4x4 with valid values"""
#         if not isinstance(grid, list) or len(grid) != 4:
#             return False

#         for row in grid:
#             if not isinstance(row, list) or len(row) != 4:
#                 return False
#             for cell in row:
#                 if cell is not None and (not isinstance(cell, int) or cell < 2 or (cell & (cell - 1)) != 0):
#                     return False
#         return True

#     def add_random_tile(self, grid: List[List[Optional[int]]]) -> List[List[Optional[int]]]:
#         """Add a random tile (2 or 4) to an empty cell"""
#         empty_cells = []
#         for r in range(4):
#             for c in range(4):
#                 if grid[r][c] is None:
#                     empty_cells.append((r, c))

#         if not empty_cells:
#             return grid

#         r, c = random.choice(empty_cells)
#         grid[r][c] = 2 if random.random() < 0.9 else 4
#         return grid

#     def move_left(self, grid: List[List[Optional[int]]]) -> Tuple[List[List[Optional[int]]], bool]:
#         """Move and merge tiles to the left, return (new_grid, changed)"""
#         new_grid = [[None for _ in range(4)] for _ in range(4)]
#         changed = False

#         for r in range(4):
#             values = [cell for cell in grid[r] if cell is not None]
#             merged = []
#             i = 0
#             while i < len(values):
#                 if i + 1 < len(values) and values[i] == values[i + 1]:
#                     merged.append(values[i] * 2)
#                     i += 2
#                 else:
#                     merged.append(values[i])
#                     i += 1

#             for c in range(4):
#                 if c < len(merged):
#                     new_grid[r][c] = merged[c]
#                     if new_grid[r][c] != grid[r][c]:
#                         changed = True
#                 else:
#                     new_grid[r][c] = None
#                     if grid[r][c] is not None:
#                         changed = True

#         return new_grid, changed

#     def move_right(self, grid: List[List[Optional[int]]]) -> Tuple[List[List[Optional[int]]], bool]:
#         """Move and merge tiles to the right"""
#         reversed_grid = [row[::-1] for row in grid]
#         moved_grid, changed = self.move_left(reversed_grid)
#         result_grid = [row[::-1] for row in moved_grid]
#         return result_grid, changed

#     def move_up(self, grid: List[List[Optional[int]]]) -> Tuple[List[List[Optional[int]]], bool]:
#         """Move and merge tiles upward"""
#         transposed = [[grid[r][c] for r in range(4)] for c in range(4)]
#         moved_transposed, changed = self.move_left(transposed)
#         result_grid = [[moved_transposed[c][r] for c in range(4)] for r in range(4)]
#         return result_grid, changed

#     def move_down(self, grid: List[List[Optional[int]]]) -> Tuple[List[List[Optional[int]]], bool]:
#         """Move and merge tiles downward"""
#         transposed = [[grid[r][c] for r in range(4)] for c in range(4)]
#         moved_transposed, changed = self.move_right(transposed)
#         result_grid = [[moved_transposed[c][r] for c in range(4)] for r in range(4)]
#         return result_grid, changed

#     def make_move(self, grid: List[List[Optional[int]]], direction: str) -> Tuple[List[List[Optional[int]]], bool]:
#         """Make a move in the specified direction"""
#         direction = direction.upper()

#         if direction == "LEFT":
#             return self.move_left(grid)
#         elif direction == "RIGHT":
#             return self.move_right(grid)
#         elif direction == "UP":
#             return self.move_up(grid)
#         elif direction == "DOWN":
#             return self.move_down(grid)
#         else:
#             raise ValueError(f"Invalid direction: {direction}")

#     def check_win(self, grid: List[List[Optional[int]]]) -> bool:
#         """Check if the player has won (reached 2048)"""
#         for row in grid:
#             for cell in row:
#                 if cell and cell >= 2048:
#                     return True
#         return False

#     def can_move(self, grid: List[List[Optional[int]]]) -> bool:
#         """Check if any moves are possible"""
#         for row in grid:
#             if None in row:
#                 return True

#         for r in range(4):
#             for c in range(3):
#                 if grid[r][c] == grid[r][c + 1]:
#                     return True

#         for r in range(3):
#             for c in range(4):
#                 if grid[r][c] == grid[r + 1][c]:
#                     return True

#         return False

#     def process_move(self, grid: List[List[Optional[int]]], direction: str) -> Dict[str, Any]:
#         """Process a move and return the result"""
#         try:
#             if not self.is_valid_grid(grid):
#                 return {
#                     "nextGrid": grid,
#                     "endGame": None,
#                     "error": "Invalid grid format"
#                 }

#             new_grid, changed = self.make_move(grid, direction)

#             if not changed:
#                 return {
#                     "nextGrid": grid,
#                     "endGame": None
#                 }

#             new_grid = self.add_random_tile(new_grid)

#             if self.check_win(new_grid):
#                 return {
#                     "nextGrid": new_grid,
#                     "endGame": "win"
#                 }

#             if not self.can_move(new_grid):
#                 return {
#                     "nextGrid": new_grid,
#                     "endGame": "lose"
#                 }

#             return {
#                 "nextGrid": new_grid,
#                 "endGame": None
#             }

#         except Exception as e:
#             logger.error(f"Error processing move: {e}")
#             return {
#                 "nextGrid": grid,
#                 "endGame": None,
#                 "error": str(e)
#             }


# # Global game instance
# game_2048 = Game2048()


# @app.route('/2048', methods=['POST', 'OPTIONS'])
# def game_2048_endpoint():
#     """
#     2048 game endpoint
#     Expected input: {"grid": 4x4 array, "mergeDirection": "UP"/"DOWN"/"LEFT"/"RIGHT"}
#     Returns: {"nextGrid": 4x4 array, "endGame": null/"win"/"lose"}
#     """
#     if request.method == 'OPTIONS':
#         response = jsonify()
#         response.headers.add('Access-Control-Allow-Origin', '*')
#         response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
#         response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
#         return response

#     try:
#         data = request.get_json(force=True)
#         if not data:
#             return jsonify({"error": "No JSON data provided"}), 400

#         grid = data.get('grid')
#         merge_direction = data.get('mergeDirection')

#         if not grid:
#             return jsonify({"error": "Missing 'grid' field"}), 400
#         if not merge_direction:
#             return jsonify({"error": "Missing 'mergeDirection' field"}), 400

#         logger.info(f"2048 move: direction={merge_direction}")
#         logger.debug(f"Current grid: {grid}")

#         result = game_2048.process_move(grid, merge_direction)

#         logger.debug(f"Result grid: {result.get('nextGrid')}")
#         logger.info(f"End game status: {result.get('endGame')}")

#         response = jsonify(result)
#         response.headers.add('Access-Control-Allow-Origin', '*')
#         return response

#     except Exception as e:
#         logger.error(f"Error in 2048 endpoint: {e}")
#         response = jsonify({"error": "Internal server error"})
#         response.headers.add('Access-Control-Allow-Origin', '*')
#         return response, 500
