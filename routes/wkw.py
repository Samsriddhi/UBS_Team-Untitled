"""
2048 Game Server Implementation
Handles the game logic for the 2048 puzzle game.
"""

import json
import random
import logging
from typing import List, Dict, Any, Tuple
from flask import request, jsonify

from routes import app

# --- Logging ---
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Game2048:
    """2048 game logic implementation"""

    def __init__(self):
        pass  # No fixed grid_size constraint

    # ---------------- Validation ----------------
    def is_valid_grid(self, grid: List[List]) -> bool:
        """Validate that the grid is NxN with valid values"""
        if not isinstance(grid, list) or not grid:
            return False

        grid_size = len(grid)
        for row in grid:
            if not isinstance(row, list) or len(row) != grid_size:
                return False
            for cell in row:
                if cell is not None and not self.is_valid_cell_value(cell):
                    return False
        return True

    def is_valid_cell_value(self, cell) -> bool:
        """Check if a cell value is valid"""
        if cell is None:
            return True
        if cell in (0, "0", "*2", 1, "1"):
            return True
        return isinstance(cell, int) and cell >= 2 and (cell & (cell - 1)) == 0

    # ---------------- Random Tile ----------------
    def add_random_tile(self, grid: List[List]) -> List[List]:
        """Add a random tile (2 or 4) to an empty cell"""
        grid_size = len(grid)
        empty_cells = [(r, c) for r in range(grid_size) for c in range(grid_size) if grid[r][c] is None]

        if not empty_cells:
            return grid

        r, c = random.choice(empty_cells)
        grid[r][c] = 2 if random.random() < 0.9 else 4
        return grid

    # ---------------- Moves ----------------
    def move_left(self, grid: List[List]) -> Tuple[List[List], bool]:
        """Move and merge tiles to the left"""
        grid_size = len(grid)
        new_grid = [[None] * grid_size for _ in range(grid_size)]
        changed = False

        for r in range(grid_size):
            new_row = self.process_row_left(grid[r])
            for c in range(grid_size):
                new_grid[r][c] = new_row[c]
                if new_grid[r][c] != grid[r][c]:
                    changed = True

        return new_grid, changed

    def process_row_left(self, row: List) -> List:
        """Process a row with special tile rules"""
        result = [None] * len(row)
        i = 0

        while i < len(row):
            if row[i] in (0, "0"):
                result[i] = row[i]
                left_part = row[:i]
                if left_part:
                    processed_left = self.process_segment_left(left_part)
                    result[:len(processed_left)] = processed_left
                right_part = row[i + 1:]
                if right_part:
                    processed_right = self.process_row_left(right_part)
                    result[i + 1:i + 1 + len(processed_right)] = processed_right
                return result
            i += 1

        processed = self.process_segment_left(row)
        result[:len(processed)] = processed
        return result

    def process_segment_left(self, segment: List) -> List:
        """Process a segment without '0' tiles"""
        non_none = [cell for cell in segment if cell is not None]
        if not non_none:
            return []

        # Handle special tile interactions
        processed, skip_next, converted_positions = [], False, set()
        for i, cell in enumerate(non_none):
            if skip_next:
                skip_next = False
                continue
            if (cell in (1, "1") and i + 1 < len(non_none) and non_none[i + 1] == "*2"):
                processed.append(2)
                converted_positions.add(len(processed) - 1)
                skip_next = True
            else:
                processed.append(cell)

        result = []
        for i, cell in enumerate(processed):
            if cell == "*2":
                if result and isinstance(result[-1], int) and (len(result) - 1) not in converted_positions:
                    result[-1] *= 2
                result.append("*2")
            elif result and result[-1] == cell and isinstance(cell, int) and cell >= 2:
                result[-1] *= 2
            else:
                result.append(cell)

        return self.final_times2_compression(result)

    def final_times2_compression(self, tiles: List) -> List:
        """Handle '*2' compression rules"""
        if len(tiles) >= 2 and isinstance(tiles[-1], int) and tiles[-2] == "*2":
            times2_count = 0
            for i in range(len(tiles) - 2, -1, -1):
                if tiles[i] == "*2":
                    times2_count += 1
                else:
                    break
            if times2_count > 1:
                tiles = tiles[:]
                tiles.pop(len(tiles) - 2)
        return tiles

    def move_right(self, grid: List[List]) -> Tuple[List[List], bool]:
        reversed_grid = [row[::-1] for row in grid]
        moved_grid, changed = self.move_left(reversed_grid)
        return [row[::-1] for row in moved_grid], changed

    def move_up(self, grid: List[List]) -> Tuple[List[List], bool]:
        transposed = list(map(list, zip(*grid)))
        moved, changed = self.move_left(transposed)
        return [list(row) for row in zip(*moved)], changed

    def move_down(self, grid: List[List]) -> Tuple[List[List], bool]:
        transposed = list(map(list, zip(*grid)))
        moved, changed = self.move_right(transposed)
        return [list(row) for row in zip(*moved)], changed

    def make_move(self, grid: List[List], direction: str) -> Tuple[List[List], bool]:
        direction = direction.upper()
        if direction == "LEFT":
            return self.move_left(grid)
        if direction == "RIGHT":
            return self.move_right(grid)
        if direction == "UP":
            return self.move_up(grid)
        if direction == "DOWN":
            return self.move_down(grid)
        raise ValueError(f"Invalid direction: {direction}")

    # ---------------- Game State ----------------
    def check_win(self, grid: List[List]) -> bool:
        return any(isinstance(cell, int) and cell >= 2048 for row in grid for cell in row)

    def can_merge(self, cell1, cell2) -> bool:
        if cell1 is None or cell2 is None:
            return False
        if isinstance(cell1, int) and isinstance(cell2, int) and cell1 == cell2 and cell1 >= 2:
            return True
        if (cell1 == "*2" and isinstance(cell2, int) and cell2 >= 1) or \
           (cell2 == "*2" and isinstance(cell1, int) and cell1 >= 1):
            return True
        return False

    def can_move(self, grid: List[List]) -> bool:
        grid_size = len(grid)
        if any(None in row for row in grid):
            return True
        for r in range(grid_size):
            for c in range(grid_size - 1):
                if self.can_merge(grid[r][c], grid[r][c + 1]):
                    return True
        for r in range(grid_size - 1):
            for c in range(grid_size):
                if self.can_merge(grid[r][c], grid[r + 1][c]):
                    return True
        return False

    def process_move(self, grid: List[List], direction: str) -> Dict[str, Any]:
        """Process a move and return the result"""
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


# ---------------- Flask Endpoints ----------------
game_2048 = Game2048()


@app.route("/2048", methods=["POST", "OPTIONS"])
def game_2048_endpoint():
    """Main 2048 game endpoint"""
    if request.method == "OPTIONS":
        response = jsonify()
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Methods", "POST, OPTIONS")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type")
        return response

    try:
        data = request.get_json(force=True)
        grid = data.get("grid")
        merge_direction = data.get("mergeDirection")

        if not grid:
            return jsonify({"error": "Missing 'grid' field"}), 400
        if not merge_direction:
            return jsonify({"error": "Missing 'mergeDirection' field"}), 400

        result = game_2048.process_move(grid, merge_direction)
        response = jsonify(result)
        response.headers.add("Access-Control-Allow-Origin", "*")
        return response
    except Exception as e:
        logger.error(f"Error in 2048 endpoint: {e}")
        response = jsonify({"error": "Internal server error"})
        response.headers.add("Access-Control-Allow-Origin", "*")
        return response, 500


@app.route("/2048/test", methods=["GET"])
def test_2048():
    """Test endpoint for 2048 game logic"""
    test_grid = [[2, 2, None, None], [4, 4, None, None], [None] * 4, [None] * 4]
    result = game_2048.process_move(test_grid, "LEFT")
    return jsonify({"test_input": {"grid": test_grid, "direction": "LEFT"}, "result": result})
