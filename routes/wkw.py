"""
Enhanced 2048 Game Server Implementation
Handles the game logic for the 2048 puzzle game with advanced features.
"""

import json
import random
import logging
from typing import List, Optional, Tuple, Dict, Any, Union
from flask import request, jsonify

from routes import app

logger = logging.getLogger(__name__)


class Game2048:
    """Enhanced 2048 game logic implementation"""

    def __init__(self, grid_size: int = 4):
        self.grid_size = grid_size

    def is_valid_grid(self, grid: List[List]) -> bool:
        """Validate that the grid is NxN with valid values"""
        if not isinstance(grid, list) or len(grid) != self.grid_size:
            return False

        for row in grid:
            if not isinstance(row, list) or len(row) != self.grid_size:
                return False
            for cell in row:
                if cell is not None:
                    if cell == '0' or cell == '*2' or cell == '1':
                        continue
                    elif isinstance(cell, int) and cell >= 2 and (cell & (cell - 1)) == 0:
                        continue
                    else:
                        return False
        return True

    def add_random_tile(self, grid: List[List[Optional[Union[int, str]]]]) -> List[List[Optional[Union[int, str]]]]:
        """Add a random tile (2 or 4) to an empty cell"""
        empty_cells = []
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if grid[r][c] is None:
                    empty_cells.append((r, c))

        if not empty_cells:
            return grid

        r, c = random.choice(empty_cells)
        grid[r][c] = 2 if random.random() < 0.9 else 4
        return grid

    def process_row_left(self, row: List[Optional[Union[int, str]]]) -> List[Optional[Union[int, str]]]:
        """Process a single row moving left with special tile logic"""
        result = [None] * self.grid_size
        pos = 0
        i = 0
        
        while i < len(row):
            current = row[i]
            
            if current is None:
                i += 1
                continue
            
            if current == '0':
                # '0' tiles don't move and block movement
                result[i] = '0'
                pos = i + 1
                i += 1
                continue
            
            if current == '*2':
                # '*2' tries to merge with the next non-null tile
                next_pos = i + 1
                while next_pos < len(row) and row[next_pos] is None:
                    next_pos += 1
                
                if next_pos < len(row) and row[next_pos] not in ['0', '*2']:
                    # Merge with the next tile
                    next_tile = row[next_pos]
                    if next_tile == '1':
                        # '*2' converts '1' to 2
                        result[pos] = 2
                    elif isinstance(next_tile, int):
                        # '*2' doubles the number
                        result[pos] = next_tile * 2
                    pos += 1
                    i = next_pos + 1
                else:
                    # No tile to merge with, '*2' stays
                    result[pos] = '*2'
                    pos += 1
                    i += 1
                continue
            
            if current == '1':
                # '1' tiles move but don't merge with other '1's
                result[pos] = '1'
                pos += 1
                i += 1
                continue
            
            if isinstance(current, int):
                # Regular number tiles
                # Check if next tile is the same number for merging
                next_pos = i + 1
                while next_pos < len(row) and row[next_pos] is None:
                    next_pos += 1
                
                if (next_pos < len(row) and 
                    isinstance(row[next_pos], int) and 
                    row[next_pos] == current):
                    # Merge two identical numbers
                    result[pos] = current * 2
                    pos += 1
                    i = next_pos + 1
                else:
                    # Just move the tile
                    result[pos] = current
                    pos += 1
                    i += 1
                continue
            
            # Fallback: just move the tile
            result[pos] = current
            pos += 1
            i += 1
        
        return result

    def move_left(self, grid: List[List[Optional[Union[int, str]]]]) -> Tuple[List[List[Optional[Union[int, str]]]], bool]:
        """Move and merge tiles to the left, return (new_grid, changed)"""
        new_grid = []
        changed = False

        for r in range(self.grid_size):
            new_row = self.process_row_left(grid[r])
            new_grid.append(new_row)
            
            # Check if this row changed
            if new_row != grid[r]:
                changed = True

        return new_grid, changed

    def move_right(self, grid: List[List[Optional[Union[int, str]]]]) -> Tuple[List[List[Optional[Union[int, str]]]], bool]:
        """Move and merge tiles to the right"""
        reversed_grid = [row[::-1] for row in grid]
        moved_grid, changed = self.move_left(reversed_grid)
        result_grid = [row[::-1] for row in moved_grid]
        return result_grid, changed

    def move_up(self, grid: List[List[Optional[Union[int, str]]]]) -> Tuple[List[List[Optional[Union[int, str]]]], bool]:
        """Move and merge tiles upward"""
        transposed = [[grid[r][c] for r in range(self.grid_size)] for c in range(self.grid_size)]
        moved_transposed, changed = self.move_left(transposed)
        result_grid = [[moved_transposed[c][r] for c in range(self.grid_size)] for r in range(self.grid_size)]
        return result_grid, changed

    def move_down(self, grid: List[List[Optional[Union[int, str]]]]) -> Tuple[List[List[Optional[Union[int, str]]]], bool]:
        """Move and merge tiles downward"""
        transposed = [[grid[r][c] for r in range(self.grid_size)] for c in range(self.grid_size)]
        moved_transposed, changed = self.move_right(transposed)
        result_grid = [[moved_transposed[c][r] for c in range(self.grid_size)] for r in range(self.grid_size)]
        return result_grid, changed

    def make_move(self, grid: List[List[Optional[Union[int, str]]]], direction: str) -> Tuple[List[List[Optional[Union[int, str]]]], bool]:
        """Make a move in the specified direction"""
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

    def check_win(self, grid: List[List[Optional[Union[int, str]]]]) -> bool:
        """Check if the player has won (reached 2048)"""
        for row in grid:
            for cell in row:
                if isinstance(cell, int) and cell >= 2048:
                    return True
        return False

    def can_move(self, grid: List[List[Optional[Union[int, str]]]]) -> bool:
        """Check if any moves are possible"""
        # Check for empty cells
        for row in grid:
            if None in row:
                return True

        # Check for possible horizontal merges
        for r in range(self.grid_size):
            for c in range(self.grid_size - 1):
                current = grid[r][c]
                next_cell = grid[r][c + 1]
                
                # Regular number merge
                if (isinstance(current, int) and isinstance(next_cell, int) and 
                    current == next_cell):
                    return True
                
                # '*2' can merge with numbers or '1'
                if current == '*2' and next_cell not in [None, '0', '*2']:
                    return True
                
                # Check if tiles can move past '0' tiles
                if current is not None and current != '0' and next_cell is None:
                    return True

        # Check for possible vertical merges
        for r in range(self.grid_size - 1):
            for c in range(self.grid_size):
                current = grid[r][c]
                next_cell = grid[r + 1][c]
                
                # Regular number merge
                if (isinstance(current, int) and isinstance(next_cell, int) and 
                    current == next_cell):
                    return True
                
                # '*2' can merge with numbers or '1'
                if current == '*2' and next_cell not in [None, '0', '*2']:
                    return True
                
                # Check if tiles can move past '0' tiles
                if current is not None and current != '0' and next_cell is None:
                    return True

        return False

    def process_move(self, grid: List[List[Optional[Union[int, str]]]], direction: str) -> Dict[str, Any]:
        """Process a move and return the result"""
        try:
            if not self.is_valid_grid(grid):
                return {
                    "nextGrid": grid,
                    "endGame": None,
                    "error": "Invalid grid format"
                }

            new_grid, changed = self.make_move(grid, direction)

            if not changed:
                return {
                    "nextGrid": grid,
                    "endGame": None
                }

            new_grid = self.add_random_tile(new_grid)

            if self.check_win(new_grid):
                return {
                    "nextGrid": new_grid,
                    "endGame": "win"
                }

            if not self.can_move(new_grid):
                return {
                    "nextGrid": new_grid,
                    "endGame": "lose"
                }

            return {
                "nextGrid": new_grid,
                "endGame": None
            }

        except Exception as e:
            logger.error(f"Error processing move: {e}")
            return {
                "nextGrid": grid,
                "endGame": None,
                "error": str(e)
            }


# Global game instance (default 4x4, can pass grid_size=5 for bigger grid)
game_2048 = Game2048()


@app.route('/2048', methods=['POST', 'OPTIONS'])
def game_2048_endpoint():
    """
    Enhanced 2048 game endpoint
    Expected input: {"grid": NxN array, "mergeDirection": "UP"/"DOWN"/"LEFT"/"RIGHT"}
    Returns: {"nextGrid": NxN array, "endGame": null/"win"/"lose"}
    
    Special tiles supported:
    - '0': Immovable tiles that block movement
    - '*2': Multiplier tiles that merge with adjacent numbers/1s
    - '1': Movable tiles that don't merge with other 1s but can be converted by *2
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

        # Dynamically create game instance with correct grid size
        grid_size = len(grid) if grid else 4
        game_instance = Game2048(grid_size)

        logger.info(f"2048 move: direction={merge_direction}, grid_size={grid_size}")
        logger.debug(f"Current grid: {grid}")

        result = game_instance.process_move(grid, merge_direction)

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
