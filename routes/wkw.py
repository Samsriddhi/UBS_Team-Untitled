"""
2048 Game Server Implementation
Handles the game logic for the 2048 puzzle game with advanced features.
"""

import json
import random
import logging
from typing import List, Optional, Tuple, Dict, Any
from flask import request, jsonify

from routes import app

logger = logging.getLogger(__name__)

class Game2048:
    """2048 game logic implementation with advanced features"""
    
    def __init__(self):
        pass
        
    def is_valid_grid(self, grid: List[List]) -> bool:
        """Validate that the grid is NxN with valid values"""
        if not isinstance(grid, list) or len(grid) == 0:
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
        
        # Handle special tiles
        if cell == 0 or cell == '0':
            return True
        if cell == '*2' or cell == '×2':
            return True
        if cell == 1 or cell == '1':
            return True
            
        # Handle regular number tiles (must be power of 2 and >= 2)
        if isinstance(cell, int) and cell >= 2 and (cell & (cell - 1)) == 0:
            return True
            
        return False
    
    def add_random_tile(self, grid: List[List]) -> List[List]:
        """Add a random tile to an empty cell"""
        grid_size = len(grid)
        
        # Find empty cells
        empty_cells = []
        for r in range(grid_size):
            for c in range(grid_size):
                if grid[r][c] is None:
                    empty_cells.append((r, c))
        
        if not empty_cells:
            return grid  # No empty cells
        
        # Choose random empty cell
        r, c = random.choice(empty_cells)
        
        # 90% chance for 2, 10% chance for 4
        grid[r][c] = 2 if random.random() < 0.9 else 4
        
        return grid
    
    def process_row_left(self, row: List) -> List:
        """Process a single row moving left with all special tile rules"""
        # First, handle '0' tiles which act as barriers
        segments = []
        current_segment = []
        
        for cell in row:
            if cell == 0 or cell == '0':
                if current_segment:
                    segments.append(current_segment)
                    current_segment = []
                segments.append([cell])  # '0' as a separate segment
            else:
                current_segment.append(cell)
        
        if current_segment:
            segments.append(current_segment)
        
        # Process each segment
        processed_segments = []
        for segment in segments:
            if segment == [0] or segment == ['0']:
                processed_segments.append([0])  # Keep '0' as is
            else:
                processed_segments.append(self.process_segment_left(segment))
        
        # Reconstruct the row
        result = []
        for segment in processed_segments:
            result.extend(segment)
        
        # Pad with None to maintain row length
        while len(result) < len(row):
            result.append(None)
            
        return result
    
    def process_segment_left(self, segment: List) -> List:
        """Process a segment without '0' tiles"""
        if not segment:
            return []
            
        # Remove None values first
        non_none = [cell for cell in segment if cell is not None]
        if not non_none:
            return [None] * len(segment)
        
        # First pass: handle '1' + '*2' conversions
        processed = []
        i = 0
        while i < len(non_none):
            current = non_none[i]
            
            # Check if current is '1' and next is '*2'
            if (current == 1 or current == '1') and i + 1 < len(non_none) and non_none[i + 1] == '*2':
                processed.append(2)  # Convert '1' to 2
                i += 2  # Skip the '*2'
            else:
                processed.append(current)
                i += 1
        
        # Second pass: handle regular merging and '*2' multiplication
        result = []
        i = 0
        while i < len(processed):
            current = processed[i]
            
            if current == '*2':
                # If there's a number before this '*2', multiply it
                if result and isinstance(result[-1], int) and result[-1] >= 2:
                    result[-1] *= 2
                    # The '*2' tile is consumed in the process
                else:
                    # No number to multiply, keep the '*2'
                    result.append('*2')
                i += 1
            else:
                # Regular number
                if i + 1 < len(processed) and processed[i + 1] == '*2':
                    # Current number followed by '*2' - multiply current number
                    if isinstance(current, int):
                        result.append(current * 2)
                    else:
                        result.append(current)
                    i += 2  # Skip the '*2'
                elif (result and isinstance(result[-1], int) and 
                      isinstance(current, int) and result[-1] == current):
                    # Merge with previous same number
                    result[-1] = current * 2
                    i += 1
                else:
                    # No merge possible
                    result.append(current)
                    i += 1
        
        # Pad with None to maintain segment length
        while len(result) < len(segment):
            result.append(None)
            
        return result
    
    def move_left(self, grid: List[List]) -> Tuple[List[List], bool]:
        """Move and merge tiles to the left, return (new_grid, changed)"""
        grid_size = len(grid)
        new_grid = [[None for _ in range(grid_size)] for _ in range(grid_size)]
        changed = False
        
        for r in range(grid_size):
            # Process each row
            row = grid[r][:]
            new_row = self.process_row_left(row)
            
            for c in range(grid_size):
                new_grid[r][c] = new_row[c]
                if new_grid[r][c] != grid[r][c]:
                    changed = True
        
        return new_grid, changed
    
    def move_right(self, grid: List[List]) -> Tuple[List[List], bool]:
        """Move and merge tiles to the right"""
        # Reverse each row, move left, then reverse back
        reversed_grid = [row[::-1] for row in grid]
        moved_grid, changed = self.move_left(reversed_grid)
        result_grid = [row[::-1] for row in moved_grid]
        return result_grid, changed
    
    def move_up(self, grid: List[List]) -> Tuple[List[List], bool]:
        """Move and merge tiles upward"""
        grid_size = len(grid)
        # Transpose, move left, then transpose back
        transposed = [[grid[r][c] for r in range(grid_size)] for c in range(grid_size)]
        moved_transposed, changed = self.move_left(transposed)
        result_grid = [[moved_transposed[c][r] for c in range(grid_size)] for r in range(grid_size)]
        return result_grid, changed
    
    def move_down(self, grid: List[List]) -> Tuple[List[List], bool]:
        """Move and merge tiles downward"""
        grid_size = len(grid)
        # Transpose, move right, then transpose back
        transposed = [[grid[r][c] for r in range(grid_size)] for c in range(grid_size)]
        moved_transposed, changed = self.move_right(transposed)
        result_grid = [[moved_transposed[c][r] for c in range(grid_size)] for r in range(grid_size)]
        return result_grid, changed
    
    def make_move(self, grid: List[List], direction: str) -> Tuple[List[List], bool]:
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
    
    def check_win(self, grid: List[List]) -> bool:
        """Check if the player has won (reached 2048)"""
        for row in grid:
            for cell in row:
                if isinstance(cell, int) and cell >= 2048:
                    return True
        return False
    
    def can_move(self, grid: List[List]) -> bool:
        """Check if any moves are possible"""
        grid_size = len(grid)
        
        # Check for empty cells
        for row in grid:
            if None in row:
                return True
        
        # Check for possible merges horizontally
        for r in range(grid_size):
            for c in range(grid_size - 1):
                cell1, cell2 = grid[r][c], grid[r][c + 1]
                if self.can_merge(cell1, cell2):
                    return True
        
        # Check for possible merges vertically
        for r in range(grid_size - 1):
            for c in range(grid_size):
                cell1, cell2 = grid[r][c], grid[r + 1][c]
                if self.can_merge(cell1, cell2):
                    return True
        
        return False
    
    def can_merge(self, cell1, cell2) -> bool:
        """Check if two cells can merge"""
        if cell1 is None or cell2 is None:
            return False
        
        # Regular numbers can merge if they're equal and >= 2
        if (isinstance(cell1, int) and isinstance(cell2, int) and 
            cell1 == cell2 and cell1 >= 2):
            return True
            
        # '*2' can merge with numbers
        if cell1 == '*2' and isinstance(cell2, int) and cell2 >= 2:
            return True
        if cell2 == '*2' and isinstance(cell1, int) and cell1 >= 2:
            return True
            
        # '1' can merge with '*2' to become 2
        if (cell1 == 1 or cell1 == '1') and cell2 == '*2':
            return True
        if (cell2 == 1 or cell2 == '1') and cell1 == '*2':
            return True
            
        return False
    
    def process_move(self, grid: List[List], direction: str) -> Dict[str, Any]:
        """Process a move and return the result"""
        try:
            # Validate input
            if not self.is_valid_grid(grid):
                return {
                    "nextGrid": grid,
                    "endGame": None,
                    "error": "Invalid grid format"
                }
            
            # Make the move
            new_grid, changed = self.make_move(grid, direction)
            
            # If nothing changed, return the original grid
            if not changed:
                return {
                    "nextGrid": grid,
                    "endGame": None
                }
            
            # Add a new random tile
            new_grid = self.add_random_tile(new_grid)
            
            # Check win condition
            if self.check_win(new_grid):
                return {
                    "nextGrid": new_grid,
                    "endGame": "win"
                }
            
            # Check lose condition
            if not self.can_move(new_grid):
                return {
                    "nextGrid": new_grid,
                    "endGame": "lose"
                }
            
            # Game continues
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

# Global game instance
game_2048 = Game2048()

@app.route('/2048', methods=['POST', 'OPTIONS'])
def game_2048_endpoint():
    """
    2048 game endpoint
    Expected input: {"grid": NxN array, "mergeDirection": "UP"/"DOWN"/"LEFT"/"RIGHT"}
    Returns: {"nextGrid": NxN array, "endGame": null/"win"/"lose"}
    """
    # Handle preflight CORS request
    if request.method == 'OPTIONS':
        response = jsonify()
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        return response
    
    try:
        # Parse request
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
        
        # Process the move
        result = game_2048.process_move(grid, merge_direction)
        
        logger.debug(f"Result grid: {result.get('nextGrid')}")
        logger.info(f"End game status: {result.get('endGame')}")
        
        # Add CORS headers to response
        response = jsonify(result)
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
        
    except Exception as e:
        logger.error(f"Error in 2048 endpoint: {e}")
        response = jsonify({"error": "Internal server error"})
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response, 500
