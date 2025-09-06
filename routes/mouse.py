from flask import Flask, request, jsonify
import numpy as np
from collections import deque
from enum import Enum
import copy
import logging
from flask import request, jsonify
from routes import app   # reuse the same Flask app instance

logger = logging.getLogger(__name__)

from flask import Flask, request, jsonify
import numpy as np
from collections import deque
from enum import Enum
import copy
# Direction constants

# Direction constants
class Direction(Enum):
    NORTH = 0
    NORTHEAST = 1
    EAST = 2
    SOUTHEAST = 3
    SOUTH = 4
    SOUTHWEST = 5
    WEST = 6
    NORTHWEST = 7

class MicroMouseSolver:
    def __init__(self):
        self.maze_size = 16
        self.goal_center = (7, 8)  # Center of 2x2 goal area
        self.goal_cells = [(7, 7), (7, 8), (8, 7), (8, 8)]
        
        # Maze representation: 0=unknown, 1=wall, 2=open
        self.maze = np.zeros((self.maze_size, self.maze_size), dtype=int)
        self.visited = np.zeros((self.maze_size, self.maze_size), dtype=bool)
        
        # Current state
        self.current_pos = (0, 0)  # Bottom-left corner
        self.current_direction = Direction.NORTH
        self.algorithm = "floodfill"  # Default algorithm
        
        # Floodfill distance matrix
        self.flood_distances = np.full((self.maze_size, self.maze_size), float('inf'))
        self.initialize_flood_distances()
        
        self.last_sensor_data = [0, 0, 0, 0, 0]  # Store last sensor reading
        
        # DFS variables (needed for status endpoint)
        self.dfs_stack = []
        self.dfs_path = []
        self.exploration_complete = False
        self.path_to_goal = []
        
    def initialize_flood_distances(self):
        """Initialize flood distances with goal at center"""
        # Set goal cells to distance 0
        for goal in self.goal_cells:
            self.flood_distances[goal] = 0
        
        # Initialize floodfill from goal
        self.update_flood_distances()
    
    def update_flood_distances(self):
        """Update flood distances using BFS from goal"""
        queue = deque()
        for goal in self.goal_cells:
            queue.append(goal)
        
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # N, E, S, W
        
        while queue:
            row, col = queue.popleft()
            current_dist = self.flood_distances[row, col]
            
            for dr, dc in directions:
                new_row, new_col = row + dr, col + dc
                
                if (0 <= new_row < self.maze_size and 
                    0 <= new_col < self.maze_size):
                    
                    # Check if path is not blocked by wall
                    if not self.is_wall_between((row, col), (new_row, new_col)):
                        new_dist = current_dist + 1
                        if new_dist < self.flood_distances[new_row, new_col]:
                            self.flood_distances[new_row, new_col] = new_dist
                            queue.append((new_row, new_col))
    
    def is_wall_between(self, pos1, pos2):
        """Check if there's a wall between two adjacent cells"""
        # This would need actual wall data - for now assume open if not explicitly marked
        return False
    
    def update_maze_from_sensors(self, sensor_data, position, direction):
        """Update maze based on sensor readings"""
        # sensor_data: [left_90, left_45, front, right_45, right_90]
        # Convert direction to angle
        angle = direction.value * 45
        
        sensor_angles = [-90, -45, 0, 45, 90]
        row, col = position
        
        for i, sensor_reading in enumerate(sensor_data):
            sensor_angle = (angle + sensor_angles[i]) % 360
            
            # Convert angle to direction vector
            if sensor_angle == 0:    # North
                check_pos = (row + 1, col)
            elif sensor_angle == 90:  # East
                check_pos = (row, col + 1)
            elif sensor_angle == 180: # South
                check_pos = (row - 1, col)
            elif sensor_angle == 270: # West
                check_pos = (row, col - 1)
            else:
                continue  # Skip diagonal sensors for wall detection
            
            if (0 <= check_pos[0] < self.maze_size and 
                0 <= check_pos[1] < self.maze_size):
                
                if sensor_reading == 1:  # Wall detected
                    self.maze[check_pos] = 1
                elif sensor_reading == 0:  # No wall
                    self.maze[check_pos] = 2
        
        self.visited[position] = True
        self.maze[position] = 2  # Mark current cell as open
    
    def get_neighbors(self, pos):
        """Get valid neighboring cells based on sensor data"""
        row, col = pos
        neighbors = []
        
        # For now, assume we can move in any direction not blocked by walls
        # In a real implementation, you'd use the actual maze state
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # N, E, S, W
        
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if (0 <= new_row < self.maze_size and 
                0 <= new_col < self.maze_size):
                neighbors.append((new_row, new_col))
        
        return neighbors
    
    def dfs_explore(self):
        """DFS exploration algorithm"""
        current = self.current_pos
        
        if not self.exploration_complete:
            # Mark current cell as visited
            self.visited[current] = True
            
            # Get unvisited neighbors
            neighbors = self.get_neighbors(current)
            unvisited = [n for n in neighbors if not self.visited[n]]
            
            if unvisited:
                # Choose next unvisited cell
                next_cell = unvisited[0]
                self.dfs_stack.append(current)
                return self.get_movement_to_cell(current, next_cell)
            elif self.dfs_stack:
                # Backtrack
                next_cell = self.dfs_stack.pop()
                return self.get_movement_to_cell(current, next_cell)
            else:
                # Exploration complete
                self.exploration_complete = True
                self.path_to_goal = self.find_shortest_path_to_goal()
        
        # Return to goal using shortest path
        if self.path_to_goal:
            next_cell = self.path_to_goal.pop(0)
            return self.get_movement_to_cell(current, next_cell)
        
        return ["BB"]  # Stop if no path
    
    def floodfill_solve(self):
        """Floodfill algorithm implementation"""
        current = self.current_pos
        
        # Update flood distances based on current maze knowledge
        self.update_flood_distances()
        
        # Find neighbor with lowest flood distance
        neighbors = self.get_neighbors(current)
        if not neighbors:
            return ["BB"]  # Stop if trapped
        
        best_neighbor = min(neighbors, key=lambda pos: self.flood_distances[pos])
        
        # If we're at goal, stop
        if current in self.goal_cells and self.flood_distances[current] == 0:
            return ["BB"]
        
        return self.get_movement_to_cell(current, best_neighbor)
    
    def find_shortest_path_to_goal(self):
        """Find shortest path to goal using A* or Dijkstra"""
        start = self.current_pos
        goal = self.goal_center
        
        # Simple BFS for shortest path
        queue = deque([(start, [start])])
        visited = {start}
        
        while queue:
            pos, path = queue.popleft()
            
            if pos in self.goal_cells:
                return path[1:]  # Remove starting position
            
            for neighbor in self.get_neighbors(pos):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        return []
    
    def get_movement_to_cell(self, from_pos, to_pos):
        """Generate movement commands to go from one cell to another"""
        row_diff = to_pos[0] - from_pos[0]
        col_diff = to_pos[1] - from_pos[1]
        
        # Determine target direction
        if row_diff == 1 and col_diff == 0:
            target_dir = Direction.NORTH
        elif row_diff == 0 and col_diff == 1:
            target_dir = Direction.EAST
        elif row_diff == -1 and col_diff == 0:
            target_dir = Direction.SOUTH
        elif row_diff == 0 and col_diff == -1:
            target_dir = Direction.WEST
        else:
            return ["BB"]  # Invalid move
        
        # Calculate rotation needed
        current_angle = self.current_direction.value * 45
        target_angle = target_dir.value * 45
        angle_diff = (target_angle - current_angle) % 360
        
        commands = []
        
        # Add rotation commands
        if angle_diff == 45:
            commands.append("R")
        elif angle_diff == 90:
            commands.extend(["R", "R"])
        elif angle_diff == 135:
            commands.extend(["R", "R", "R"])
        elif angle_diff == 180:
            commands.extend(["R", "R", "R", "R"])
        elif angle_diff == 225:
            commands.extend(["L", "L", "L"])
        elif angle_diff == 270:
            commands.extend(["L", "L"])
        elif angle_diff == 315:
            commands.append("L")
        
        # Add forward movement
        commands.extend(["F2", "F1", "BB"])  # Accelerate, maintain, brake
        
        # Update current direction
        self.current_direction = target_dir
        
        return commands
    
    def solve(self, sensor_data, position, momentum, goal_reached):
        """Main solving method"""
        # Update position
        self.current_pos = position
        
        # Update maze from sensor data
        self.update_maze_from_sensors(sensor_data, position, self.current_direction)
        
        # If at goal, stop
        if goal_reached:
            return ["BB"]
        
        # If we're stuck (no available moves), try to turn around
        available_moves = self.get_available_moves(sensor_data)
        if not available_moves:
            return ["R", "R"]  # Turn right twice (90 degrees)
        
        # Use wall following strategy for exploration
        return self.wall_follower_strategy(sensor_data)


# Global solver instance
solver = MicroMouseSolver()



@app.route('/micro-mouse', methods=['POST'])
def micro_mouse():
    try:
        data = request.get_json()
        logger.info(f"Received payload: {data}")   
        
        # Extract data from request
        sensor_data = data.get('sensor_data', [0, 0, 0, 0, 0])
        total_time_ms = data.get('total_time_ms', 0)
        goal_reached = data.get('goal_reached', False)
        best_time_ms = data.get('best_time_ms', None)
        run_time_ms = data.get('run_time_ms', 0)
        run = data.get('run', 0)
        momentum = data.get('momentum', 0)
        game_uuid = data.get('game_uuid', '')
        
        # Convert position (assuming we track it somehow - for now use run info)
        # In a real implementation, you'd track position based on movements
        if run == 0 and run_time_ms == 0:
            position = (0, 0)  # Start position
        else:
            # For now, assume we can derive position from game state
            position = solver.current_pos
        
        # Check if we should end
        if total_time_ms > 290000:  # Near time limit
            return jsonify({
                "instructions": [],
                "end": True
            })
        
        # Switch algorithm based on run
        if run < 3:
            solver.algorithm = "dfs"  # Explore first
        else:
            solver.algorithm = "floodfill"  # Optimize later runs
        
        # Get movement commands
        instructions = solver.solve(sensor_data, position, momentum, goal_reached)
        
        return jsonify({
            "instructions": instructions,
            "end": False
        })
        
    except Exception as e:
        print(f"Error in micro_mouse endpoint: {e}")
        return jsonify({
            "instructions": ["BB"],
            "end": True
        }), 500

@app.route('/micro-mouse/status', methods=['GET'])
def get_status():
    """Get current solver status"""
    return jsonify({
        "algorithm": solver.algorithm,
        "position": solver.current_pos,
        "direction": solver.current_direction.name,
        "exploration_complete": solver.exploration_complete,
        "visited_cells": int(np.sum(solver.visited))
    })

@app.route('/micro-mouse/reset', methods=['POST'])
def reset_solver():
    """Reset the solver for a new game"""
    global solver
    solver = MicroMouseSolver()
    return jsonify({"status": "reset"})

@app.route('/micro-mouse/algorithm', methods=['POST'])
def set_algorithm():
    """Set the solving algorithm"""
    data = request.get_json()
    algorithm = data.get('algorithm', 'floodfill')
    
    if algorithm in ['dfs', 'floodfill']:
        solver.algorithm = algorithm
        return jsonify({"algorithm": algorithm})
    else:
        return jsonify({"error": "Invalid algorithm"}), 400








