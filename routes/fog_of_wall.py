import os
import sys
import logging
from flask import request, jsonify
import json
from collections import deque, defaultdict
import heapq
import math

# Add parent directory to path to allow importing routes module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import logging

from flask import request

from routes import app

logger = logging.getLogger(__name__)


"""
Flask endpoint implementing an efficient strategy for the Fog-of-Wall challenge.

This solution uses a *preplanned scanning pattern* to cover the entire grid with
a small number of scans. The maze is a square grid of size `length_of_grid`.
Each scan reveals the 5×5 area centred on a crow. By placing scans on a
lattice with spacing 5 cells (centres at positions 2, 7, 12, …), we can cover
every cell in the grid in a predictable number of scans.  Crows are assigned
to these scanning centres and move using a breadth-first search (BFS) path
finder that avoids known walls but allows unknown cells.  Once a crow reaches
a scanning centre it performs a scan (only if that scan would reveal new
information), otherwise it moves toward its next assigned centre.  When all
walls have been discovered (or no remaining scan centres need scanning), the
agent submits the discovered wall positions.

The endpoint expects POST requests in the format specified by the problem
statement.  State is kept per `(challenger_id, game_id)` so multiple test
cases can run concurrently.  The code defensively handles malformed inputs,
different shapes of `move_result`, and avoids loops when bumping into walls.

"""

import json
import logging
from collections import deque
from typing import Dict, Tuple, List, Set, Optional

from flask import request
from routes import app  # Assumes there is an existing Flask app instance

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Global state keyed by (challenger_id, game_id)
STATE: Dict[Tuple[str, str], Dict] = {}

# Direction vectors and a stable order for iteration
DIRS = {
    "N": (0, -1),
    "S": (0, 1),
    "W": (-1, 0),
    "E": (1, 0),
}
DIR_ORDER = ["N", "S", "W", "E"]  # used for deterministic tie-breaking


def within_bounds(x: int, y: int, L: int) -> bool:
    """Return True if (x, y) lies within a 0≤x,y< L grid."""
    return 0 <= x < L and 0 <= y < L


def add_cell(known: Dict[Tuple[int, int], str], pos: Tuple[int, int], tag: str) -> None:
    """Record knowledge for a cell as 'W' (wall) or 'E' (empty). Don't downgrade walls."""
    if tag not in ("W", "E"):
        return
    cur = known.get(pos)
    if cur == "W":
        return
    if cur == "E" and tag == "E":
        return
    known[pos] = tag


def parse_move_result(mr) -> Optional[Tuple[int, int]]:
    """
    Accept a `move_result` either as a two-element list [x, y] or as a dict
    with 'x' and 'y' keys.  Returns a tuple (x, y) or `None` if parsing
    fails.
    """
    if isinstance(mr, list) and len(mr) == 2:
        try:
            return int(mr[0]), int(mr[1])
        except (TypeError, ValueError):
            return None
    if isinstance(mr, dict) and "x" in mr and "y" in mr:
        try:
            return int(mr["x"]), int(mr["y"])
        except (TypeError, ValueError):
            return None
    return None


def update_known_from_scan(center: Tuple[int, int], scan: List[List[str]], L: int,
                           known: Dict[Tuple[int, int], str]) -> None:
    """
    Update the `known` map based on a 5×5 scan result centred at `center`.
    The scan is a list of five lists of five characters.  Cells labelled 'W'
    become walls; '_' and 'C' (the scanning crow itself) mark empty cells; 'X'
    represents out-of-bounds and is ignored.  Only in-bounds coordinates are
    recorded.
    """
    cx, cy = center
    if not (isinstance(scan, list) and len(scan) == 5 and all(len(r) == 5 for r in scan)):
        return
    for j in range(5):
        for i in range(5):
            symbol = scan[j][i]
            dx, dy = i - 2, j - 2
            x, y = cx + dx, cy + dy
            if symbol == "X":
                continue  # out of bounds
            if within_bounds(x, y, L):
                if symbol == "W":
                    add_cell(known, (x, y), "W")
                else:  # '_' or 'C'
                    add_cell(known, (x, y), "E")


def scan_would_reveal(center: Tuple[int, int], L: int,
                      known: Dict[Tuple[int, int], str]) -> bool:
    """
    Return True if a scan at `center` would reveal at least one unknown
    in-bounds cell.  It checks the 5×5 neighbourhood around `center`.
    """
    x0, y0 = center
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            x, y = x0 + dx, y0 + dy
            if within_bounds(x, y, L) and (x, y) not in known:
                return True
    return False


def compute_scan_positions(L: int) -> List[int]:
    """
    Compute the list of axis positions (along x or y) for scanning centres.
    Starting at coordinate 2 and stepping by 5 ensures each 5×5 scan covers
    adjacent ranges without gaps.  For tiny grids (L <= 3) this returns [0]
    so there is at least one centre.
    """
    positions: List[int] = []
    pos = 2
    # Place centres every 5 cells (2, 7, 12, …) as long as pos < L
    while pos < L:
        positions.append(pos)
        pos += 5
    # Fallback for small grids: ensure there's at least one centre within [0, L-1]
    if not positions:
        positions.append(min(2, L - 1))
    return positions


def bfs_next_step(start: Tuple[int, int], goal: Tuple[int, int], L: int,
                  known: Dict[Tuple[int, int], str]) -> Optional[str]:
    """
    Compute the first step along a shortest path from `start` to `goal`
    avoiding cells known to be walls.  Unknown cells are treated as passable.
    Returns a direction character ('N', 'S', 'E', 'W') or `None` if
    already at the goal or no path exists.
    """
    if start == goal:
        return None
    # BFS queue holds positions; `parent` maps child -> parent to reconstruct path
    q = deque([start])
    parent: Dict[Tuple[int, int], Tuple[int, int]] = {start: start}
    while q:
        x, y = q.popleft()
        for d in DIR_ORDER:
            dx, dy = DIRS[d]
            nx, ny = x + dx, y + dy
            np = (nx, ny)
            if not within_bounds(nx, ny, L):
                continue
            if known.get(np) == "W":
                continue  # cannot go through known walls
            if np in parent:
                continue  # already visited
            parent[np] = (x, y)
            if np == goal:
                # Reconstruct the path back to start to find the first step
                cur = np
                while parent[cur] != start:
                    cur = parent[cur]
                fx, fy = cur
                dx2, dy2 = fx - start[0], fy - start[1]
                for direction, (vx, vy) in DIRS.items():
                    if (vx, vy) == (dx2, dy2):
                        return direction
                # fallthrough if direction not found (shouldn't happen)
            q.append(np)
    # No path found (perhaps boxed in by walls); try a greedy nudge
    sx, sy = start
    tx, ty = goal
    greedy: List[str] = []
    if tx > sx:
        greedy.append("E")
    if tx < sx:
        greedy.append("W")
    if ty > sy:
        greedy.append("S")
    if ty < sy:
        greedy.append("N")
    # Append the other directions as fallbacks to avoid being stuck
    for D in DIR_ORDER:
        if D not in greedy:
            greedy.append(D)
    for D in greedy:
        dx, dy = DIRS[D]
        nx, ny = sx + dx, sy + dy
        if within_bounds(nx, ny, L) and known.get((nx, ny)) != "W":
            return D
    return None


def decide_action(state: Dict) -> Dict:
    """
    Decide the next action based on the current state.  The strategy is to
    follow a precomputed list of scanning centres to cover the entire grid.
    Each crow is assigned to the nearest unvisited scanning centre.  When a
    crow arrives at a centre and scanning there would reveal something, it
    scans.  Otherwise, the crow moves one step along a path to its target.
    Once all walls have been discovered or all centres are complete, the
    strategy submits the wall positions.
    """
    L = state["grid_size"]
    known = state["known"]
    crows = state["crows"]
    num_walls = state["num_walls"]
    centres_to_scan: Set[Tuple[int, int]] = state["scan_centres_to_scan"]
    centres_done: Set[Tuple[int, int]] = state["scan_centres_done"]
    targets = state["targets"]
    last_failed_edge: Set[Tuple[int, int, str]] = state["last_failed_edge"]

    # If we already discovered all walls, submit immediately
    if len([1 for v in known.values() if v == "W"]) >= num_walls:
        return {"action_type": "submit"}

    # Remove centres that no longer need scanning (no unknown cells in 5×5 area)
    # We iterate over a copy to avoid modifying set during iteration
    for centre in list(centres_to_scan):
        if not scan_would_reveal(centre, L, known):
            centres_to_scan.remove(centre)
            centres_done.add(centre)

    # If all centres are processed and we haven't found all walls, fall back to
    # exploring unknown cells with local scans (same as previous strategy)
    if not centres_to_scan:
        # If there are still unknown cells, use the original frontier-based scanning
        # strategy to clear them; otherwise, submit whatever walls we have.
        # Build frontier cells: unknown cells
        frontiers: Set[Tuple[int, int]] = set()
        for y in range(L):
            for x in range(L):
                if (x, y) not in known and known.get((x, y)) != "W":
                    frontiers.add((x, y))
        # If no unknown cells remain, submit
        if not frontiers:
            return {"action_type": "submit"}
        # Otherwise behave like previous algorithm: scan if standing on a cell
        # whose scan would reveal new info; else move to nearest frontier
        # 1) Scan if any crow can reveal new info at current location
        for cid, pos in crows.items():
            if scan_would_reveal(pos, L, known):
                return {"action_type": "scan", "crow_id": cid}
        # 2) Assign nearest frontier to each crow and move toward it
        # Simple greedy assignment per crow
        best_cid = None
        best_goal = None
        best_dist = 10**9
        for cid, pos in crows.items():
            # find nearest frontier
            nearest = None
            nearest_d = 10**9
            for f in frontiers:
                d = abs(pos[0] - f[0]) + abs(pos[1] - f[1])
                if d < nearest_d:
                    nearest = f
                    nearest_d = d
            if nearest is not None and nearest_d < best_dist:
                best_cid = cid
                best_goal = nearest
                best_dist = nearest_d
        # If we found a target, move toward it
        if best_cid is not None:
            step = bfs_next_step(crows[best_cid], best_goal, L, known)
            # If BFS fails, just scan to learn more
            if step is None:
                return {"action_type": "scan", "crow_id": best_cid}
            return {"action_type": "move", "crow_id": best_cid, "direction": step}
        # Fallback: scan with any crow
        any_cid = next(iter(crows.keys()))
        return {"action_type": "scan", "crow_id": any_cid}

    # Primary scanning plan: check if any crow is standing on a centre needing scanning
    for cid, pos in crows.items():
        if pos in centres_to_scan:
            return {"action_type": "scan", "crow_id": cid}

    # Assign targets to crows if they don't have one or their target is complete
    assigned: Set[Tuple[int, int]] = set(targets.values())
    for cid, pos in crows.items():
        tgt = targets.get(cid)
        if tgt not in centres_to_scan:
            # find nearest centre not taken
            nearest = None
            nearest_d = 10**9
            for centre in centres_to_scan:
                if centre in assigned:
                    continue
                d = abs(pos[0] - centre[0]) + abs(pos[1] - centre[1])
                if d < nearest_d:
                    nearest = centre
                    nearest_d = d
            if nearest is not None:
                targets[cid] = nearest
                assigned.add(nearest)

    # Choose the crow that can make progress to its target fastest (smallest
    # Manhattan distance).  Compute BFS step and issue a move.
    selected_cid = None
    selected_goal = None
    best_dist = 10**9
    for cid, pos in crows.items():
        tgt = targets.get(cid)
        if tgt is None:
            continue
        d = abs(pos[0] - tgt[0]) + abs(pos[1] - tgt[1])
        if d < best_dist:
            best_dist = d
            selected_cid = cid
            selected_goal = tgt

    # There should always be a selected crow here because centres_to_scan is not empty
    if selected_cid is None or selected_goal is None:
        # Fallback to scanning with any crow
        any_cid = next(iter(crows.keys()))
        return {"action_type": "scan", "crow_id": any_cid}

    # Determine next step via BFS
    step = bfs_next_step(crows[selected_cid], selected_goal, L, known)
    # If no step (already there or path blocked), we scan or pick another direction
    if step is None:
        # At the goal but not scanning centre? or path blocked; just scan to learn more
        return {"action_type": "scan", "crow_id": selected_cid}

    # Avoid repeating known failed edge (loop protection)
    cx, cy = crows[selected_cid]
    if (cx, cy, step) in last_failed_edge:
        # Try alternate directions
        for alt in DIR_ORDER:
            if alt == step:
                continue
            dx, dy = DIRS[alt]
            nx, ny = cx + dx, cy + dy
            if within_bounds(nx, ny, L) and known.get((nx, ny)) != "W":
                step = alt
                break
    return {"action_type": "move", "crow_id": selected_cid, "direction": step}


def walls_as_strings(known: Dict[Tuple[int, int], str]) -> List[str]:
    """Convert positions of known walls into the required 'x-y' string format."""
    return [f"{x}-{y}" for (x, y), tag in known.items() if tag == "W"]


@app.route('/fog-of-wall', methods=['POST'])
def fog_of_wall():  # pragma: no cover - entry point for server
    """
    Core endpoint for the Fog-of-Wall game.  It manages per-game state,
    updates knowledge from the previous action's result, and selects the next
    action using the scanning pattern strategy.  Returns JSON with one of
    three actions: `move`, `scan`, or `submit`.
    """
    payload = request.get_json(force=True, silent=True) or {}
    logger.info("Fog-of-Wall request: %s", payload)

    challenger_id = str(payload.get("challenger_id", ""))
    game_id = str(payload.get("game_id", ""))
    if not challenger_id or not game_id:
        return json.dumps({"error": "challenger_id and game_id are required"}), 400
    key = (challenger_id, game_id)

    # Initialise a new game when `test_case` is present
    if key not in STATE and payload.get("test_case"):
        tc = payload["test_case"] or {}
        L = int(tc.get("length_of_grid", 0))
        num_walls = int(tc.get("num_of_walls", 0))
        crows_list = tc.get("crows", [])
        crows: Dict[str, Tuple[int, int]] = {str(c["id"]): (int(c["x"]), int(c["y"]))
                                             for c in crows_list}
        # Known map: mark initial crow positions as empty
        known: Dict[Tuple[int, int], str] = {}
        for pos in crows.values():
            add_cell(known, pos, "E")
        # Compute scanning centres on a lattice covering the grid
        axis_positions = compute_scan_positions(L)
        scan_centres: Set[Tuple[int, int]] = set((x, y) for x in axis_positions for y in axis_positions)
        STATE[key] = {
            "grid_size": L,
            "num_walls": num_walls,
            "crows": crows,
            "known": known,
            "scan_centres_to_scan": set(scan_centres),
            "scan_centres_done": set(),
            "targets": {},  # crow_id -> centre
            "last_failed_edge": set(),  # track bump edges to avoid loops
        }

    # If the game has not been initialised properly, reject
    if key not in STATE:
        return json.dumps({"error": "Unknown game. Send initial test_case first."}), 400

    state = STATE[key]
    L = state["grid_size"]
    known = state["known"]
    crows = state["crows"]

    # -------------------------------------------------------------------------
    # Ingest previous action result
    prev = payload.get("previous_action") or {}
    if prev:
        act = prev.get("your_action")
        cid = str(prev.get("crow_id")) if prev.get("crow_id") is not None else None
        # Handle move action
        if act == "move" and cid in crows:
            direction = prev.get("direction")
            mr = parse_move_result(prev.get("move_result"))
            old_pos = crows[cid]
            if mr is not None:
                new_pos = mr
                # If we didn't move, we bumped into a wall; record it
                if new_pos == old_pos:
                    if direction in DIRS:
                        dx, dy = DIRS[direction]
                        wx, wy = old_pos[0] + dx, old_pos[1] + dy
                        if within_bounds(wx, wy, L):
                            add_cell(known, (wx, wy), "W")
                        # Record failed edge to avoid repeating
                        state["last_failed_edge"].add((old_pos[0], old_pos[1], direction))
                else:
                    # Successful move
                    crows[cid] = new_pos
                    add_cell(known, new_pos, "E")
        # Handle scan action
        elif act == "scan" and cid in crows:
            scan = prev.get("scan_result")
            update_known_from_scan(crows[cid], scan, L, known)
            # If the crow was standing on a planned centre, mark it done
            pos = crows[cid]
            if pos in state["scan_centres_to_scan"]:
                state["scan_centres_to_scan"].remove(pos)
                state["scan_centres_done"].add(pos)

    # -------------------------------------------------------------------------
    # Decide next action based on updated state
    decision = decide_action(state)

    # Submit action
    if decision.get("action_type") == "submit":
        submission = walls_as_strings(known)
        resp = {
            "challenger_id": challenger_id,
            "game_id": game_id,
            "action_type": "submit",
            "submission": submission,
        }
        logger.info("Submitting %d/%d walls", len(submission), state["num_walls"])
        # Remove state for this game to free memory
        STATE.pop(key, None)
        return json.dumps(resp)

    # Scan action
    if decision.get("action_type") == "scan":
        resp = {
            "challenger_id": challenger_id,
            "game_id": game_id,
            "crow_id": decision["crow_id"],
            "action_type": "scan",
        }
        logger.info("Action: scan with crow %s", decision["crow_id"])
        return json.dumps(resp)

    # Move action
    if decision.get("action_type") == "move":
        resp = {
            "challenger_id": challenger_id,
            "game_id": game_id,
            "crow_id": decision["crow_id"],
            "action_type": "move",
            "direction": decision["direction"],
        }
        logger.info("Action: move crow %s %s", decision["crow_id"], decision["direction"])
        return json.dumps(resp)

    # Fallback (shouldn't happen) – default to scanning with the first crow
    any_cid = next(iter(crows.keys()))
    return json.dumps({
        "challenger_id": challenger_id,
        "game_id": game_id,
        "crow_id": any_cid,
        "action_type": "scan",
    })


# class FogOfWallGame:
#     """
#     Game state manager for Fog of Wall maze exploration
#     """
    
#     def __init__(self):
#         self.games = {}  # Store game states by game_id
        
#     def start_new_game(self, game_id, test_case):
#         """Initialize a new game with test case data"""
#         # Handle None or missing test_case
#         if not test_case:
#             logger.error(f"Empty or None test_case provided for game {game_id}")
#             raise ValueError("test_case cannot be None or empty")
            
#         # Validate test_case structure
#         if not isinstance(test_case, dict):
#             logger.error(f"Invalid test_case type for game {game_id}: {type(test_case)}")
#             raise ValueError(f"test_case must be a dictionary, got {type(test_case)}")
            
#         # Handle None or missing crows
#         crows_data = test_case.get('crows', [])
#         if not crows_data:
#             logger.warning(f"No crows data in test_case for game {game_id}")
#             crows_data = []
#         elif not isinstance(crows_data, list):
#             logger.error(f"Invalid crows data type for game {game_id}: {type(crows_data)}")
#             raise ValueError(f"crows must be a list, got {type(crows_data)}")
            
#         # Safely process crows, filtering out None values
#         crows = {}
#         for i, crow in enumerate(crows_data):
#             if crow is None:
#                 logger.warning(f"Skipping None crow at index {i} in game {game_id}")
#                 continue
#             if not isinstance(crow, dict):
#                 logger.warning(f"Skipping invalid crow at index {i} in game {game_id}: {type(crow)}")
#                 continue
#             if 'id' not in crow or 'x' not in crow or 'y' not in crow:
#                 logger.warning(f"Skipping crow at index {i} in game {game_id} missing required fields: {crow}")
#                 continue
#             try:
#                 x, y = int(crow['x']), int(crow['y'])
#                 crows[crow['id']] = {'x': x, 'y': y}
#             except (ValueError, TypeError) as e:
#                 logger.warning(f"Skipping crow at index {i} in game {game_id} with invalid coordinates: {e}")
#                 continue
        
#         if not crows:
#             logger.error(f"No valid crows found in test_case for game {game_id}")
#             raise ValueError("No valid crows found in test_case")
        
#         # Validate grid size
#         grid_size = test_case.get('length_of_grid', 10)
#         try:
#             grid_size = int(grid_size)
#             if grid_size <= 0:
#                 logger.warning(f"Invalid grid_size {grid_size} for game {game_id}, using default 10")
#                 grid_size = 10
#         except (ValueError, TypeError):
#             logger.warning(f"Invalid grid_size type for game {game_id}, using default 10")
#             grid_size = 10
            
#         # Validate number of walls
#         num_walls = test_case.get('num_of_walls', 0)
#         try:
#             num_walls = int(num_walls)
#             if num_walls < 0:
#                 logger.warning(f"Invalid num_walls {num_walls} for game {game_id}, using 0")
#                 num_walls = 0
#         except (ValueError, TypeError):
#             logger.warning(f"Invalid num_walls type for game {game_id}, using 0")
#             num_walls = 0
        
#         self.games[game_id] = {
#             'crows': crows,
#             'grid_size': grid_size,
#             'num_walls': num_walls,
#             'discovered_walls': set(),
#             'explored_cells': set(),
#             'scan_results': {},  # Store scan results for each position
#             'move_count': 0,
#             'game_complete': False,
#             'max_moves': min(grid_size * grid_size, 200)  # Limit moves to prevent timeout
#         }
#         logger.info(f"Started new game {game_id} with {len(crows)} crows, grid_size={grid_size}, num_walls={num_walls}")
        
#     def get_crow_position(self, game_id, crow_id):
#         """Get current position of a crow"""
#         if game_id not in self.games:
#             return None
#         return self.games[game_id]['crows'].get(crow_id)
        
#     def update_crow_position(self, game_id, crow_id, new_x, new_y):
#         """Update crow position after a move"""
#         if game_id in self.games and crow_id in self.games[game_id]['crows']:
#             self.games[game_id]['crows'][crow_id] = {'x': new_x, 'y': new_y}
#             self.games[game_id]['move_count'] += 1
            
#     def add_scan_result(self, game_id, crow_id, x, y, scan_data):
#         """Process and store scan results"""
#         if game_id not in self.games:
#             return
            
#         if not scan_data or not isinstance(scan_data, list):
#             return
            
#         game = self.games[game_id]
#         game['move_count'] += 1
        
#         # Mark the center cell as explored
#         game['explored_cells'].add((x, y))
        
#         # Process the 5x5 scan grid
#         for i, row in enumerate(scan_data):
#             if not isinstance(row, list):
#                 continue
#             for j, cell in enumerate(row):
#                 if cell == 'W':  # Wall found
#                     # Convert relative position to absolute
#                     wall_x = x + (j - 2)  # j-2 because center is at [2][2]
#                     wall_y = y + (i - 2)  # i-2 because center is at [2][2]
                    
#                     # Check bounds
#                     if 0 <= wall_x < game['grid_size'] and 0 <= wall_y < game['grid_size']:
#                         game['discovered_walls'].add((wall_x, wall_y))
                        
#         # Store scan result for this position
#         game['scan_results'][(x, y)] = scan_data
        
#     def get_discovered_walls(self, game_id):
#         """Get all discovered walls in submission format"""
#         if game_id not in self.games:
#             return []
#         return [f"{x}-{y}" for x, y in self.games[game_id]['discovered_walls']]
        
#     def is_game_complete(self, game_id):
#         """Check if all walls have been discovered"""
#         if game_id not in self.games:
#             return False
#         game = self.games[game_id]
#         return len(game['discovered_walls']) >= game['num_walls']
        
#     def get_game_stats(self, game_id):
#         """Get current game statistics"""
#         if game_id not in self.games:
#             return None
#         game = self.games[game_id]
#         return {
#             'walls_discovered': len(game['discovered_walls']),
#             'total_walls': game['num_walls'],
#             'cells_explored': len(game['explored_cells']),
#             'move_count': game['move_count'],
#             'completion_percentage': len(game['discovered_walls']) / game['num_walls'] * 100
#         }

# class MazeExplorer:
#     """
#     Intelligent maze exploration strategy with multi-crow coordination
#     """
    
#     def __init__(self, grid_size):
#         self.grid_size = grid_size
#         self.explored = set()
#         self.walls = set()
#         self.frontier = set()
#         self.crow_assignments = {}  # Track which areas each crow is exploring
        
#     def get_next_action(self, game_state, crows):
#         """
#         Determine the best next action for any crow using optimized exploration
#         Returns (crow_id, action_type, direction_or_none)
#         """
#         # Check if we should submit early (found enough walls or running out of moves)
#         walls_found = len(game_state['discovered_walls'])
#         total_walls = game_state['num_walls']
#         moves_used = game_state['move_count']
#         max_moves = game_state['max_moves']
        
#         # Submit early if we found all walls or are running out of moves
#         # Be more conservative about early submission to find more walls
#         if (walls_found >= total_walls or 
#             moves_used >= max_moves * 0.95):  # Submit at 95% of max moves
#             logger.info(f"Submitting: walls={walls_found}/{total_walls}, moves={moves_used}/{max_moves}")
#             return None, 'submit', None
        
#         # Strategy: Prioritize scanning unexplored positions, then move to new areas
        
#         # First, find any crow that can scan an unexplored position
#         # Prioritize crows in areas with more potential for wall discovery
#         best_scan_crow = None
#         best_scan_score = -1
        
#         for crow_id, crow_pos in crows.items():
#             if not crow_pos or not isinstance(crow_pos, dict):
#                 continue
#             x, y = crow_pos['x'], crow_pos['y']
            
#             # Skip if already scanned this position
#             if (x, y) in game_state['scan_results']:
#                 continue
            
#             # Calculate scan value for this position
#             scan_score = self._calculate_scan_value(x, y, game_state)
#             if scan_score > best_scan_score:
#                 best_scan_score = scan_score
#                 best_scan_crow = crow_id
                
#         if best_scan_crow:
#             logger.info(f"Scanning with crow {best_scan_crow} at position {crows[best_scan_crow]}")
#             return best_scan_crow, 'scan', None
        
#         # If all crows have scanned their positions, find the best move
#         best_crow = None
#         best_direction = None
#         best_score = -1
        
#         for crow_id, crow_pos in crows.items():
#             if not crow_pos or not isinstance(crow_pos, dict):
#                 continue
                
#             x = crow_pos.get('x')
#             y = crow_pos.get('y')
            
#             if x is None or y is None:
#                 continue
            
#             # Try each direction and score the move
#             for direction in ['N', 'S', 'E', 'W']:
#                 if not self._is_valid_move(crow_pos, direction, game_state):
#                     continue
                    
#                 new_x, new_y = self._get_new_position(x, y, direction)
                
#                 # Skip if we've already explored this position
#                 if (new_x, new_y) in game_state['explored_cells']:
#                     continue
                
#                 # Calculate score for this move
#                 score = self._calculate_move_score(new_x, new_y, game_state)
                
#                 if score > best_score:
#                     best_score = score
#                     best_crow = crow_id
#                     best_direction = direction
        
#         if best_crow and best_direction:
#             logger.info(f"Moving crow {best_crow} {best_direction} from {crows[best_crow]}")
#             return best_crow, 'move', best_direction
        
#         # If no good moves found, try any valid move (even to explored areas)
#         for crow_id, crow_pos in crows.items():
#             if not crow_pos or not isinstance(crow_pos, dict):
#                 continue
                
#             x = crow_pos.get('x')
#             y = crow_pos.get('y')
            
#             if x is None or y is None:
#                 continue
            
#             # Try each direction for any valid move
#             for direction in ['N', 'S', 'E', 'W']:
#                 if not self._is_valid_move(crow_pos, direction, game_state):
#                     continue
                    
#                 new_x, new_y = self._get_new_position(x, y, direction)
                
#                 # Even if explored, try to move there if it's not a wall
#                 return crow_id, 'move', direction
        
#         # If absolutely no moves possible, submit what we have
#         logger.warning("No valid moves found, submitting current results")
#         return None, 'submit', None
    
#     def _calculate_move_score(self, x, y, game_state):
#         """Calculate how valuable it would be to move to position (x, y)"""
#         if not (0 <= x < self.grid_size and 0 <= y < self.grid_size):
#             return 0
            
#         # Check if position is already explored
#         if (x, y) in game_state['explored_cells']:
#             return 0
            
#         # Check if position is a known wall
#         if (x, y) in game_state['discovered_walls']:
#             return 0
            
#         # Base score for unexplored position
#         score = 20  # Higher base score to encourage exploration
        
#         # Bonus for being near unexplored areas (potential for scanning)
#         unexplored_nearby = 0
#         for dx in range(-2, 3):
#             for dy in range(-2, 3):
#                 check_x, check_y = x + dx, y + dy
#                 if (0 <= check_x < self.grid_size and 0 <= check_y < self.grid_size):
#                     if (check_x, check_y) not in game_state['explored_cells']:
#                         unexplored_nearby += 1
        
#         # Higher score for positions with more unexplored nearby cells
#         score += min(unexplored_nearby, 15)  # Increased cap
        
#         # Bonus for being far from explored areas (frontier exploration)
#         min_distance = float('inf')
#         for explored_x, explored_y in game_state['explored_cells']:
#             distance = abs(x - explored_x) + abs(y - explored_y)
#             min_distance = min(min_distance, distance)
            
#         if min_distance != float('inf'):
#             # Prefer positions that are far from explored areas
#             score += min(min_distance, 10)  # Increased bonus
#         else:
#             # Unexplored area - very high priority
#             score += 20
        
#         return score
        
        
                
        
        
        
        
        
            
#     def _calculate_scan_value(self, x, y, game_state):
#         """Calculate how valuable it would be to scan at position (x, y)"""
#         if not (0 <= x < self.grid_size and 0 <= y < self.grid_size):
#             return 0
            
#         # Skip if already scanned
#         if (x, y) in game_state['scan_results']:
#             return 0
            
#         value = 0
        
#         # Count unexplored cells in 5x5 area that could be walls
#         unexplored_count = 0
#         for dx in range(-2, 3):
#             for dy in range(-2, 3):
#                 check_x, check_y = x + dx, y + dy
#                 if (0 <= check_x < self.grid_size and 0 <= check_y < self.grid_size):
#                     if (check_x, check_y) not in game_state['explored_cells']:
#                         unexplored_count += 1
        
#         # Base value on unexplored cells in scan area
#         value += unexplored_count * 3  # Increased multiplier
        
#         # Bonus for being near known walls (might discover more walls nearby)
#         wall_proximity = 0
#         for wall_x, wall_y in game_state['discovered_walls']:
#             distance = abs(x - wall_x) + abs(y - wall_y)
#             if distance <= 4:  # Within 4 cells of a known wall
#                 wall_proximity += 1
                
#         if wall_proximity > 0:
#             value += wall_proximity * 5  # Increased bonus
            
#         return value
        
#     def _is_valid_move(self, crow_pos, direction, game_state):
#         """Check if a move in the given direction is valid"""
#         if not crow_pos or not isinstance(crow_pos, dict):
#             return False
            
#         x = crow_pos.get('x')
#         y = crow_pos.get('y')
        
#         if x is None or y is None:
#             return False
            
#         new_x, new_y = self._get_new_position(x, y, direction)
        
#         if new_x is None or new_y is None:
#             return False
        
#         # Check bounds
#         if not (0 <= new_x < self.grid_size and 0 <= new_y < self.grid_size):
#             return False
        
#         # Check if the destination is a known wall
#         if (new_x, new_y) in game_state['discovered_walls']:
#             return False
            
#         return True
        
#     def _get_new_position(self, x, y, direction):
#         """Get new position after moving in given direction"""
#         if x is None or y is None:
#             return None, None
            
#         if direction == 'N':
#             return x, y - 1
#         elif direction == 'S':
#             return x, y + 1
#         elif direction == 'E':
#             return x + 1, y
#         elif direction == 'W':
#             return x - 1, y
#         return x, y

# # Global game manager
# game_manager = FogOfWallGame()

# @app.route('/fog-of-wall', methods=['POST'])
# def fog_of_wall():
#     """
#     Main endpoint for Fog of Wall game
#     Handles initial setup, move results, scan results, and submissions
#     """
#     try:
#         # Get raw data first for debugging
#         raw_data = request.get_data()
#         logger.info(f"Raw request data: {raw_data}")
        
#         try:
#             payload = request.get_json(force=True)
#         except Exception as e:
#             logger.error(f"Failed to parse JSON: {e}")
#             return jsonify({'error': 'Invalid JSON in request body'}), 400
            
#         if not payload:
#             logger.error("Empty payload received")
#             return jsonify({'error': 'Empty request body'}), 400
            
#         challenger_id = payload.get('challenger_id')
#         game_id = payload.get('game_id')
        
#         # Determine request type for better logging
#         has_test_case = 'test_case' in payload and payload['test_case'] is not None and payload['test_case'] != 'null'
#         has_previous_action = 'previous_action' in payload and payload['previous_action'] is not None
        
#         request_type = "initial" if has_test_case else "follow-up" if has_previous_action else "invalid"
#         logger.info(f"Received {request_type} request: challenger_id={challenger_id}, game_id={game_id}, has_test_case={has_test_case}, has_previous_action={has_previous_action}")
        
#         # Log full payload for debugging
#         logger.info(f"Full payload: {payload}")
        
#         # Log test_case structure for debugging
#         if 'test_case' in payload:
#             test_case = payload['test_case']
#             logger.info(f"Test case data: {test_case} (type: {type(test_case)})")
#             if isinstance(test_case, dict) and 'crows' in test_case:
#                 logger.info(f"Crows data: {test_case['crows']}")
#             elif test_case is None or test_case == 'null':
#                 logger.info("Test case is None/null - this is expected for follow-up requests")
        
#         if not challenger_id or not game_id:
#             return jsonify({'error': 'Missing challenger_id or game_id'}), 400
            
#         # Check if this is an initial request with valid test_case data
#         if 'test_case' in payload and payload['test_case'] is not None and payload['test_case'] != 'null':
#             test_case = payload['test_case']
            
#             # Add validation for test_case
#             if not isinstance(test_case, dict):
#                 logger.error(f"Invalid test_case type: {type(test_case)}, value: {test_case}")
#                 return jsonify({'error': 'Invalid test_case data - must be a dictionary'}), 400
                
#             # Validate crows data before starting game
#             crows_data = test_case.get('crows', [])
#             if not crows_data or not isinstance(crows_data, list):
#                 logger.error(f"Invalid crows data: {crows_data}")
#                 return jsonify({'error': 'No valid crows data found in test_case'}), 400
                
#             # Check if game already exists
#             if game_id in game_manager.games:
#                 logger.warning(f"Game {game_id} already exists, restarting with new test case")
#                 # Restart the game with the new test case
#                 try:
#                     game_manager.start_new_game(game_id, test_case)
#                 except Exception as e:
#                     logger.error(f"Failed to restart game: {str(e)}")
#                     return jsonify({'error': f'Failed to restart game: {str(e)}'}), 400
#             else:
#                 # Initialize new game
#                 try:
#                     game_manager.start_new_game(game_id, test_case)
#                 except Exception as e:
#                     logger.error(f"Failed to start new game: {str(e)}")
#                     return jsonify({'error': f'Failed to start new game: {str(e)}'}), 400
            
#             # Get initial action
#             game_state = game_manager.games[game_id]
#             crows = game_state['crows']
            
#             # Check if we have any crows
#             if not crows:
#                 return jsonify({'error': 'No crows available'}), 400
                
#             # Start with scanning at initial positions
#             first_crow_id = list(crows.keys())[0]
#             return jsonify({
#                 'challenger_id': challenger_id,
#                 'game_id': game_id,
#                 'crow_id': first_crow_id,
#                 'action_type': 'scan'
#             })
            
#         # Handle previous action result
#         previous_action = payload.get('previous_action')
#         if not previous_action:
#             # If we don't have a test_case and no previous_action, this is an invalid request
#             logger.error(f"Invalid request: no test_case and no previous_action for game {game_id}")
#             return jsonify({'error': 'Invalid request: must provide either test_case or previous_action'}), 400
            
#         action_type = previous_action.get('your_action')
#         crow_id = previous_action.get('crow_id')
        
#         # Validate that we have the required fields
#         if not action_type or not crow_id:
#             return jsonify({'error': 'Missing action_type or crow_id in previous_action'}), 400
            
#         # Check if game exists
#         if game_id not in game_manager.games:
#             logger.error(f"Game {game_id} not found when processing previous action")
#             return jsonify({'error': 'Game not found'}), 404
        
#         if action_type == 'move':
#             # Process move result
#             move_result = previous_action.get('move_result')
#             if move_result:
#                 new_x, new_y = None, None
                
#                 # Handle list format [x, y]
#                 if isinstance(move_result, list) and len(move_result) == 2:
#                     new_x, new_y = move_result
#                 # Handle dict format {x: x, y: y} or {crow_id: id, x: x, y: y}
#                 elif isinstance(move_result, dict):
#                     new_x = move_result.get('x')
#                     new_y = move_result.get('y')
                
#                 # Validate coordinates are numbers
#                 if (new_x is not None and new_y is not None and 
#                     isinstance(new_x, (int, float)) and isinstance(new_y, (int, float))):
#                     game_manager.update_crow_position(game_id, crow_id, int(new_x), int(new_y))
#                     logger.info(f"Updated crow {crow_id} position to ({int(new_x)}, {int(new_y)})")
#                 else:
#                     logger.warning(f"Invalid move result coordinates: {move_result}")
#             else:
#                 logger.warning(f"Invalid move result format: {move_result}")
                
#         elif action_type == 'scan':
#             # Process scan result
#             scan_result = previous_action.get('scan_result')
#             crow_pos = game_manager.get_crow_position(game_id, crow_id)
#             if crow_pos and scan_result and isinstance(scan_result, list):
#                 # Validate scan result is 5x5 grid
#                 if len(scan_result) == 5 and all(isinstance(row, list) and len(row) == 5 for row in scan_result):
#                     game_manager.add_scan_result(game_id, crow_id, crow_pos['x'], crow_pos['y'], scan_result)
#                 else:
#                     logger.warning(f"Invalid scan result format: {scan_result}")
#             else:
#                 logger.warning(f"Invalid scan result or crow position: scan_result={scan_result}, crow_pos={crow_pos}")
                
#         # Check if game is complete or move limit reached
#         game_state = game_manager.games[game_id]
#         if game_manager.is_game_complete(game_id) or game_state['move_count'] >= game_state['max_moves']:
#             # Submit the discovered walls
#             discovered_walls = game_manager.get_discovered_walls(game_id)
#             logger.info(f"Game {game_id} completed: walls={len(discovered_walls)}, moves={game_state['move_count']}, max_moves={game_state['max_moves']}")
#             return jsonify({
#                 'challenger_id': challenger_id,
#                 'game_id': game_id,
#                 'action_type': 'submit',
#                 'submission': discovered_walls
#             })
            
#         # Get next action
#         if game_id not in game_manager.games:
#             return jsonify({'error': 'Game not found'}), 404
            
#         game_state = game_manager.games[game_id]
#         if not game_state:
#             return jsonify({'error': 'Game state is None'}), 500
            
#         crows = game_state.get('crows', {})
#         if not crows:
#             return jsonify({'error': 'No crows available'}), 400
            
#         grid_size = game_state.get('grid_size')
#         if not grid_size:
#             return jsonify({'error': 'Grid size not found'}), 500
            
#         try:
#             explorer = MazeExplorer(grid_size)
#             next_crow, next_action, direction = explorer.get_next_action(game_state, crows)
#         except Exception as e:
#             logger.error(f"Error in get_next_action: {str(e)}")
#             # Fallback: submit what we have
#             discovered_walls = game_manager.get_discovered_walls(game_id)
#             return jsonify({
#                 'challenger_id': challenger_id,
#                 'game_id': game_id,
#                 'action_type': 'submit',
#                 'submission': discovered_walls
#             })
        
#         if not next_crow or not next_action or next_action == 'submit':
#             # No valid actions or submit requested, submit what we have
#             discovered_walls = game_manager.get_discovered_walls(game_id)
#             logger.info(f"Submitting game {game_id} with {len(discovered_walls)} walls found")
#             return jsonify({
#                 'challenger_id': challenger_id,
#                 'game_id': game_id,
#                 'action_type': 'submit',
#                 'submission': discovered_walls
#             })
            
#         if next_action == 'scan':
#             return jsonify({
#                 'challenger_id': challenger_id,
#                 'game_id': game_id,
#                 'crow_id': next_crow,
#                 'action_type': 'scan'
#             })
#         elif next_action == 'move':
#             # Validate direction
#             if direction not in ['N', 'S', 'E', 'W']:
#                 logger.warning(f"Invalid direction: {direction}, submitting instead")
#                 discovered_walls = game_manager.get_discovered_walls(game_id)
#                 return jsonify({
#                     'challenger_id': challenger_id,
#                     'game_id': game_id,
#                     'action_type': 'submit',
#                     'submission': discovered_walls
#                 })
#             return jsonify({
#                 'challenger_id': challenger_id,
#                 'game_id': game_id,
#                 'crow_id': next_crow,
#                 'action_type': 'move',
#                 'direction': direction
#             })
#         else:
#             # Fallback: submit current results
#             discovered_walls = game_manager.get_discovered_walls(game_id)
#             return jsonify({
#                 'challenger_id': challenger_id,
#                 'game_id': game_id,
#                 'action_type': 'submit',
#                 'submission': discovered_walls
#             })
            
#     except Exception as e:
#         logger.error(f"Error in fog_of_wall endpoint: {str(e)}")
#         return jsonify({'error': f'Internal server error: {str(e)}'}), 500

# @app.route('/fog-of-wall/stats/<game_id>', methods=['GET'])
# def get_game_stats(game_id):
#     """Get statistics for a specific game"""
#     stats = game_manager.get_game_stats(game_id)
#     if stats is None:
#         return jsonify({'error': 'Game not found'}), 404
#     return jsonify(stats)

# if __name__ == "__main__":
#     # Only run the app if this file is executed directly
#     app.run(debug=True, port=5001)
