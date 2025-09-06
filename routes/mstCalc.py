import os
import json
import logging
from typing import List, Tuple, Dict, Any
from flask import Flask, request, jsonify
from openai import OpenAI
from dotenv import load_dotenv

import json
import logging

from flask import request

from routes import app

from flask import Flask, request, jsonify


import base64
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
import json
from flask import request, jsonify

class UnionFind:
    """Union-Find data structure for Kruskal's algorithm"""
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return False
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        return True

def decode_image(base64_string):
    """Decode base64 string to OpenCV image"""
    image_data = base64.b64decode(base64_string)
    image = Image.open(BytesIO(image_data))
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

def detect_nodes(image):
    """Detect black circular nodes in the image"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Use HoughCircles to detect circular nodes
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1, minDist=30,
        param1=50, param2=30, minRadius=5, maxRadius=25
    )
    
    nodes = []
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            # Check if the circle is dark (black node)
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.circle(mask, (x, y), r-2, 255, -1)
            mean_intensity = cv2.mean(gray, mask)[0]
            
            if mean_intensity < 100:  # Dark threshold for black nodes
                nodes.append((x, y))
    
    return nodes

def detect_edges_and_weights(image, nodes):
    """Detect edges and extract weights from the image"""
    if len(nodes) < 2:
        return []
    
    edges = []
    
    # Convert to different color spaces for better edge detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Create a mask to exclude node areas
    node_mask = np.ones(gray.shape, dtype=np.uint8) * 255
    for x, y in nodes:
        cv2.circle(node_mask, (x, y), 15, 0, -1)
    
    # Detect edges using Canny edge detection
    edges_img = cv2.Canny(gray, 50, 150)
    edges_img = cv2.bitwise_and(edges_img, node_mask)
    
    # Find contours to identify edge lines
    contours, _ = cv2.findContours(edges_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # For each pair of nodes, check if there's a connection
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            x1, y1 = nodes[i]
            x2, y2 = nodes[j]
            
            # Check if there's a line between these nodes
            if is_connected(edges_img, (x1, y1), (x2, y2)):
                # Extract weight along the edge
                weight = extract_weight_between_nodes(image, (x1, y1), (x2, y2))
                if weight > 0:
                    edges.append((i, j, weight))
    
    return edges

def is_connected(edges_img, node1, node2):
    """Check if two nodes are connected by drawing a line and checking for edge pixels"""
    x1, y1 = node1
    x2, y2 = node2
    
    # Create a line between the nodes
    line_img = np.zeros(edges_img.shape, dtype=np.uint8)
    cv2.line(line_img, (x1, y1), (x2, y2), 255, 3)
    
    # Check for intersection with detected edges
    intersection = cv2.bitwise_and(edges_img, line_img)
    return np.sum(intersection) > 100  # Threshold for connection detection

def extract_weight_between_nodes(image, node1, node2):
    """Extract numerical weight between two nodes using OCR-like approach"""
    x1, y1 = node1
    x2, y2 = node2
    
    # Find midpoint
    mid_x = (x1 + x2) // 2
    mid_y = (y1 + y2) // 2
    
    # Extract region around midpoint
    region_size = 30
    x_start = max(0, mid_x - region_size)
    x_end = min(image.shape[1], mid_x + region_size)
    y_start = max(0, mid_y - region_size)
    y_end = min(image.shape[0], mid_y + region_size)
    
    region = image[y_start:y_end, x_start:x_end]
    
    if region.size == 0:
        return 0
    
    # Convert to grayscale and apply thresholding
    gray_region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    
    # Try multiple threshold values to find text
    for threshold in [127, 100, 150, 80]:
        _, binary = cv2.threshold(gray_region, threshold, 255, cv2.THRESH_BINARY)
        
        # Find contours that might be digits
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by size and shape
        digit_contours = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if 5 < w < 25 and 8 < h < 25:  # Reasonable digit size
                digit_contours.append(contour)
        
        if digit_contours:
            # Simple pattern matching for digits 1-9
            weight = recognize_digits(binary, digit_contours)
            if weight > 0:
                return weight
    
    # Fallback: try to detect based on edge color and position
    return detect_weight_by_color(image, node1, node2)

def recognize_digits(binary_img, contours):
    """Simple digit recognition using template matching patterns"""
    # Sort contours left to right
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
    
    digits = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        digit_roi = binary_img[y:y+h, x:x+w]
        
        if digit_roi.size == 0:
            continue
            
        # Resize to standard size for comparison
        digit_roi = cv2.resize(digit_roi, (12, 16))
        
        # Simple pattern matching for digits 1-9
        digit = match_digit_pattern(digit_roi)
        if digit > 0:
            digits.append(digit)
    
    # Convert digits to number
    if digits:
        return int(''.join(map(str, digits)))
    return 0

def match_digit_pattern(digit_roi):
    """Match digit ROI against simple patterns"""
    # Count white pixels in different regions
    h, w = digit_roi.shape
    total_white = np.sum(digit_roi == 255)
    
    if total_white < 10:
        return 0
    
    # Simple heuristics based on pixel patterns
    top_half = np.sum(digit_roi[:h//2, :] == 255)
    bottom_half = np.sum(digit_roi[h//2:, :] == 255)
    left_half = np.sum(digit_roi[:, :w//2] == 255)
    right_half = np.sum(digit_roi[:, w//2:] == 255)
    
    # Very basic digit recognition
    if total_white < 20:
        return 1
    elif top_half > bottom_half * 1.5:
        return 2
    elif bottom_half > top_half * 1.5:
        return 3
    elif left_half > right_half:
        return 4
    elif right_half > left_half:
        return 5
    elif total_white > 50:
        return 8
    elif total_white > 40:
        return 6
    elif total_white > 30:
        return 7
    else:
        return 9

def detect_weight_by_color(image, node1, node2):
    """Fallback method to detect weights by analyzing colored text along edges"""
    x1, y1 = node1
    x2, y2 = node2
    
    # Sample points along the edge
    num_samples = 10
    weights_found = []
    
    for i in range(1, num_samples):
        t = i / num_samples
        sample_x = int(x1 + t * (x2 - x1))
        sample_y = int(y1 + t * (y2 - y1))
        
        # Extract small region around sample point
        region_size = 15
        x_start = max(0, sample_x - region_size)
        x_end = min(image.shape[1], sample_x + region_size)
        y_start = max(0, sample_y - region_size)
        y_end = min(image.shape[0], sample_y + region_size)
        
        region = image[y_start:y_end, x_start:x_end]
        if region.size == 0:
            continue
        
        # Look for non-black, non-white pixels (colored text)
        gray_region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        colored_mask = (gray_region > 50) & (gray_region < 200)
        
        if np.sum(colored_mask) > 5:
            # Found potential weight, estimate value based on distance and common weights
            distance = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            # Simple heuristic: shorter edges tend to have lower weights
            estimated_weight = max(1, int(distance / 20))
            weights_found.append(min(estimated_weight, 9))
    
    if weights_found:
        return int(np.median(weights_found))
    
    # Final fallback: return a reasonable weight based on edge length
    distance = np.sqrt((x2-x1)**2 + (y2-y1)**2)
    return max(1, min(9, int(distance / 25)))

def kruskal_mst(edges, num_nodes):
    """Calculate MST weight using Kruskal's algorithm"""
    if not edges or num_nodes < 2:
        return 0
    
    # Sort edges by weight
    edges.sort(key=lambda x: x[2])
    
    uf = UnionFind(num_nodes)
    mst_weight = 0
    edges_added = 0
    
    for u, v, weight in edges:
        if uf.union(u, v):
            mst_weight += weight
            edges_added += 1
            if edges_added == num_nodes - 1:
                break
    
    return mst_weight

def process_graph_image(base64_image):
    """Process a single graph image and return MST weight"""
    try:
        # Decode image
        image = decode_image(base64_image)
        
        # Detect nodes
        nodes = detect_nodes(image)
        
        if len(nodes) < 2:
            return 0
        
        # Detect edges and weights
        edges = detect_edges_and_weights(image, nodes)
        
        if not edges:
            return 0
        
        # Calculate MST
        mst_weight = kruskal_mst(edges, len(nodes))
        
        return mst_weight
    
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return 0

@app.route('/mst-calculation', methods=['POST'])
def mst_calculation():
    """Flask endpoint for MST calculation"""
    try:
        # Parse JSON request
        data = request.get_json()
        
        if not data or not isinstance(data, list):
            return jsonify({"error": "Invalid input format"}), 400
        
        results = []
        
        # Process each test case
        for test_case in data:
            if 'image' not in test_case:
                return jsonify({"error": "Missing image field"}), 400
            
            base64_image = test_case['image']
            mst_weight = process_graph_image(base64_image)
            results.append({"value": mst_weight})
        
        return jsonify(results)
    
    except Exception as e:
        return jsonify({"error": f"Processing error: {str(e)}"}), 500




# load_dotenv()

# # --- Config / logging ---
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger("mst-openai")


# OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# client = OpenAI(api_key=OPENAI_API_KEY)


# # --- Kruskal MST ---
# def kruskal_mst(num_nodes: int, edges: List[Tuple[int, int, int]]) -> int:
#     parent = list(range(num_nodes))
#     rank = [0] * num_nodes

#     def find(x: int) -> int:
#         while parent[x] != x:
#             parent[x] = parent[parent[x]]
#             x = parent[x]
#         return x

#     def union(x: int, y: int) -> bool:
#         rx, ry = find(x), find(y)
#         if rx == ry:
#             return False
#         if rank[rx] < rank[ry]:
#             parent[rx] = ry
#         elif rank[rx] > rank[ry]:
#             parent[ry] = rx
#         else:
#             parent[ry] = rx
#             rank[rx] += 1
#         return True

#     total, used = 0, 0
#     for u, v, w in sorted(edges, key=lambda e: e[2]):
#         if 0 <= u < num_nodes and 0 <= v < num_nodes and union(u, v):
#             total += w
#             used += 1
#             if used == num_nodes - 1:
#                 break
#     return total

# # --- Vision system prompt ---
# VISION_SYS_PROMPT = (
#     "You are an expert visual graph reader. The image shows an undirected, connected, weighted graph. "
#     "Black filled circles are nodes. Colored lines are edges. Near the midpoint of each edge there is an integer weight "
#     "drawn (same color as the edge).\n\n"
#     "Assign node IDs deterministically as follows: find black circle centers and sort by (y, then x), ascending "
#     "(top-to-bottom then left-to-right). Number them 0,1,2,... in that order.\n\n"
#     "Return STRICT JSON ONLY with this schema:\n"
#     "{\n"
#     '  "nodes": <int>,\n'
#     '  "edges": [{"u": <int>, "v": <int>, "w": <int>}, ...]\n'
#     "}\n"
#     "No markdown fences, no prose. Weights are positive integers. Edges are undirected (list each once)."
# )

# def call_openai_extract(img_b64: str) -> Dict[str, Any]:
#     """Call OpenAI vision to get nodes + edges JSON."""
#     resp = client.chat.completions.create(
#         model="gpt-4o-mini",  # Using gpt-4o-mini for cheaper/faster runs
#         messages=[
#             {
#                 "role": "system",
#                 "content": VISION_SYS_PROMPT,
#             },
#             {
#                 "role": "user",
#                 "content": [
#                     {"type": "text", "text": "Extract the graph and return ONLY the JSON object."},
#                     {
#                         "type": "image_url",
#                         "image_url": {"url": f"data:image/png;base64,{img_b64}"},
#                     },
#                 ],
#             },
#         ],
#         temperature=0,
#     )
#     text = resp.choices[0].message.content.strip()

#     # Handle possible code fences
#     if text.startswith("```"):
#         text = text.strip("`")
#         if "\n" in text:
#             text = text.split("\n", 1)[1]

#     # Extract JSON
#     start, end = text.find("{"), text.rfind("}")
#     if start == -1 or end == -1 or end <= start:
#         raise ValueError(f"Model did not return JSON: {text[:200]}...")
#     data = json.loads(text[start:end+1])

#     if "nodes" not in data or "edges" not in data:
#         raise ValueError("Missing 'nodes' or 'edges' in model output.")

#     edges = [(int(e["u"]), int(e["v"]), int(e["w"])) for e in data["edges"]]
#     return {"nodes": int(data["nodes"]), "edges": edges}

# @app.route("/mst-calculation", methods=["POST"])
# def mst_calculation():
#     """
#     Input JSON: [{"image": "<base64 png>"}, {"image": "<base64 png>"}]
#     Output JSON: [{"value": int}, {"value": int}]
#     """
#     if not OPENAI_API_KEY:
#         return jsonify({"error": "OPENAI_API_KEY is not set"}), 500

#     cases = request.get_json(force=True)
#     results = []

#     for idx, case in enumerate(cases):
#         img_b64 = (case or {}).get("image", "") or ""
#         if not img_b64:
#             logger.warning(f"Case {idx}: missing image")
#             results.append({"value": 0})
#             continue

#         # ✅ Log the raw base64 string
#         logger.info(f"Case {idx}: received base64 image (length={len(img_b64)})")
#         logger.info(f"Case {idx}: base64 string = {img_b64}")

#         try:
#             parsed = call_openai_extract(img_b64)
#             num_nodes = parsed["nodes"]
#             edges = parsed["edges"]
#             logger.info(f"Case {idx}: nodes={num_nodes}, edges={len(edges)}")
#             value = kruskal_mst(num_nodes, edges)
#         except Exception as e:
#             logger.error(f"Case {idx}: vision parse failed: {e}")
#             value = 0

#         results.append({"value": int(value)})

#     return jsonify(results)



