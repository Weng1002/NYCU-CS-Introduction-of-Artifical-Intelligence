import csv
import heapq  
from collections import defaultdict

edgeFile = 'edges.csv'

def load_graph(filename):
    """
    讀取 edges.csv 檔案。
    """
    graph = defaultdict(list)  # 使用 defaultdict 儲存圖，格式為 {node_id: [(neighbor_id, distance), ...]}
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            start_node, end_node, distance = row[0], row[1], float(row[2])  
            graph[int(start_node)].append((int(end_node), distance))

    return graph

def ucs(start, end):
    """
    返回 (path, dist, num_visited)
    """
    graph = load_graph(edgeFile)
    
    if start not in graph or end not in graph:
        return [], 0.0, 0  # 如果起點或終點不存在，返回空路徑、距離 = 0 和 0 個拜訪節點
    
    dist_dict = defaultdict(lambda: float('inf'))
    dist_dict[start] = 0.0

    # 使用優先隊列（heap），根據總距離排序
    queue = [(0.0, start, [start])]  # (總距離, 當前節點, 路徑)
    heapq.heapify(queue)

    visited = set()  
    dist_dict = defaultdict(lambda: float('inf'))  # 紀錄最短距離
    dist_dict[start] = 0.0
    num_visited = 0  
    

    while queue:
        current_dist, current_node, path = heapq.heappop(queue)
        num_visited += 1
        
        if current_node in visited:
            continue
        
        visited.add(current_node)
        
        if current_node == end:
            num_visited -= 1
            return path, current_dist, num_visited  # 找到終點，返回路徑、總距離和拜訪數
        
        # 探索鄰居節點（僅考慮出邊，因為是有向圖）
        for neighbor, distance in graph[current_node]:
            new_dist = current_dist + distance
            if new_dist < dist_dict[neighbor]:  # 只有更短距離才更新
                dist_dict[neighbor] = new_dist
                new_path = path + [neighbor]
                heapq.heappush(queue, (new_dist, neighbor, new_path))

    return [], 0.0, num_visited  # 如果找不到路徑，返回空路徑、距離 = 0 和 0 個拜訪節點


if __name__ == '__main__':
    path, dist, num_visited = ucs(2270143902, 1079387396)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total distance of path: {dist}')
    print(f'The number of visited nodes: {num_visited}')
