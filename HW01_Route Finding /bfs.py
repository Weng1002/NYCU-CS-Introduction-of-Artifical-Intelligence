import csv
from collections import defaultdict, deque

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


def bfs(start, end):
    """
    返回 (path, dist, num_visited)
    """
    graph = load_graph(edgeFile)
    
    if start not in graph or end not in graph:
        print(f"起點 {start} 或終點 {end} 不在圖中")
        return [], 0.0, 0  # 如果起點或終點不存在，返回空路徑、距離 = 0 和 0 個拜訪節點

    queue = deque([(start, [start], 0.0)])  # (當前節點, 路徑, 總距離)
    visited = set()  # 記錄拜訪的節點
    num_visited = 0  # 拜訪的節點數量


    while queue:
        current_node, path, current_dist = queue.popleft()
        
        if current_node in visited:
            continue
        
        visited.add(current_node)
        num_visited += 1  
        
        
        if current_node == end:
            num_visited -= 1
            return path, current_dist, num_visited  # 找到終點
        
        # 看鄰居節點（僅考慮出邊，因為是有向圖）
        for neighbor, distance in graph[current_node]:
            if neighbor not in visited:
                new_path = path + [neighbor]
                new_dist = current_dist + distance
                queue.append((neighbor, new_path, new_dist))

    return [], 0.0, num_visited  # 如果找不到路徑，返回空路徑、距離 = 0 和 0 個拜訪節點


if __name__ == '__main__':
    path, dist, num_visited = bfs(2270143902, 1079387396)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total distance of path: {dist}')
    print(f'The number of visited nodes: {num_visited}')
