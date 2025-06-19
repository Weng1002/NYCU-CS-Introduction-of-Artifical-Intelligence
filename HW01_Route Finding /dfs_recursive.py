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

def dfs_helper(start, end, graph, visited, path, current_dist):
    """
    追踪路徑和距離。
    """
    visited.add(start)
    num_visited = len(visited) - 1  # 不包括起點的拜訪數量

    if start == end:
        return path, current_dist, num_visited  # 找到終點，返回路徑、總距離和拜訪數

    for neighbor, distance in graph[start]:
        if neighbor not in visited:
            new_path = path + [neighbor]
            new_dist = current_dist + distance
            result = dfs_helper(neighbor, end, graph, visited, new_path, new_dist)
            if result[0]:  # 如果找到路徑
                return result
    
    return [], 0.0, num_visited  # 如果找不到路徑，返回空路徑、距離 = 0 和 0 個拜訪節點


def dfs(start, end):
    """
    返回 (path, dist, num_visited)
    """
    graph = load_graph(edgeFile)
    
    if start not in graph or end not in graph:
        return [], 0.0, 0  # 如果起點或終點不存在，返回空路徑、距離 = 0 和 0 個拜訪節點

    visited = set() 
    path = [start]  
    current_dist = 0.0  

    # 呼叫遞迴輔助函數
    result = dfs_helper(start, end, graph, visited, path, current_dist)
    return result


if __name__ == '__main__':
    path, dist, num_visited = dfs(2270143902, 1079387396)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total distance of path: {dist}')
    print(f'The number of visited nodes: {num_visited}')
