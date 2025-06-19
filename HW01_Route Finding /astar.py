import csv
import heapq
from collections import defaultdict

edgeFile = 'edges.csv'
heuristicFile = 'heuristic_values.csv'

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

def load_heuristics(filename, goal):
    """
    讀取 heuristic.csv 並建立啟發式函數的字典。
    """
    heuristics = {}
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)  # 讀取標頭
        goal_index = headers.index(str(goal))  # 找到對應的目標列索引

        for row in reader:
            node = int(row[0])
            heuristics[node] = float(row[goal_index])  # 記錄該節點的啟發式值
    return heuristics


def astar(start, end):
    """"
    返回: (最短路徑, 總距離, 拜訪節點數)
    """
    graph = load_graph(edgeFile)
    heuristics = load_heuristics(heuristicFile, end)

    if start not in graph or end not in graph:
        return [], 0.0, 0  # 如果起點或終點不存在，返回空路徑、距離 = 0 和 0 個拜訪節點

    # 優先隊列 (f(n), g(n), 當前節點, 路徑)
    queue = [(heuristics.get(start, 0.0), 0.0, start, [start])]
    heapq.heapify(queue)

    visited = set()  # 記錄已擴展的節點
    dist_dict = defaultdict(lambda: float('inf'))  # 記錄 g(n)（起點到當前節點的實際距離）
    dist_dict[start] = 0.0
    num_visited = 0 

    while queue:
        _, current_dist, current_node, path = heapq.heappop(queue)
        num_visited += 1  

        if current_node in visited:
            continue

        visited.add(current_node)

        if current_node == end:
            num_visited -= 1
            return path, current_dist, num_visited  # 找到目標節點，返回結果

        for neighbor, distance in graph[current_node]:
            new_dist = current_dist + distance
            if new_dist < dist_dict[neighbor]:  # 只有找到更短距離才更新
                dist_dict[neighbor] = new_dist
                f_value = new_dist + heuristics.get(neighbor, 0.0)  # f(n) = g(n) + h(n)
                new_path = path + [neighbor]
                heapq.heappush(queue, (f_value, new_dist, neighbor, new_path))

    return [], 0.0, num_visited  # 如果找不到路徑，返回空路徑、距離 = 0 和 0 個拜訪節點


if __name__ == '__main__':
    path, dist, num_visited = astar(2270143902, 1079387396)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total distance of path: {dist}')
    print(f'The number of visited nodes: {num_visited}')
