import numpy as np
import random
import game

def print_INFO():
    """
    Prints your homework submission details.
    Please replace the placeholders (date, name, student ID) with valid information
    before submitting.
    """
    print(
        """========================================
        DATE: 2025/03/28
        STUDENT NAME: YOUR NAME
        STUDENT ID: XXXXXXXXX
        ========================================
        """)


#
# Basic search functions: Minimax and Alpha‑Beta
#

def minimax(grid, depth, maximizingPlayer, dep=4):
    """
    TODO (Part 1): Implement recursive Minimax search for Connect Four.

    Return:
      (boardValue, {setOfCandidateMoves})

    Where:
      - boardValue is the evaluated utility of the board state
      - {setOfCandidateMoves} is a set of columns that achieve this boardValue
    """
    # Placeholder return to keep function structure intact
    if depth == 0 or grid.terminate():
        return get_heuristic(grid), None
    
    if maximizingPlayer:
        cur_max = -np.inf
        cur_best_move = set()
        for col in grid.valid:
            nxt = game.drop_piece(grid, col)
            score, _ = minimax(nxt, depth - 1, False)
            if score > cur_max:
                cur_max = score
                cur_best_move = {col}
            elif score == cur_max:
                cur_best_move.add(col)
        return cur_max, cur_best_move
    else:
        cur_min = np.inf
        cur_worst_move = set()
        for col in grid.valid:
            nxt = game.drop_piece(grid, col)
            score, _ = minimax(nxt, depth - 1, True)
            if score < cur_min:
                cur_min = score
                cur_worst_move = {col}
            elif score == cur_min:
                cur_worst_move.add(col)
        return cur_min, cur_worst_move


def alphabeta(grid, depth, maximizingPlayer, alpha, beta, dep=4):
    """
    TODO (Part 2): Implement Alpha-Beta pruning as an optimization to Minimax.

    Return:
      (boardValue, {setOfCandidateMoves})

    Where:
      - boardValue is the evaluated utility of the board state
      - {setOfCandidateMoves} is a set of columns that achieve this boardValue
      - Prune branches when alpha >= beta
    """
    # Placeholder return to keep function structure intact
    if depth == 0 or grid.terminate():
        return get_heuristic(grid), None
    
    if maximizingPlayer:
        cur_max = -np.inf
        cur_best_move = set()
        for col in grid.valid:
            nxt = game.drop_piece(grid, col)
            score, _ = alphabeta(nxt, depth - 1, False, alpha, beta)
            if score > cur_max:
                cur_max = score
                cur_best_move = {col}
            elif score == cur_max:
                cur_best_move.add(col)
            alpha = max(alpha, cur_max)
            if beta <= alpha:
                break
        return cur_max, cur_best_move
    else:
        cur_min = np.inf
        cur_worst_move = set()
        for col in grid.valid:
            nxt = game.drop_piece(grid, col)
            score, _ = alphabeta(nxt, depth - 1, True, alpha, beta)
            if score < cur_min:
                cur_min = score
                cur_worst_move = {col}
            elif score == cur_min:
                cur_worst_move.add(col)
            beta = min(beta, cur_min)
            if beta <= alpha:
                break
        return cur_min, cur_worst_move


#
# Basic agents
#

def agent_minimax(grid):
    """
    Agent that uses the minimax() function with a default search depth (e.g., 4).
    Must return a single column (integer) where the piece is dropped.
    """
    return random.choice(list(minimax(grid, 4, True)[1]))


def agent_alphabeta(grid):
    """
    Agent that uses the alphabeta() function with a default search depth (e.g., 4).
    Must return a single column (integer) where the piece is dropped.
    """
    return random.choice(list(alphabeta(grid, 4, True, -np.inf, np.inf)[1]))


def agent_reflex(grid):
    """
    A simple reflex agent provided as a baseline:
      - Checks if there's an immediate winning move.
      - Otherwise picks a random valid column.
    """
    wins = [c for c in grid.valid if game.check_winning_move(grid, c, grid.mark)]
    if wins:
        return random.choice(wins)
    return random.choice(grid.valid)


def agent_strong(grid):
    """
    TODO (Part 3): Design your own agent (depth = 4) to consistently beat the Alpha-Beta agent (depth = 4).
    This agent will typically act as Player 2.
    """
    # Placeholder logic that calls your_function().
    return random.choice(list(your_function(grid, 4, 1, -np.inf, np.inf)[1]))


#
# Heuristic functions
#

# 以 user1 的角度
def get_heuristic(board):
    """
    Evaluates the board from Player 1's perspective using a basic heuristic.

    Returns:
      - Large positive value if Player 1 is winning
      - Large negative value if Player 2 is winning
      - Intermediate scores based on partial connect patterns
    """
    num_twos       = game.count_windows(board, 2, 1)
    num_threes     = game.count_windows(board, 3, 1)
    num_twos_opp   = game.count_windows(board, 2, 2)
    num_threes_opp = game.count_windows(board, 3, 2)

    score = (
          1e10 * board.win(1)
        + 1e6  * num_threes
        + 10   * num_twos
        - 10   * num_twos_opp
        - 1e6  * num_threes_opp
        - 1e10 * board.win(2)
    )
    return score


# 改以 user2 的角度
def get_heuristic_strong(board, player):
    """
    TODO (Part 3): Implement a more advanced board evaluation for agent_strong.
    Currently a placeholder that returns 0.
    """
    opponent = 3 - player  

    num_twos       = game.count_windows(board, 2, player)
    num_threes     = game.count_windows(board, 3, player)
    num_twos_opp   = game.count_windows(board, 2, opponent)
    num_threes_opp = game.count_windows(board, 3, opponent)

    score = (
        1e10 * board.win(player)       # 立即勝利
      - 1e11 * board.win(opponent)     # 阻止對方勝利
      + 1e6  * num_threes              # 快速 3 連線
      - 1e7  * num_threes_opp          # 阻止對方3連線
      + 10   * num_twos                # 2 連線
      - 20   * num_twos_opp            # 阻止對方2連線
    )

    # 提高選擇正中間欄位的獎勵，逐步遞減
    column_weights = [3, 4, 5, 7, 5, 4, 3]  

    for r in range(6):  # 行
        for c in range(7):  # 列
            if board.table[r, c] == player:
                score += column_weights[c] + (6-r) # 獎勵越高的行，獎勵越高(因為底部的棋子優勢更大)
    return score


def your_function(grid, depth, maximizingPlayer, alpha, beta, dep=4):
    """
    A stronger search function that uses get_heuristic_strong() instead of get_heuristic().
    You can employ advanced features (e.g., improved move ordering, deeper lookahead).

    Return:
      (boardValue, {setOfCandidateMoves})

    Currently a placeholder returning (0, {0}).
    """
    player = 2  
    opponent = 1

    # 提早偵測必勝或必敗的情況
    if grid.win(player):
        return 1e10 + depth * 100, None
    if grid.win(opponent):
        return -1e10, None
    
    if depth == 0 or grid.terminate():
        return get_heuristic_strong(grid, player), None

    # Move ordering: 評估所有合法棋子的分數，排序後探索
    def order_moves(grid):
        scored_cols = []
        center_column = 3
        if center_column in grid.valid:
            nxt = game.drop_piece(grid, center_column)
            score = get_heuristic_strong(nxt, player)
            scored_cols.append((score, center_column))

        for col in grid.valid:
            if col == center_column:
                continue
            nxt = game.drop_piece(grid, col)
            score = get_heuristic_strong(nxt, player)
            scored_cols.append((score, col))

        scored_cols.sort(reverse=maximizingPlayer)  # 1 -> 高分優先, 0 -> 低分優先
        return [col for _, col in scored_cols]
    
    ordered_cols = order_moves(grid)

    if maximizingPlayer:  
        cur_max = -np.inf
        cur_best_move = set()
        for col in ordered_cols:
            nxt = game.drop_piece(grid, col)
            score, _ = your_function(nxt, depth - 1, False, alpha, beta)
            if score > cur_max:
                cur_max = score
                cur_best_move = {col}
            elif score == cur_max:
                cur_best_move.add(col)
            alpha = max(alpha, cur_max)
            if beta <= alpha:
                break
        return cur_max, cur_best_move

    else:  
        cur_min = np.inf
        cur_worst_move = set()
        for col in ordered_cols:
            nxt = game.drop_piece(grid, col)
            score, _ = your_function(nxt, depth - 1, True, alpha, beta)
            if score < cur_min:
                cur_min = score
                cur_worst_move = {col}
            elif score == cur_min:
                cur_worst_move.add(col)
            beta = min(beta, cur_min)
            if beta <= alpha:
                break
        return cur_min, cur_worst_move
