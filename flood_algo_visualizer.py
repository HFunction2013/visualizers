#!/usr/bin/env python3
"""
终端迷宫洪水填充算法演示器
  红色 #  = 墙
  空格    = 未探索的路
  蓝色 %  = 洪水
  绿色 %  = 到达终点后标出的最短路径
用法:
  python3 maze_flood.py [高度] [宽度] [速度]
  速度单位为秒/步，传 -1 表示极速无延迟。
  只传一个数字时视为正方形迷宫。
"""

import sys
import time
import random
from collections import deque

# ── ANSI 转义序列 ──────────────────────────────────────────
RED   = '\033[91m'
BLUE  = '\033[94m'
GREEN = '\033[92m'
RESET = '\033[0m'
HOME    = '\033[H'       # 光标归位 (0,0)，不清屏
CLR_EOL = '\033[K'       # 清除到行尾（标题长度变化时防残留）
HIDE    = '\033[?25l'
SHOW    = '\033[?25h'


# ── 迷宫生成（迭代式递归回溯 / 完美迷宫）──────────────────
def generate_maze(rows, cols, seed=None):
    if seed is not None:
        random.seed(seed)
    maze = [['#'] * cols for _ in range(rows)]
    stack = [(1, 1)]
    maze[1][1] = ' '
    while stack:
        r, c = stack[-1]
        dirs = [(0, 2), (0, -2), (2, 0), (-2, 0)]
        random.shuffle(dirs)
        moved = False
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if 1 <= nr < rows - 1 and 1 <= nc < cols - 1 and maze[nr][nc] == '#':
                maze[r + dr // 2][c + dc // 2] = ' '
                maze[nr][nc] = ' '
                stack.append((nr, nc))
                moved = True
                break
        if not moved:
            stack.pop()
    return maze


# ── 渲染 ────────────────────────────────────────────────────
def render(maze, water, path, start, end, step):
    rows, cols = len(maze), len(maze[0])
    lines = []
    for r in range(rows):
        row = []
        for c in range(cols):
            p = (r, c)
            if p == start:
                row.append(GREEN + 'S' + RESET)
            elif p == end:
                row.append(GREEN + 'E' + RESET)
            elif p in path:
                row.append(GREEN + '%' + RESET)
            elif p in water:
                row.append(BLUE + '%' + RESET)
            elif maze[r][c] == '#':
                row.append(RED + '#' + RESET)
            else:
                row.append(' ')
        lines.append(''.join(row))
    header = (f"  洪水填充演示  |  波次: {step}  |  "
              f"已淹没: {len(water)} 格  |  起点 S → 终点 E")
    print(HOME + HIDE + header + CLR_EOL + '\n' + '\n'.join(lines))


# ── BFS 洪水填充 ────────────────────────────────────────────
def flood_fill(maze, start, end, speed):
    rows, cols = len(maze), len(maze[0])
    visited = {start}
    parent = {}
    queue = deque([start])
    water = set()
    step = 0
    found = False

    render(maze, water, set(), start, end, step)
    if speed >= 0:
        time.sleep(speed)

    while queue and not found:
        step += 1
        for _ in range(len(queue)):
            r, c = queue.popleft()
            if (r, c) == end:
                found = True
                break
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if (0 <= nr < rows and 0 <= nc < cols
                        and maze[nr][nc] == ' '
                        and (nr, nc) not in visited):
                    visited.add((nr, nc))
                    parent[(nr, nc)] = (r, c)
                    queue.append((nr, nc))
                    water.add((nr, nc))
        render(maze, water, set(), start, end, step)
        if speed >= 0:
            time.sleep(speed)

    # 回溯最短路径
    path = set()
    if found:
        curr = end
        while curr != start:
            path.add(curr)
            curr = parent[curr]
        path.add(start)

    render(maze, water, path, start, end, step)
    print(SHOW)
    return path, found


# ── 主入口 ───────────────────────────────────────────────────
def main():
    args = sys.argv[1:]

    # 解析尺寸
    if len(args) >= 2:
        rows, cols = int(args[0]), int(args[1])
    elif len(args) == 1:
        rows = cols = int(args[0])
    else:
        try:
            raw = input("请输入迷宫大小（如 21 或 21 41）: ").strip()
            parts = raw.split()
            if len(parts) == 1:
                rows = cols = int(parts[0])
            else:
                rows, cols = int(parts[0]), int(parts[1])
        except (EOFError, ValueError):
            rows, cols = 21, 41

    # 解析速度
    if len(args) >= 3:
        speed = float(args[2])
    else:
        try:
            raw = input("水流速度（秒/步，-1 = 极速）: ").strip()
            speed = float(raw) if raw else 0.05
        except (EOFError, ValueError):
            speed = 0.05

    # 最小尺寸 & 奇数修正
    rows = max(5, rows)
    cols = max(5, cols)
    if rows % 2 == 0:
        rows += 1
    if cols % 2 == 0:
        cols += 1

    print(f"正在生成 {rows}x{cols} 迷宫...")
    maze = generate_maze(rows, cols)
    start = (1, 1)
    end = (rows - 2, cols - 2)

    try:
        path, found = flood_fill(maze, start, end, speed)
    except KeyboardInterrupt:
        print(SHOW + "\n已中断。")
        return

    if found:
        print(f"\n到达终点！最短路径长度: {len(path)} 格")
    else:
        print("\n无法到达终点。")
    print(f"迷宫: {rows}x{cols}  |  速度: {'极速(-1)' if speed < 0 else f'{speed}s/步'}")


if __name__ == '__main__':
    main()
