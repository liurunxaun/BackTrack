from BackTrack import back
from anytree import Node

def collect_paths(conditions,  max_pop, label_dict):
    """
    从用户问题中提取条件后，收集以条件为起点的全部路径，并返回
    输入：条件[[entity, label],...]，最大跳数，标签字典
    处理过程：
        构建树root
        将所有条件加入
        upFind
        剪枝去除回路
        深度优先搜索输出全部路径
    输出：全部路径
    """
    paths = []

    # 1. 构造路径树
    root = Node("root")
    for condition in conditions:
        if condition[1] != 'none':
            Node(condition[1], parent=root)
    for child in root.children:
        # print(child.name)
        pop = 1
        back.upFind(child, pop, max_pop, label_dict)
    # 使用 DotExporter 输出树的结构
    # DotExporter(root).to_picture("./output/rule_base/tree.png")

    # 2. 然后进行剪枝，深度优先搜索每条路径，去除回路
    back.prune_tree_repeat(root, set())
    # 使用 DotExporter 输出树的结构
    # DotExporter(root).to_picture("./output/reason_tree/prune_tree_repeat.png")

    # 3. 深度优先搜索输出全部路径
    back.dfs_paths(root, [], paths)

    # 4. 去除root节点
    for i in range(len(paths)):
        # 移除 root 并反转路径
        if paths[i][0] == "root":
            paths[i] = paths[i][1:]  # 去掉 root 节点
    unique_paths = list(map(list, set(tuple(path) for path in paths)))  # 对子列表去重
    return unique_paths