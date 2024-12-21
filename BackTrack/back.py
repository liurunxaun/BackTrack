from anytree import Node
from anytree.exporter import DotExporter

# 读取txt文件并构建字典
def build_label_dict(txt_file):
    label_dict = {}

    # 打开txt文件读取
    with open(txt_file, 'r', encoding='utf-8') as file:
        for line in file:
            # 去掉每行的前后空白字符
            line = line.strip()
            if not line:
                continue

            # 按制表符分割每行，忽略关系
            parts = line.split('-')
            if len(parts) < 3:
                continue  # 如果格式不对，跳过该行

            # 提取实体，去除label前缀
            label1 = parts[0]
            label2 = parts[2]

            # 将实体添加到字典
            if label1 not in label_dict:
                label_dict[label1] = set()
            if label2 not in label_dict:
                label_dict[label2] = set()

            # 添加相关实体
            label_dict[label1].add(label2)
            label_dict[label2].add(label1)

    return label_dict


def upFind(node, pop, max_pop, label_dict):
    """
    输入：当前被找的节点，最大向上阶数，标签邻接表
    处理过程：
        如果达到上限，
            return None
        else:
            到表中查这一节点的所有相关节点，作为子节点，加入树中
    输出：None（找到的路径直接存储在树中了）
    """
    # 第1个返回条件：检查是否到达上溯层数限制
    pop += 1
    if pop >= max_pop:
        return

    # 在本体关系表中查找语义相关实体
    related_labels = label_dict.get(node.name, set())

    for label in related_labels:
        Node(label, parent=node)
    for child in node.children:
        upFind(child, pop, max_pop, label_dict)


def prune_tree_repeat(node, visited):
    """
        深度优先搜索 (DFS) 剪枝树。
        遍历节点，移除重复路径中的节点。

        :param node: 当前树的节点
        :param visited: 已访问节点集合
    """
    # 如果当前节点已访问过，则移除节点
    if node.name in visited:
        node.parent = None  # 将当前节点与其父节点断开
        return

    # 将当前节点标记为已访问
    visited.add(node.name)

    # 遍历子节点并递归调用
    for child in list(node.children):  # 使用 list 防止动态修改影响循环
        prune_tree_repeat(child, visited)

    # 回溯时移除当前节点的访问记录
    visited.remove(node.name)


def prune_tree_by_conditions(node, condition_labels, path=None):
    """
    对树结构进行基于条件的路径剪枝。
    1. 如果路径中没有出现 conditions 的节点，删除整个路径。
    2. 如果路径中出现了 conditions 的节点，只保留路径中最后一个条件节点及之前的部分。

    :param node: 当前树的根节点
    :param condition_labels: 条件数组
    :param path: 当前递归访问的路径（从根到当前节点）
    """

    # 初始化路径
    if path is None:
        path = []

    # 将当前节点加入路径
    path.append(node)

    # 如果是叶子节点，开始剪枝逻辑
    if not node.children:
        # 获取路径中所有节点的名称
        path_names = [n.name for n in path]

        # 检查路径是否包含条件中的元素
        last_condition_idx = -1
        for i, name in enumerate(path_names):
            if name in condition_labels:
                last_condition_idx = i

        if last_condition_idx == -1:
            # 如果路径中不包含任何条件元素，删除该路径
            node.parent = None
        else:
            # 如果路径中包含条件元素，保留到最后一个条件节点的部分，删除后续部分
            # 找到最后一个条件节点，并删除其后的所有分支
            for n in path[last_condition_idx + 1:]:
                n.parent = None

    # 递归处理子节点
    for child in list(node.children):
        prune_tree_by_conditions(child, condition_labels, path[:])  # 使用 path[:] 传递路径副本，防止路径污染


def dfs_paths(node, current_path, all_paths):
    """
    深度优先搜索遍历树，并将每条路径存储到 all_paths 中。

    :param node: 当前树的节点
    :param current_path: 当前路径（一个列表）
    :param all_paths: 所有路径（一个列表，每条路径也是一个列表）
    """
    # 将当前节点加入路径
    current_path.append(node.name)

    # 如果是叶子节点，将路径加入到 all_paths 中
    if not node.children:
        all_paths.append(list(current_path))  # 保存当前路径的副本
    else:
        # 否则递归处理每个子节点
        for child in node.children:
            dfs_paths(child, current_path, all_paths)

    # 回溯：移除当前节点
    current_path.pop()


def prune_paths_by_conditions(all_paths, conditions):
    """
    根据条件数组剪枝路径。

    :param all_paths: 所有路径（每条路径是一个列表）
    :param conditions: 条件数组（必须包含的节点）
    :return: 剪枝后的路径列表
    """
    pruned_paths = []

    for path in all_paths:
        # 找到路径中最后一个条件节点的索引
        condition_indices = [i for i, node in enumerate(path) if node in conditions]

        if condition_indices:
            # 如果路径中包含条件节点，保留从根到最后一个条件节点的子路径
            last_condition_index = condition_indices[-1]
            pruned_paths.append(path[:last_condition_index + 1])

    return pruned_paths


def reverse_and_remove_root(paths):
    """
    反转路径并移除 root 节点。

    :param paths: 剪枝后的路径列表
    :return: 处理后的路径列表
    """
    processed_paths = []
    for path in paths:
        # 移除 root 并反转路径
        if path[0] == "root":
            path = path[1:]  # 去掉 root 节点
        processed_paths.append(list(reversed(path)))  # 反转路径
    return processed_paths


def aim_back(conditions, aims, max_pop, label_dict):
    """
    倒退找推理路径
    输入：条件[[entity, label]]，目的[[entity, label]]
    处理过程：
        对于每一个目的label，
            函数递归调用：
                输入：当前路径，向上阶数
                如果达到上限，return.else:
                到表中查找出现在同一个关系的另外一个label
                如果找到的label在条件内，将label append到当前路径内，并将当前路径添加到path内
                输出：none。（找到的路径直接存储在path中了）
    输出：找到的所有推理路径
    """
    final_paths = []

    # 1. 构造路径树，第0层1个节点是root，第1层是所有目的实体类型，然后调用upFind向上寻找路径
    root = Node("root")
    for aim in aims:
        if aim[1] != 'none':
            Node(aim[1], parent=root)
    for child in root.children:
        # print(child.name)
        pop = 1
        upFind(child, pop, max_pop, label_dict)
    # 使用 DotExporter 输出树的结构
    # DotExporter(root).to_picture("./output/reason_tree/tree.png")
    
    # 2. 然后进行剪枝，深度优先搜索每条路径，去除回路
    prune_tree_repeat(root, set())
    # 使用 DotExporter 输出树的结构
    # DotExporter(root).to_picture("./output/reason_tree/prune_tree_repeat.png")

    # 3. 执行基于条件的路径剪枝
    condition_labels = []
    for condition in conditions:
        condition_labels.append(condition[1])
    prune_tree_by_conditions(root, condition_labels)
    # 输出剪枝后的树
    # DotExporter(root).to_picture("./output/reason_tree/prune_tree_conditions.png")

    # 4. 生成所有深度优先路径
    all_paths = []
    dfs_paths(root, [], all_paths)

    # 5. 去除最后一个节点不在条件中的路径
    pruned_paths = prune_paths_by_conditions(all_paths, condition_labels)

    # 6. 反转路径并去掉 root 节点
    final_paths = reverse_and_remove_root(pruned_paths)

    return final_paths