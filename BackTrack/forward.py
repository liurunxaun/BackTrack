# 从上向下，按照倒推得到的推理路径，真正地把具体的条件填入，得到具体推理路径。交给大模型，生成最终答案

from anytree import Node
from anytree.exporter import DotExporter
import random


def dfs_paths(root):
    """深度优先搜索，获取所有路径（排除根节点）。"""
    paths = []

    def dfs(node, path):
        path.append(node)
        if not node.children:  # 到达叶子节点
            paths.append(path[1:])  # 跳过根节点
        for child in node.children:
            dfs(child, path[:])  # 深拷贝路径，避免修改
    dfs(root, [])
    return paths


def format_path(path):
    """将路径中的节点信息格式化为字符串。"""
    formatted_path = []
    for i, node in enumerate(path):
        # 提取节点信息
        node_info = f"[{node.label}] {node.name.split('/')[-1]}"
        # 如果有 parent_edge，并且不是第一个节点，添加边的描述
        if i > 0 and hasattr(node, 'parent_edge'):
            edge_info = f" {node.parent_edge} "
            node_info = edge_info + node_info
        formatted_path.append(node_info)
    return " -> ".join(formatted_path)


def merge_paths(paths):
    """根据条件合并路径的最后一个节点。"""
    merged = {}
    for path in paths:
        first = path[0]  # 第一个节点
        last = path[-1]  # 最后一个节点
        first_key = (first.name, first.label)
        last_label = last.label

        # 合并条件：第一个节点完全相同，最后节点 label 相同
        if first_key in merged and last_label in merged[first_key]:
            merged[first_key][last_label].append(last)
        else:
            merged.setdefault(first_key, {}).setdefault(last_label, []).append(last)
    return merged


def filter_results_by_aims(merged, aims):
    """筛选出符合 aims 条件的结果"""
    valid_labels = {aim[1] for aim in aims}  # 提取 aims 中的所有第二位置
    filtered_results = {}

    for first_key, last_groups in merged.items():
        filtered_last_groups = {}
        for last_label, last_nodes in last_groups.items():
            if last_label in valid_labels:
                filtered_last_groups[last_label] = last_nodes

        if filtered_last_groups:  # 如果有符合条件的分组，保留
            filtered_results[first_key] = filtered_last_groups

    return filtered_results


def display_merged_results(merged):
    """汇总合并过滤后的结果为一个字符串并返回。"""
    result_str = ""
    for first_key, last_groups in merged.items():
        result_str += f"\nConditions: {first_key}\n"
        for last_label, last_nodes in last_groups.items():
            node_names = ", ".join(last_node.name for last_node in last_nodes)
            result_str += f"  Aims: {last_label}\n"
            result_str += f"    Nodes: [{node_names}]\n"
    return result_str


def neo4j_match(source_node, driver, neo4j_database_name, i, path, conditions, top_k):
    i = i + 1
    if i >= len(path):
        return

    source_name = source_node.name
    end_label = path[i]

    neighbor_list = []

    # 看看这次要找的是不是在条件里用作筛选条件
    for condition in conditions:
        relation = ""
        if condition[1] == end_label:
            with driver.session(database=neo4j_database_name) as session:
                if neo4j_database_name == "neo4j" or "chatdoctor5k":
                    print(source_name, condition[0], end_label)
                    result = session.run(
                        """
                        MATCH (n)-[r]-(m)
                        WHERE n.name = $source_name AND m.name = $condition_name AND m.label = $end_label
                        RETURN type(r) AS relation
                        """,
                        source_name=source_name,
                        condition_name=condition[0],
                        end_label=end_label
                    )

                for record in result:
                    relation = record["relation"]  # 通过键名访问值

        if relation != "":
            neighbor_list.append(condition[0])

    sampled_neighbors = neighbor_list


    if relation == '' and len(neighbor_list) == 0:
        # 执行查询
        with driver.session(database = neo4j_database_name) as session:
            # 由于iflytec_nlp的label是属性，所以分开
            if neo4j_database_name == "neo4j"  or "chatdoctor5k": # ifytec_nlp在neo4j数据库使用的是默认数据库名称neo4j
                # print(source_name,end_label)
                result = session.run(
                    """
                    MATCH (n)-[r]-(m)
                    WHERE n.name = $source_name AND m.label = $end_label
                    RETURN collect(m.name) AS neighbors, type(r) AS relation
                    """,
                    source_name=source_name,
                    end_label=end_label
                )

            elif neo4j_database_name == "cardiovascularMini":
                result = session.run(
                    """
                    MATCH (n)-[r]-(m)
                    WHERE n.name = $source_name AND $end_label IN labels(m)
                    RETURN collect(m.name) AS neighbors, type(r) AS relation
                    """,
                    source_name=source_name,
                    end_label=end_label
                )

            neighbor_list = []
            relation = ''

            # 遍历查询结果
            for record in result:
                neighbor_list = record["neighbors"]  # 获取邻居列表
                relation = record["relation"]  # 获取关系名


            # 如果邻居列表超过10条，从中随机选取10条
            # todo:考虑采用什么策略，选择路径。两种考量：1是有可能用户问“还有呢？”。2是选取相关的，用户最关心的
            if len(neighbor_list) > top_k:
                sampled_neighbors = random.sample(neighbor_list, top_k)
            else:
                sampled_neighbors = neighbor_list

    if relation != '' and len(sampled_neighbors) != 0:
        # r = Node("relation:" + relation, parent=source_node)
        # label = Node("label:" + end_label, parent=r)

        for neighbor in sampled_neighbors:
            n = Node(neighbor, parent=source_node, label=path[i], parent_edge=relation)

        for child in source_node.children:
            neo4j_match(child, driver, neo4j_database_name, i, path, conditions, top_k)

    else:
        # todo: 采用其他方式
        return


def forward(paths, conditions, driver, neo4j_database_name, aims, top_k):
    """
    输入：1.倒推得到的路径path[] 2.条件[]
    处理过程：
        连接neo4j数据库
        遍历每一条路径
            try:
                判断第0个元素是哪个条件label，在条件数组中引用具体值
            except:
                知识图谱中这一个不完全
                todo:然后怎么处理，RAG？
    输出：具体推理路径
    """
    last_node_str = ""
    reasoning_path_str = ""

    forward_root = Node("forward_root") # 因为每一个节点的邻居中满足next_label条件的可能有多个，用树的形式来组织更好。

    for path in paths:
        condition_entities = [] # 有可能条件中有多个相同label的实体

        for condition in conditions:
            if path[0] == condition[1]:
                condition_entities.append(condition[0]) # 这一条推理路径的条件实体名
                # todo: 如果找不到呢

        # 先把起始节点加入到结果中
        if len(condition_entities) != 0:
            for condition_entity in condition_entities:
                condition_node = Node(condition_entity, parent=forward_root, label=path[0])
                # Cypher查询
                i = 0
                neo4j_match(condition_node, driver, neo4j_database_name, i, path, conditions, top_k)

    # DotExporter(forward_root).to_picture("./output/reason_tree/forward.png")

    all_paths = dfs_paths(forward_root)  # 深度优先搜索所有路径

    formatted_paths = [format_path(path) for path in all_paths]  # 构建所有路径的描述
    reasoning_path_str = "\n\n".join(
        [f"reasoning path {i + 1}:\n{path}" for i, path in enumerate(formatted_paths)])  # 将路径组合成输入大模型的最终格式

    merged = merge_paths(all_paths)  # 按条件合并路径
    filtered_merged = filter_results_by_aims(merged, aims) # 从输出中筛选出 last Label 符合 aims 的路径
    last_node_str = display_merged_results(filtered_merged)  # 汇总合并过滤后的结果为一个字符串

    return last_node_str, reasoning_path_str


def rules_forward(rules, conditions, driver, neo4j_database_name, top_k):

    last_node_str = ""
    reasoning_path_str = ""

    forward_root = Node("forward_root") # 因为每一个节点的邻居中满足next_label条件的可能有多个，用树的形式来组织更好。

    for path in rules:
        condition_entities = []  # 有可能条件中有多个相同label的实体

        for condition in conditions:
            if path[0] == condition[1]:
                condition_entities.append(condition[0])  # 这一条推理路径的条件实体名
                # todo: 如果找不到呢

        # 先把起始节点加入到结果中
        if len(condition_entities) != 0:
            for condition_entity in condition_entities:
                condition_node = Node(condition_entity, parent=forward_root, label=path[0])
                # Cypher查询
                i = 0
                neo4j_match(condition_node, driver, neo4j_database_name, i, path, conditions, top_k)

    # DotExporter(forward_root).to_picture("./output/reason_tree/forward.png")

    all_paths = dfs_paths(forward_root)  # 深度优先搜索所有路径

    formatted_paths = [format_path(path) for path in all_paths] # 构建所有路径的描述
    reasoning_path_str = "\n\n".join([f"reasoning path {i + 1}:\n{path}" for i, path in enumerate(formatted_paths)]) # 将路径组合成输入大模型的最终格式

    merged = merge_paths(all_paths)  # 按条件合并路径

    aims = []
    for rule in rules:
        aims.append(['', rule[-1]])

    filtered_merged = filter_results_by_aims(merged, aims)  # 从输出中筛选出 last Label 符合 aims 的路径
    last_node_str = display_merged_results(filtered_merged)  # 汇总合并过滤后的结果为一个字符串

    return last_node_str, reasoning_path_str
        