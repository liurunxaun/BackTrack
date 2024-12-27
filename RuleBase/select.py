# 不采用倒推的形式，转而从由条件构建的树经过去重后，不经过目的的硬性筛选，而是直接交给大模型，让他结合问题，选择留下哪些本体路径。
# 实际上倒推是“规则”的进阶版，“规则”是让大模型直接选路径，倒推是先让大模型从问题中提取出意图来，再硬性筛选
# 可以说是两种筛选路径的方式
# 但是把问题和所有待选路径一并直接交给大模型，或许会参考信息更多，效果更好

from utils.LLM import spark as llm
import re


def select_rules(paths, question, aims):
    """
    输入：全部条件的全部本体路径，用户的问题
    处理过程：一并交给大模型，筛选出回答问题有用的本体路径
    输出：筛选出来的路径（或者说规则）
    """

    rules = []

    query = f"""
        请根据用户提供的问题和给定的推理路径，筛选出与问题相关的推理路径。对于每个路径，只需要提取出相关部分，路径中不相关的部分可以省略。

        用户问题: {question}
        待筛选推理路径: {paths}

        解释说明：
        - 我已使用大模型从用户问题中提取出已知条件。
        - 从这些已知条件出发，我在领域知识图谱中进行了深度优先搜索，提取出以条件标签为起点的所有推理路径。
        - 每个路径都是从一个条件实体开始的，路径通过多个实体标签相连。

        请根据以下标准筛选路径：
        - 仅保留与问题相关的路径部分。例如，如果问题问的是“这篇论文使用了什么方法？”，则路径“标题->方法->领域”应该简化为“标题->方法”，因为领域信息不相关。
        - 输出路径时只包含必要的部分，去除冗余的部分。
        - 你可以选择截断路径，只保留前面部分。

        请返回筛选后的推理路径。
    """

    query_English_wide_and_aim = f"""
        Please filter the reasoning paths based on the user question and the given possible reasoning paths. For each path, only the relevant parts need to be included, and irrelevant parts can be omitted.

        User question: {question}
        Possible reasoning paths: {paths}
        Problem intention: {aims}

        Explanation:
        - I have used a large model to extract known conditions from the user question.
        - Starting from these known conditions, I performed a depth-first search in the domain knowledge graph to extract all reasoning paths that start with the labels of these conditions.
        - Each path begins with a condition entity, and the path connects multiple entity labels.
        - Besides, I also extract intentions of the problem. You can utilize them to filter paths.
        
        

        Please return the filtered reasoning paths.
    """

    query_English = f"""
        Please filter the reasoning paths based on the user question and the given possible reasoning paths. For each path, only the relevant parts need to be included, and irrelevant parts can be omitted.

        User question: {question}
        Possible reasoning paths: {paths}

        Explanation:
        - I have used a large model to extract known conditions from the user question.
        - Starting from these known conditions, I performed a depth-first search in the domain knowledge graph to extract all reasoning paths that start with the labels of these conditions.
        - Each path begins with a condition entity, and the path connects multiple entity labels.

        Please filter the paths according to the following criteria:
        - Only keep the relevant parts of the path related to the question. For example, if the question asks “What method does this paper use?”, then the path “[标题,方法,领域]” should be simplified to “[标题,方法]”, as the 领域 information is not relevant.
        - When returning the path, only include the necessary parts and omit redundant parts.
        - You can truncate the path and only keep the earlier parts.

        Please return the filtered reasoning paths.
    """

    response = llm.spark_4_0(query_English_wide_and_aim)
    
    print(f"response:\n{response}")

    pattern = r"\['(.*?)'\]"  # 匹配类似 ['领域', '标题', '方法'] 的结构
    matches = re.findall(pattern, response)

    # 将匹配结果转换为二维列表
    rules = [match.split("', '") for match in matches]

    return rules
