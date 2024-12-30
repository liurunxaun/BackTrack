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

    query_English = f"""
        Please filter the reasoning paths based on the user question and the given possible reasoning paths.

        User question: {question}
        Possible reasoning paths: {paths}
        Problem intention: {aims}

        Explanation:
        - I have used a large model to extract known conditions from the user question.
        - Starting from these known conditions, I performed a depth-first search in the domain knowledge graph to extract all reasoning paths that start with the labels of these conditions.
        - Each path begins with a condition entity, and the path connects multiple entity labels.
        - Besides, I also extract intentions of the problem. You can utilize them to filter paths.
        
        Filter criteria:
        -Try to filter out paths that are helpful for the answer as much as possible. If the user asks' What could be the problem? ', then the pathways for diseases, medical examinations, and medication should be preserved. A separate pathway to the disease should also be kept.
        -Ensure that the output paths are not duplicated.

        Please return the filtered reasoning paths.
    """

    query_Chinese = f"""
            请根据用户提供的问题和给定的推理路径，筛选出与问题相关的推理路径。

            用户问题: {question}
            待筛选推理路径: {paths}
            问题意图: {aims}

            解释说明：
            - 我已使用大模型从用户问题中提取出已知条件。
            - 从这些已知条件出发，我在领域知识图谱中进行了深度优先搜索，提取出以条件标签为起点的所有推理路径。
            - 每个路径都是从一个条件实体开始的，路径通过多个实体标签相连。
            - 此外，我还提取了问题的意图。您可以使用它们来过滤路径。

            筛选条件：
            - 尽可能筛选出对回答有帮助的路径。如用户问"What could be the problem?"，那么带疾病、医学检查、药物的路径都应该保留。还应该单独保留一条到疾病的路径。
            - 确保输出的路径之间不要重复。
            请返回筛选后的推理路径。
        """

    response = llm.spark_4_0_company(query_Chinese)
    
    # print(f"response:\n{response}")

    pattern = r"\['(.*?)'\]"  # 匹配类似 ['领域', '标题', '方法'] 的结构
    matches = re.findall(pattern, response)

    # 将匹配结果转换为二维列表
    rules = [match.split("', '") for match in matches]
    unique_rules = list(map(list, set(tuple(rule) for rule in rules)))  # 对规则去重
    return unique_rules
