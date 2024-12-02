from BackTrack import extract as extract
from BackTrack import back as back
from BackTrack import forward as forward
from BackTrack import answer as answer
from neo4j import GraphDatabase
import time


def main(question, max_pop, label_dict, driver, model):
    print(f"\n问题:{question}")

    # 1. 从问题中提取条件实体、目的实体、实体类型
    print("\n======1. 从问题中提取条件实体、目的实体、实体类型======")
    conditions, aims = extract.extract(question)

    if len(conditions) != 0 and len(aims) != 0:
        print(f"conditions:{conditions}")
        print(f"aims:{aims}")
    else:
        print("问题中不包含知识图谱范围内的条件和目的")
        print("\n======2. 调用大模型生成最终答案======")
        generation = answer.generate_answer(question)
        return generation


    # 2. 倒推找抽象本体推理路径
    print("\n======2.倒推找抽象本体推理路径======")
    paths = back.aim_back(conditions, aims, max_pop, label_dict)

    if len(paths) != 0:
        print("抽象本体推理路径:")
        for path in paths:
            print(" -> ".join(path))
    else:
        print("没找到抽象本体推理路径。请尝试换个说法，或者描述的更详细一些")
        print("\n======3. 调用大模型生成最终答案======")
        generation = answer.generate_answer(question)
        return generation


    # 3. 正推找具体实体推理路径，筛选得到条件和目的实体
    print("\n======3. 正推具体实体推理路径，筛选得到条件和目的实体======")
    reference = forward.forward(paths, conditions, driver, aims)

    if reference != "":
        print(reference)
    else:
        print("没有匹配到实体")


    # 4. 调用大模型生成最终答案
    print("======4. 调用大模型生成最终答案======")
    generation = answer.generate_answer(question, reference, model)
    return generation


if __name__ == "__main__":
    question = "machine translation领域和Computational linguistics领域有哪些相同的数据集？" # 用户输入的问题
    max_pop = 5 # 构建推理树时最大的推理跳数
    top_k = 10 # 如果一个实体满足next_label的邻居有多个，最多取top_k个
    model = "gpt-4o-mini" # 选择生成最终答案使用的模型。（提取条件和目的就使用spark，因为便宜，而且效果也还不错）
    schema_text_path = "data/IFLYTEC-NLP/GraphKnowledge/schema.txt"  # 所用知识图谱的关系定义文件，格式是：label:entity_name-relation-label:entity_name
    label_dict = back.build_label_dict(schema_text_path)

    uri = "bolt://10.43.108.62:7687"  # Neo4j连接URI
    user = "neo4j"  # 用户名
    password = "12345678"  # 密码
    driver = GraphDatabase.driver(uri, auth=(user, password)) # 创建数据库连接

    time0 = time.time()
    final_answer = main(question, max_pop, label_dict, driver, model)
    time1 = time.time()
    print(f"answer:\n{final_answer}\n用时:{time1-time0}")