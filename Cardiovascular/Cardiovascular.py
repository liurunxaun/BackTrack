import json

from flask import Flask, request
from neo4j import GraphDatabase

from BackTrack import answer as answer
from BackTrack import back as back
from BackTrack import extract as extract
from BackTrack import forward as forward
from RuleBase import collect
from RuleBase import select

app = Flask(__name__)


def json_pack(answers):
    result = {"answer": answers}
    print()
    print(result)
    # 封装成json格式，并确保中文正常显示
    json_result = json.dumps(result, ensure_ascii=False)
    # 将处理结果返回给 Java
    return json_result


@app.route('/process-data', methods=['POST'])
def main():
    # 接收 Java 发来的数据
    data = request.json
    question = data.get('question')
    print('\nQuestion:', question)

    uri = "bolt://localhost:7687"  # Neo4j连接URI
    user = "neo4j"  # 用户名
    password = "12345678"  # 密码
    driver = GraphDatabase.driver(uri, auth=(user, password))  # 创建数据库连接

    max_pop = 5  # 构建推理树时最大的推理跳数
    top_k = 5  # 如果一个实体满足next_label的邻居有多个，最多取top_k个
    model = "spark"  # 选择生成最终答案使用的模型。包括：spark, gpt-4o-mini（提取条件和目的就使用spark，因为便宜，而且效果也还不错）

    schema_text_path = "../data/Cardiovascular/schema.txt"  # 所用知识图谱的关系定义文件，格式是：label:entity_name-relation-label:entity_name
    label_dict = back.build_label_dict(schema_text_path)
    label_description_path = "../data/Cardiovascular/label_description.txt"
    entity_extract_example_path = "../data/Cardiovascular/entity_extract_example.txt"
    neo4j_database_name = "cardiovascularMini"

    # 1. 从问题中提取条件实体、目的实体、实体类型
    print("\n======1. 从问题中提取条件实体、目的实体、实体类型======")
    conditions, aims = extract.extract(question, label_description_path, entity_extract_example_path)

    if len(conditions) != 0 and len(aims) != 0:
        print(f"conditions:{conditions}")
        print(f"aims:{aims}")
    else:
        print("问题中不包含知识图谱范围内的条件和目的")
        print("\n======2. 调用大模型生成最终答案======")
        generation = answer.generate_answer(question, "", model)
        return json_pack([generation, "", "", []])

    # 2. 倒推找抽象本体推理路径
    print("\n======2. 从条件出发收集全部路径======")
    paths = collect.collect_paths(conditions, max_pop, label_dict)

    if len(paths) != 0:
        print(paths)
    else:
        print("没找到抽象本体推理路径。请尝试换个说法，或者描述的更详细一些")
        print("\n======3. 调用大模型生成最终答案======")
        generation = answer.generate_answer(question, "", model)
        return json_pack([generation,"","",[]])

    # 3. 筛选对回答问题有帮助的问题
    print("\n======3. 大模型筛选对回答问题有帮助的路径======")
    rules = select.select_rules(paths, question)

    if len(rules) != 0:
        rules_string = ""
        for rule in rules:
            rules_string += " -> ".join(rule) + "\n"
        print(f"rules:\n{rules_string}")

    # 4. 正推生成实体路径
    print("\n======4. 正推生成实体路径======")
    reference = forward.rules_forward(rules, conditions, driver, neo4j_database_name, top_k)

    if reference != "":
        print(reference)
    else:
        print("没有匹配到实体")
        print("\n======5. 调用大模型生成最终答案======")
        generation = answer.generate_answer(question, reference, model)
        return json_pack([generation, rules_string, "",[]])

    # 4. 调用大模型生成最终答案
    print("\n======5. 调用大模型生成最终答案======")
    generation = answer.generate_answer(question, reference, model)
    return json_pack([generation, rules_string, reference, []])


if __name__ == "__main__":
    # 监听接口，生成并返回答案
    print("\n开始接收请求")
    app.run(host='0.0.0.0', port=5002)
    main()
