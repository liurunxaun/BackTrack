from BackTrack import extract as extract
from BackTrack import back as back
from BackTrack import forward as forward
from BackTrack import answer as answer
from BackTrack import entity_retriever

def back_track(question, max_pop, embedding_flag, label_dict, label_description_path, entity_extract_example_path, ReferenceTemplate_path, driver, neo4j_database_name, model, top_k):
    print(f"\n问题:{question}")

    # 1. 从问题中提取条件实体、目的实体、实体类型
    print("\n======1. 从问题中提取条件实体、目的实体、实体类型======")
    conditions, aims = extract.extract(question, label_description_path, entity_extract_example_path)
    if embedding_flag == "true":
        conditions = entity_retriever.retrieve_matching_entities(conditions)

    if len(conditions) != 0 and len(aims) != 0:
        print(f"conditions:{conditions}")
        print(f"aims:{aims}")
    else:
        print("问题中不包含知识图谱范围内的条件和目的")
        print("\n======2. 调用大模型生成最终答案======")
        generation = answer.generate_answer(question, "", ReferenceTemplate_path,"", "" ,model)
        return generation


    # 2. 倒推找抽象本体推理路径
    print("\n======2.倒推找抽象本体推理路径======")
    paths = back.aim_back(conditions, aims, max_pop, label_dict)

    if len(paths) != 0:
        paths_string = ""
        for rule in paths:
            paths_string += " -> ".join(rule) + "\n\t"
        print(f"rules:\n{paths_string}")
    else:
        print("没找到抽象本体推理路径。请尝试换个说法，或者描述的更详细一些")
        print("\n======3. 调用大模型生成最终答案======")
        generation = answer.generate_answer(question, "", ReferenceTemplate_path,"", "" ,model)
        success_excute_flag = 0
        return generation, success_excute_flag


    # 3. 正推找具体实体推理路径，筛选得到条件和目的实体
    print("\n======3. 正推具体实体推理路径，筛选得到条件和目的实体======")
    last_node_str, reasoning_path_str = forward.forward(paths, conditions, driver, neo4j_database_name, aims, top_k)

    if last_node_str != "" or reasoning_path_str != "":
        print(last_node_str)
        print(reasoning_path_str)
    else:
        print("没有匹配到实体")
        print("\n======4. 调用大模型生成最终答案======")
        generation = answer.generate_answer(question, paths_string, ReferenceTemplate_path, last_node_str, reasoning_path_str, model)
        success_excute_flag = 0
        return generation, success_excute_flag


    # 4. 调用大模型生成最终答案
    print("\n======4. 调用大模型生成最终答案======")
    generation = answer.generate_answer(question, paths_string, ReferenceTemplate_path, last_node_str, reasoning_path_str, model)
    success_excute_flag = 1
    return generation, success_excute_flag