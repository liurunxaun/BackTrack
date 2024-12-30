from BackTrack import extract
from BackTrack import answer
from BackTrack import forward
from RuleBase import collect
from RuleBase import select

def rule_base(question, max_pop, embedding_flag, label_dict, label_description_path, entity_extract_example_path, ReferenceTemplate_path, driver, neo4j_database_name, model, top_k):
    print(f"\n问题:{question}")

    # 1. 从问题中提取条件实体、目的实体、实体类型
    print("\n======1. 从问题中提取条件实体和实体类型======")
    conditions, aims = extract.extract(question, label_description_path, entity_extract_example_path)
    if embedding_flag == "true":
        from BackTrack import entity_retriever
        conditions = entity_retriever.retrieve_matching_entities(conditions)

    if len(conditions) != 0:
        print(f"匹配完之后的conditions:{conditions}")
    else:
        print("问题中不包含知识图谱范围内的条件")
        print("\n======2. 调用大模型生成最终答案======")
        generation = answer.generate_answer(question, "", ReferenceTemplate_path,"", "" ,model)
        success_excute_flag = 0
        return generation, success_excute_flag

    # 2. 从条件出发收集全部路径
    print("\n======2. 从条件出发收集全部路径======")
    paths = collect.collect_paths(conditions, max_pop, label_dict)

    if len(paths) != 0:
        print("从所有条件出发的所有本体推理路径:")
        for path in paths:
            print(" -> ".join(path))
    else:
        print("没找到抽象本体推理路径。请尝试换个说法，或者描述的更详细一些")
        print("\n======3. 调用大模型生成最终答案======")
        generation = answer.generate_answer(question, "", ReferenceTemplate_path, "", "", model)
        success_excute_flag = 0
        return generation, success_excute_flag

    # 3. 筛选对回答问题有帮助的问题
    print("\n======3. 大模型筛选对回答问题有帮助的路径======")
    rules = select.select_rules(paths, question, aims)
    print(f"rules:\n{rules}")
    if len(rules) != 0:
        rules_string = ""
        for rule in rules:
            rules_string += " -> ".join(rule) + "\n\t"
        print(f"rules:\n{rules_string}")

    # 4. 正推生成实体路径
    print("\n======4. 正推生成实体路径======")
    last_node_str, reasoning_path_str = forward.rules_forward(rules, conditions, driver, neo4j_database_name, top_k)

    if last_node_str != "" or reasoning_path_str != "":
        print(last_node_str)
        print(reasoning_path_str)
    else:
        print("没有匹配到实体")
        print("\n======5. 调用大模型生成最终答案======")
        generation = answer.generate_answer(question, rules_string, ReferenceTemplate_path, last_node_str, reasoning_path_str, model)
        success_excute_flag = 0
        return generation, success_excute_flag

    # 5. 调用大模型生成最终答案
    print("\n======5. 调用大模型生成最终答案======")
    generation = answer.generate_answer(question, rules_string, ReferenceTemplate_path, last_node_str, reasoning_path_str, model)
    success_excute_flag = 1
    return generation, success_excute_flag
