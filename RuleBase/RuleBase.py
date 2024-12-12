from BackTrack import extract
from BackTrack import answer
from BackTrack import forward
from RuleBase import collect
from RuleBase import select

def rule_base(question, max_pop, label_dict, driver, model, top_k):
    print(f"\n问题:{question}")

    # 1. 从问题中提取条件实体、目的实体、实体类型
    print("\n======1. 从问题中提取条件实体和实体类型======")
    conditions, aims = extract.extract(question)

    if len(conditions) != 0:
        print(f"conditions:{conditions}")
    else:
        print("问题中不包含知识图谱范围内的条件")
        print("\n======2. 调用大模型生成最终答案======")
        generation = answer.generate_answer(question)
        return generation

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
        generation = answer.generate_answer(question)
        return generation

    # 3. 筛选对回答问题有帮助的问题
    print("\n======3. 大模型筛选对回答问题有帮助的路径======")
    rules = select.select_rules(paths, question)
    print(f"rules:\n{rules}")

    # 4. 正推生成实体路径
    print("\n======4. 正推生成实体路径======")
    reference = forward.rules_forward(rules, conditions, driver, top_k)

    if reference != "":
        print(reference)
    else:
        print("没有匹配到实体")

    # 5. 调用大模型生成最终答案
    print("\n======5. 调用大模型生成最终答案======")
    generation = answer.generate_answer(question, reference, model)

    return generation