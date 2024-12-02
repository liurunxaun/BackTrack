from BackTrack import extract as extract
from BackTrack import back as back
from BackTrack import forward as forward
from BackTrack import answer as answer


def back_track(question, max_pop, label_dict, driver, model, top_k):
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
    reference = forward.forward(paths, conditions, driver, aims, top_k)

    if reference != "":
        print(reference)
    else:
        print("没有匹配到实体")


    # 4. 调用大模型生成最终答案
    print("\n======4. 调用大模型生成最终答案======")
    generation = answer.generate_answer(question, reference, model)
    return generation