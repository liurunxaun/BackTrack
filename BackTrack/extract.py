from utils.LLM import spark as llm


def extract(question, label_description_path, entity_extract_example_path):
    """
    输入：用户的问题
    处理过程：使用大模型提取条件实体、目的实体、实体类型。
            处理大模型的答复，为condition_entity, condition_label, aim_entity, aim_label赋值
    输出：conditions:[entity, label], aims:[entity, label]
    """

    # 读取改数据集的实体类型
    label_description = ""
    try:
        with open(label_description_path, "r", encoding="utf-8") as file:
            label_description = file.read()
    except FileNotFoundError:
        print(f"文件 {label_description_path} 未找到，请检查路径！")
    except Exception as e:
        print(f"读取文件时出错：{e}")

    # 读取实体提取示例
    entity_extract_example = ""
    try:
        with open(entity_extract_example_path, "r", encoding="utf-8") as file:
            entity_extract_example = file.read()
    except FileNotFoundError:
        print(f"文件 {entity_extract_example_path} 未找到，请检查路径！")
    except Exception as e:
        print(f"读取文件时出错：{e}")


    query = f"""
    我在做一个知识图谱增强的知识问答系统，知识图谱由从论文中提取的要素组成，节点包括作者姓名、方法、数据集等。
    你的任务是从用户输入的问题中提取**条件实体及其类型**和**目的实体及其类型**。
    
    ### 请从以下表格中选择实体的类型：
    每行描述一种实体类型，格式是-实体的类型(描述信息)
    {label_description}

    ### 规则
    - 条件实体是问题中提供的已知信息；
    - 目的实体是问题中用户想要查询的内容；
    - 如果没有合适的实体，请用 "none" 表示。

    ### 输出格式
    - 条件实体和目的实体之间用 **"."**（英文句号）隔开；
    - 每个实体的格式为 **"实体名称,实体类型"**；
    - 如果有多个条件实体或目的实体，使用 **";"**（英文分号）分隔；
    - 如果只有条件和目的其中之一例如：
        -只有条件实体，没有目的实体的情况下输出：ce1,cl1;ce2,cl2.none,none
        -只有目的实体，没有条件实体的情况下输出：none,none.ae1,al1;ae2,al2
    - **只输出最终答案**，不要包含多余的说明、解释或文字。

    ### 示例
    {entity_extract_example}

    用户问题是：{question}
    请生成符合上述格式的答案：
    """

    response = llm.spark_4_0_company(query)
    print(f"大模型返回:\n{response}\n")

    conditions = []
    aims = []

    try:
        split = response.split(".")

        split1 = split[0].split(";")
        for item in split1:
            item_split = item.split(",")
            conditions.append([ item_split[0], item_split[1] ]) # 0是实体，1是实体类型

        split2 = split[1].split(";")
        for item in split2:
            item_split = item.split(",")
            aims.append([ item_split[0], item_split[1] ]) # 0是实体，1是实体类型
    except:
        return [],[]

    # match_conditions = []
    # if len(conditions) != 0:
    #     match_conditions = match_knowledge_graph_entities(conditions)

    return conditions, aims