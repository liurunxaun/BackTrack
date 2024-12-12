from utils.LLM import spark as llm


# def match_knowledge_graph_entities(conditions, vectorizer, entity_embeddings):
#     """
#     将条件实体替换为知识图谱中的所有相似的实体
#     """
#     match_conditions = []
#     match_kg = []
#     question_match_kg = []
#     global_matched_entities = set()  # 全局去重的集合
#     # 需要过滤的包含数字的实体列表
#     exclude_entities_with_number = ['R dataset', 'Dataset S', 'Dataset-I', 'Dataset I', '6 datasets', '4 datasets',
#                                     '7 datasets', 'Model 1, Model 2, Model 3', 'Model 0', 'Model L', 'S-model', 'HoW']
#
#     def is_excluded(entity):
#         # 如果实体在排除列表中，则认为它需要被过滤
#         return entity.strip() in exclude_entities_with_number
#
#     for kg_entity in question_kg:
#         # 判断是否是 'dataset' 或 'model'（不区分大小写、单复数形式）
#         if 'dataset' == kg_entity.lower() or 'model' == kg_entity.lower():
#             # 如果是 'dataset' 或 'model'，只返回最相似的实体
#             query_embedding = vectorizer.transform([kg_entity])
#             cos_similarities = cosine_similarity(query_embedding, entity_embeddings["embeddings"]).flatten()
#             top_index = cos_similarities.argmax()  # 获取最相似的实体的索引
#             best_match_entity = entity_embeddings["entities"][top_index]  # 获取最相似的实体名称
#
#             # 判断最相似的实体是否被排除
#             if not is_excluded(best_match_entity):
#                 match_kg.append([best_match_entity])  # 只返回最相似的实体
#             else:
#                 match_kg.append([])  # 如果最相似的实体被排除，返回空列表
#
#         else:
#             # 对于其他实体，仍然返回前五个最相似的实体
#             query_embedding = vectorizer.transform([kg_entity])
#             cos_similarities = cosine_similarity(query_embedding, entity_embeddings["embeddings"]).flatten()
#             top_5_indices = cos_similarities.argsort()[-5:][::-1]  # 获取前五个最大相似度的索引
#
#             # 当前问题实体的匹配实体集合（局部去重）
#             matched_entities = set()
#             max_similarity = -1  # 用来存储最大相似度
#             best_match_entity = None  # 用来存储相似度最大的实体
#             found_similarity_1 = False
#
#             # 遍历前五个最相似的实体，检查相似度条件
#             for idx in top_5_indices:
#                 match_kg_i = entity_embeddings["entities"][idx]  # 获取相应的实体名称
#                 similarity = cos_similarities[idx]
#
#                 # 条件1: 如果相似度为 1，保存该实体
#                 if similarity == 1 and not is_excluded(match_kg_i):  # 直接检查实体是否在排除列表中
#                     if match_kg_i not in global_matched_entities:
#                         matched_entities.add(match_kg_i)
#                         found_similarity_1 = True  # 标记找到相似度为 1 的实体
#
#                 # 更新最大相似度的实体
#                 if similarity > max_similarity and match_kg_i not in global_matched_entities and not is_excluded(
#                         match_kg_i):
#                     max_similarity = similarity
#                     best_match_entity = match_kg_i
#
#             # 遍历所有实体，寻找包含关系满足的实体，只保留前五个
#             count = 0  # 初始化计数器
#             for i, match_kg_i in enumerate(entity_embeddings["entities"]):
#                 if count >= 3:  # 如果已添加五个实体，则退出循环
#                     break
#                 if kg_entity.lower() in match_kg_i.lower() and not is_excluded(match_kg_i):  # 检查包含关系，不区分大小写，且没有数字
#                     similarity = cos_similarities[i]
#                     if match_kg_i not in global_matched_entities:
#                         matched_entities.add(match_kg_i)
#                         count += 1  # 更新计数器
#
#             if best_match_entity:
#                 matched_entities.add(best_match_entity)
#
#             match_kg.append(list(matched_entities))  # 添加包含关系匹配的前五个
#
#             # 将当前匹配的实体添加到全局去重集合中
#             global_matched_entities.update(matched_entities)
#             question_match_kg.append([kg_entity, list(matched_entities)])
#
#     print('match_kg', match_kg, "\n")
#     return match_kg, question_match_kg
#
#
#
#     return match_conditions


def extract(question):
    """
    输入：用户的问题
    处理过程：使用大模型提取条件实体、目的实体、实体类型。
            处理大模型的答复，为condition_entity, condition_label, aim_entity, aim_label赋值
    输出：conditions:[entity, label], aims:[entity, label]
    """

    query = f"""
    我在做一个知识图谱增强的知识问答系统，知识图谱由从论文中提取的要素组成，节点包括作者姓名、方法、数据集等。
    你的任务是从用户输入的问题中提取**条件实体及其类型**和**目的实体及其类型**。
    
    ### 请从以下表格中选择实体的类型：
    - 作者（作者姓名）
    - 标题（论文标题）
    - 机构（作者所在机构）
    - 领域（论文所属领域）
    - 会议（论文发表的会议或期刊名称）
    - 关键词（论文关键词）
    - 问题（论文所解决的问题）
    - 方法（论文中提出的方法）
    - 模型（论文中使用的模型名称）
    - 任务（论文针对的任务）
    - 数据集（论文实验所使用的数据集）
    - 创新点（论文的创新点）
    - 指标（论文实验所使用的指标）
    - id（论文在数据库中的id）

    ### 规则
    - 条件实体是问题中提供的已知信息；
    - 目的实体是问题中用户想要查询的内容；
    - 如果没有合适的实体，请用 "none" 表示。

    ### 输出格式
    - 条件实体和目的实体之间用 **"."**（英文句号）隔开；
    - 每个实体的格式为 **"实体名称,实体类型"**；
    - 如果有多个条件实体或目的实体，使用 **";"**（英文分号）分隔；
    - **只输出最终答案**，不要包含多余的说明、解释或文字。

    ### 示例
    示例1:
    输入: Kausalgie发表过多少篇论文？
    输出: Kausalgie,作者.papers,标题

    示例2:
    输入: machine translation领域有哪些数据集？
    输出: machine translation,领域.dataset,数据集

    示例3:
    输入: SpellGCN的作者是谁？
    输出: SpellGCN,标题.author,作者

    示例4:
    输入: MindMap和KAG哪个更早？作者分别是谁？
    输出: MindMap,方法;KAG,方法.earlier,none;author,作者

    示例5:
    输入: In the field of machine translation, what are the ways to obtain and generate appropriate data sets for low-resource situations?
    输出: machine translation,领域;low-resource situations,问题;obtain and generate appropriate data sets,问题.ways,方法

    用户问题是：{question}
    请生成符合上述格式的答案：
    """

    response = llm.spark_4_0(query)
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