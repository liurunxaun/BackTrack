import pickle
import pandas as pd

def load_kg_entity_embeddings(embedding_file_path):
    print("\n提前加载实体嵌入")
    # 读取知识图谱中实体的embedding
    with open(embedding_file_path, 'rb') as f1:
        entity_embeddings = pickle.load(f1)
    entity_embeddings_emb = pd.DataFrame(entity_embeddings["embeddings"])
    print(f"加载完毕，共有{len(entity_embeddings_emb)}")
    # 从加载的数据中提取向量化器
    vectorizer = entity_embeddings["vectorizer"]
    return entity_embeddings, vectorizer

def match_knowledge_graph_entities(conditions, vectorizer, entity_embeddings):
    """
    将条件实体替换为知识图谱中的所有相似的实体
    """
    match_conditions = []
    match_kg = []
    question_match_kg = []
    global_matched_entities = set()  # 全局去重的集合
    # 需要过滤的包含数字的实体列表
    exclude_entities_with_number = ['R dataset', 'Dataset S', 'Dataset-I', 'Dataset I', '6 datasets', '4 datasets',
                                    '7 datasets', 'Model 1, Model 2, Model 3', 'Model 0', 'Model L', 'S-model', 'HoW']

    def is_excluded(entity):
        # 如果实体在排除列表中，则认为它需要被过滤
        return entity.strip() in exclude_entities_with_number

    for kg_entity in question_kg:
        # 判断是否是 'dataset' 或 'model'（不区分大小写、单复数形式）
        if 'dataset' == kg_entity.lower() or 'model' == kg_entity.lower():
            # 如果是 'dataset' 或 'model'，只返回最相似的实体
            query_embedding = vectorizer.transform([kg_entity])
            cos_similarities = cosine_similarity(query_embedding, entity_embeddings["embeddings"]).flatten()
            top_index = cos_similarities.argmax()  # 获取最相似的实体的索引
            best_match_entity = entity_embeddings["entities"][top_index]  # 获取最相似的实体名称

            # 判断最相似的实体是否被排除
            if not is_excluded(best_match_entity):
                match_kg.append([best_match_entity])  # 只返回最相似的实体
            else:
                match_kg.append([])  # 如果最相似的实体被排除，返回空列表

        else:
            # 对于其他实体，仍然返回前五个最相似的实体
            query_embedding = vectorizer.transform([kg_entity])
            cos_similarities = cosine_similarity(query_embedding, entity_embeddings["embeddings"]).flatten()
            top_5_indices = cos_similarities.argsort()[-5:][::-1]  # 获取前五个最大相似度的索引

            # 当前问题实体的匹配实体集合（局部去重）
            matched_entities = set()
            max_similarity = -1  # 用来存储最大相似度
            best_match_entity = None  # 用来存储相似度最大的实体
            found_similarity_1 = False

            # 遍历前五个最相似的实体，检查相似度条件
            for idx in top_5_indices:
                match_kg_i = entity_embeddings["entities"][idx]  # 获取相应的实体名称
                similarity = cos_similarities[idx]

                # 条件1: 如果相似度为 1，保存该实体
                if similarity == 1 and not is_excluded(match_kg_i):  # 直接检查实体是否在排除列表中
                    if match_kg_i not in global_matched_entities:
                        matched_entities.add(match_kg_i)
                        found_similarity_1 = True  # 标记找到相似度为 1 的实体

                # 更新最大相似度的实体
                if similarity > max_similarity and match_kg_i not in global_matched_entities and not is_excluded(
                        match_kg_i):
                    max_similarity = similarity
                    best_match_entity = match_kg_i

            # 遍历所有实体，寻找包含关系满足的实体，只保留前五个
            count = 0  # 初始化计数器
            for i, match_kg_i in enumerate(entity_embeddings["entities"]):
                if count >= 3:  # 如果已添加五个实体，则退出循环
                    break
                if kg_entity.lower() in match_kg_i.lower() and not is_excluded(match_kg_i):  # 检查包含关系，不区分大小写，且没有数字
                    similarity = cos_similarities[i]
                    if match_kg_i not in global_matched_entities:
                        matched_entities.add(match_kg_i)
                        count += 1  # 更新计数器

            if best_match_entity:
                matched_entities.add(best_match_entity)

            match_kg.append(list(matched_entities))  # 添加包含关系匹配的前五个

            # 将当前匹配的实体添加到全局去重集合中
            global_matched_entities.update(matched_entities)
            question_match_kg.append([kg_entity, list(matched_entities)])

    print('match_kg', match_kg, "\n")
    return match_kg, question_match_kg

    return match_conditions