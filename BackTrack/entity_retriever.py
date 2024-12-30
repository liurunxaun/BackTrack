import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import pickle

tokenizer = AutoTokenizer.from_pretrained('/Users/yanzhenxing/Desktop/科大-讯飞/code_RAG/model/sentence-transformers/all-mpnet-base-v2')
model = AutoModel.from_pretrained('/Users/yanzhenxing/Desktop/科大-讯飞/code_RAG/model/sentence-transformers/all-mpnet-base-v2')


def load_entity_embeddings(file_path):
    """
    从预存的文件加载实体库的嵌入和对应的实体名称。

    参数:
        file_path (str): 实体库嵌入文件路径。

    返回:
        tuple: (entities, embeddings)
        entities (list of str): 实体名称列表。
        embeddings (np.array): 实体的嵌入矩阵。
    """
    with open(file_path, "rb") as f:
        data = pickle.load(f)
        entities = data["entities"]
        embeddings = np.array(data["embeddings"])
    return entities, embeddings


def compute_embedding(text, tokenizer, model, device):
    """
    使用 tokenizer 和 model 计算文本的嵌入。

    参数:
        text (str): 输入文本。
        tokenizer (AutoTokenizer): Hugging Face 的分词器。
        model (AutoModel): Hugging Face 的模型。
        device (str): 运行设备（'cuda' 或 'cpu'）。

    返回:
        np.array: 文本的嵌入向量。
    """
    encoded_input = tokenizer(
        [text], padding=True, truncation=True, return_tensors="pt", max_length=128
    )
    encoded_input = {key: value.to(device) for key, value in encoded_input.items()}

    # 使用模型生成嵌入
    with torch.no_grad():
        model_output = model(**encoded_input)
        embedding = model_output.last_hidden_state.mean(dim=1)  # 取平均池化作为句子嵌入
    return embedding.cpu().numpy()


def retrieve_matching_entities(question_entities, entity_embeddings_path = "./data/chatdoctor5k/entity_embeddings.pkl", device="cpu"):
    """
    实时计算问题实体与实体库中的实体相似度，并替换原问题实体。

    参数:
        question_entities (list of lists): 问题实体列表，每个元素为 [实体, ...]。
        entity_embeddings_path (str): 实体库嵌入文件路径。
        tokenizer (AutoTokenizer): Hugging Face 的分词器。
        model (AutoModel): Hugging Face 的模型。
        device (str): 运行设备（'cuda' 或 'cpu'）。

    返回:
        list: 更新后的 question_entities，匹配的实体替换原来的 question_entities[i][0]。
    """
    # 加载实体库的嵌入和实体名称
    print("加载实体库嵌入...")
    entities, entity_embeddings = load_entity_embeddings(entity_embeddings_path)

    # 遍历每个问题实体
    for i, x in enumerate(question_entities):
        kg_entity = x[0]  # 问题实体

        # 为问题实体生成嵌入
        kg_entity_emb = compute_embedding(kg_entity, tokenizer, model, device)

        # 计算问题实体与实体库的余弦相似度
        cos_similarities = cosine_similarity(kg_entity_emb, entity_embeddings)[0]

        # 找到相似度最高的实体
        max_index = np.argmax(cos_similarities)
        best_match = entities[max_index]
        best_similarity = cos_similarities[max_index]

        # 输出匹配结果
        print(f"{kg_entity}匹配到的实体：{best_match}，相似度：{best_similarity:.4f}")

        # 替换原问题实体为匹配实体
        question_entities[i][0] = best_match.replace(" ", "_")

    return question_entities  # 返回更新后的实体列表


# 示例用法
# if __name__ == "__main__":
#     # 加载模型和分词器
#     tokenizer = AutoTokenizer.from_pretrained('model/sentence-transformers/all-mpnet-base-v2')
#     model = AutoModel.from_pretrained('model/sentence-transformers/all-mpnet-base-v2')
#
#
#     # 示例问题实体
#     question_entities = [
#         ['a hoarse voice', '症状'],
#         ['poisoning', '症状']
#     ]
#
#     # 调用函数
#     updated_question_entities = retrieve_matching_entities(
#         question_entities
#     )
#     print("更新后的实体列表：", updated_question_entities)
