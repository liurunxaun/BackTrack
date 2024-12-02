from BackTrack import back, BackTrack
from RuleBase import RuleBase
from neo4j import GraphDatabase
import time
import pandas as pd
from bert_score import score


if __name__ == "__main__":

    method = "BackTrack"  # 选择要使用的方法，包括"BackTrack"倒推，"RuleBase"基于规则
    max_pop = 5  # 构建推理树时最大的推理跳数
    top_k = 5  # 如果一个实体满足next_label的邻居有多个，最多取top_k个
    model = "gpt-4o-mini"  # 选择生成最终答案使用的模型。包括：spark, gpt-4o-mini（提取条件和目的就使用spark，因为便宜，而且效果也还不错）
    schema_text_path = "data/IFLYTEC-NLP/GraphKnowledge/schema.txt"  # 所用知识图谱的关系定义文件，格式是：label:entity_name-relation-label:entity_name
    label_dict = back.build_label_dict(schema_text_path)

    uri = "bolt://10.43.108.62:7687"  # Neo4j连接URI
    user = "neo4j"  # 用户名
    password = "12345678"  # 密码
    driver = GraphDatabase.driver(uri, auth=(user, password))  # 创建数据库连接

    df = pd.read_csv("/Users/liurunxuan/学习/科大讯飞实习/BackTrack/data/IFLYTEC-NLP/test200/200_QA_多文档.csv")
    query_en = df["query_en"]
    answer_en = df["answer_en"]

    for i in range(5):

        final_answer = ""
        question = query_en[i]  # 用户输入的问题

        time0 = time.time()

        if method == "BackTrack":
            final_answer = BackTrack.back_track(question, max_pop, label_dict, driver, model, top_k)
        elif method == "RuleBase":
            final_answer = RuleBase.rule_base(question, max_pop, label_dict, driver, model, top_k)

        print(f"final_answer:\n{final_answer}")

        # 计算 BERTScore
        cand = [final_answer]
        ref = [answer_en[i]]
        P, R, F1 = score(cand, ref, lang="en", verbose=True, model_type='bert-large-uncased', rescale_with_baseline=True)

        # 打印结果
        print(f"Precision: {P.mean():.4f}")
        print(f"Recall: {R.mean():.4f}")
        print(f"F1 Score: {F1.mean():.4f}")

        time1 = time.time()
        print(f"用时:{time1-time0}")