from BackTrack import back
from neo4j import GraphDatabase
from RuleBase import RuleBase


if __name__ == "__main__":
    uri = "bolt://10.43.108.62:7687"  # Neo4j连接URI
    user = "neo4j"  # 用户名
    password = "12345678"  # 密码
    driver = GraphDatabase.driver(uri, auth=(user, password))  # 创建数据库连接

    # 算法可供选择的参数
    max_pop = 5  # 构建推理树时最大的推理跳数
    top_k = 5  # 如果一个实体满足next_label的邻居有多个，最多取top_k个
    generate_answer_model = "spark"  # 选择生成最终答案使用的模型。包括：spark, gpt-4o-mini（提取条件和目的就使用spark，因为便宜，而且效果也还不错）

    # 使用不同数据集需要修改的参数
    schema_text_path = "./data/IFLYTEC-NLP/GraphKnowledge/schema.txt"  # 所用知识图谱的关系定义文件，格式是：label-relation-label
    label_dict = back.build_label_dict(schema_text_path)
    label_description_path = "./data/IFLYTEC-NLP/GraphKnowledge/label_description.txt"
    entity_extract_example_path = "./data/IFLYTEC-NLP/GraphKnowledge/entity_extract_example.txt"
    neo4j_database_name = "neo4j"

    question = "论文Linguistic Knowledge Generator和论文Better Translation with Knowledge Extracted from Source Text，谁的Accuracy更高"
    answer, successFlag = RuleBase.rule_base(question, max_pop, label_dict, label_description_path, entity_extract_example_path, driver,
              neo4j_database_name, generate_answer_model, top_k)
    print(answer)