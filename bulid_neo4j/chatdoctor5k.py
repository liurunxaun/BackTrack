"""
@Time: 2024/12/24 16:55
@Author: yanzx
@Desc: 
"""
from neo4j import GraphDatabase, basic_auth
import pandas as pd

def build_ne4j_label(df_path,schema_path):
    # clean all
    print("删除原有知识图谱数据库")
    session.run("MATCH (n) DETACH DELETE n")
    print("删除成功")
    df = pd.read_csv(df_path, sep='\t', header=None, names=['head', 'relation', 'tail'])
    schema = pd.read_csv(schema_path, sep='-', header=None, names=['head_label', 'relation', 'tail_label'])
    relation_to_labels = {}  # 构建关系到类别映射的字典
    for _, row in schema.iterrows():
        relation_to_labels[row['relation']] = (row['head_label'], row['tail_label'])

    print(f"正在读train")
    for index, row in df.iterrows():
        head_name,tail_name = row['head'],row['tail']
        relation_name = row['relation']
        # 打印当前处理的三元组
        print(f"处理第 {index + 1} 行: {head_name} --[{relation_name}]--> {tail_name}")

        if relation_name in relation_to_labels:
            head_label, tail_label = relation_to_labels[relation_name]

        # 提取实体和标签
        try:
            # 创建带标签的节点
            query = (
                    "MERGE (h:Entity { name: $head_name }) "
                    "SET h.label = $head_label "
                    "MERGE (t:Entity { name: $tail_name }) "
                    "SET t.label = $tail_label "
                    "MERGE (h)-[r:`" + relation_name + "`]->(t)"
            )
            session.run(query, head_name=head_name, tail_name=tail_name, head_label=head_label,
                        tail_label=tail_label, relation_name=relation_name)

        except Exception as e:
            print(f"第条数据有问题: {e}")


    print("构建完成")


if __name__ == "__main__":
    # 连接neo4j
    uri = "bolt://10.43.108.62:7687"
    username = "neo4j"
    password = "12345678"
    driver = GraphDatabase.driver(uri, auth=(username, password))
    database_name = "chatdoctor5k"
    session = driver.session(database=database_name)
    print("\n成功连接neo4j\n")

    schema_path = "/work/beiluo/BackTrack-master/data/chatdoctor5k/GraphKnowledge/schema.txt"
    df_path = "/work/beiluo/BackTrack-master/data/chatdoctor5k/train.txt"
    build_ne4j_label(df_path,schema_path)