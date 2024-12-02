from BackTrack import back, BackTrack
from RuleBase import RuleBase
from neo4j import GraphDatabase
import time
import pandas as pd
from bert_score import score
import os
from datetime import datetime


def write_results(file_path, data):
    """Write the results to a file"""
    with open(file_path, "a") as f:
        for entry in data:
            f.write(entry + "\n")


def create_output_file(output_dir, timestamp):
    """Create an output file with a timestamp to avoid overwriting."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_file = os.path.join(output_dir, f"evaluation_results_{timestamp}.txt")
    return output_file


def evaluate_and_save_results(df, method, max_pop, top_k, model, label_dict, driver, output_file):
    """
    Evaluate the answers and save the results in the specified file.
    """
    total_p = 0
    total_r = 0
    total_f1 = 0
    num_samples = len(df)

    # Write headers
    write_results(output_file, [f"Method: {method}, Model: {model}, max_pop: {max_pop}, top_k: {top_k}\n"])
    write_results(output_file, [f"Test Dataset: {df['query_en'].name}\n"])
    write_results(output_file, ["Average Precision: 0.0000"])
    write_results(output_file, ["Average Recall: 0.0000"])
    write_results(output_file, ["Average F1: 0.0000"])

    # Process each sample
    for i in range(num_samples):
        final_answer = ""
        question = df["query_en"][i]  # 用户输入的问题
        ref = [df["answer_en"][i]]

        print(f"\nquestion{i}: {question}")

        time0 = time.time()

        # Call BackTrack or RuleBase based on the method
        if method == "BackTrack":
            final_answer = BackTrack.back_track(question, max_pop, label_dict, driver, model, top_k)
        elif method == "RuleBase":
            final_answer = RuleBase.rule_base(question, max_pop, label_dict, driver, model, top_k)

        print(f"final_answer:\n{final_answer}")

        # Compute BERTScore
        cand = [final_answer]
        P, R, F1 = score(cand, ref, lang="en", verbose=True, model_type='bert-large-uncased',
                         rescale_with_baseline=True)

        print(f"Precision: {P.mean():.4f}")
        print(f"Recall: {R.mean():.4f}")
        print(f"F1 Score: {F1.mean():.4f}")

        total_p += P.mean().item()
        total_r += R.mean().item()
        total_f1 += F1.mean().item()

        time1 = time.time()
        print(f"用时:{time1 - time0}")

        # Write per-sample results
        result = (f"Question: {question}\n"
                  f"Answer: {final_answer}\n"
                  f"Cand: {cand}\n"
                  f"Ref: {ref}\n"
                  f"P: {P.mean():.4f}, R: {R.mean():.4f}, F1: {F1.mean():.4f}\n"
                  f"Time: {time1 - time0}\n")
        write_results(output_file, [result])

    # Calculate and write average P, R, F1
    avg_p = total_p / num_samples
    avg_r = total_r / num_samples
    avg_f1 = total_f1 / num_samples

    print(f"Average Precision: {avg_p:.4f}")
    print(f"Average Recall: {avg_r:.4f}")
    print(f"Average F1: {avg_f1:.4f}")

    # Update file with average scores
    with open(output_file, "r") as f:
        lines = f.readlines()

    # Insert average scores at the beginning
    lines[2] = f"Average Precision: {avg_p:.4f}\n"
    lines[3] = f"Average Recall: {avg_r:.4f}\n"
    lines[4] = f"Average F1: {avg_f1:.4f}\n"

    with open(output_file, "w") as f:
        f.writelines(lines)

    print("Evaluation completed and results saved.")


if __name__ == "__main__":
    uri = "bolt://10.43.108.62:7687"  # Neo4j连接URI
    user = "neo4j"  # 用户名
    password = "12345678"  # 密码
    driver = GraphDatabase.driver(uri, auth=(user, password))  # 创建数据库连接

    method = "BackTrack"  # 选择要使用的方法，包括"BackTrack"倒推，"RuleBase"基于规则
    max_pop = 5  # 构建推理树时最大的推理跳数
    top_k = 5  # 如果一个实体满足next_label的邻居有多个，最多取top_k个
    model = "gpt-4o-mini"  # 选择生成最终答案使用的模型。包括：spark, gpt-4o-mini（提取条件和目的就使用spark，因为便宜，而且效果也还不错）
    schema_text_path = "data/IFLYTEC-NLP/GraphKnowledge/schema.txt"  # 所用知识图谱的关系定义文件，格式是：label:entity_name-relation-label:entity_name
    output_dir = "/Users/liurunxuan/学习/科大讯飞实习/BackTrack/output"
    test_dataset = "/Users/liurunxuan/学习/科大讯飞实习/BackTrack/data/IFLYTEC-NLP/test200/200_QA_多文档.csv"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    label_dict = back.build_label_dict(schema_text_path)
    df = pd.read_csv(test_dataset)
    output_file = create_output_file(output_dir, timestamp)

    evaluate_and_save_results(df, method, max_pop, top_k, model, label_dict, driver, output_file)
