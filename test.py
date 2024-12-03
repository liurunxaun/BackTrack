from BackTrack import back, BackTrack
from RuleBase import RuleBase
from neo4j import GraphDatabase
import time
import pandas as pd
from bert_score import score
import os
from datetime import datetime


def write_results_to_csv(output_file, data):
    """Write the results to a CSV file."""
    df = pd.DataFrame(data)
    df.to_csv(output_file, mode='a', header=not os.path.exists(output_file), index=False)


def create_output_file(output_dir, method, model, max_pop, top_k, timestamp):
    """Create an output file with method, model, max_pop, and top_k in the file name to avoid overwriting."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_file = os.path.join(output_dir, f"evaluation_results_{method}_{model}_maxpop{max_pop}_topk{top_k}_{timestamp}.csv")
    return output_file


def evaluate_and_save_results(df, method, max_pop, top_k, model, label_dict, driver, output_file, metric):
    """
    Evaluate the answers and save the results in the specified file.
    """
    total_p = 0
    total_r = 0
    total_f1 = 0
    num_samples = len(df)

    # Initialize list to store the results for CSV
    results = []

    # Write headers (only for the first write)
    headers = ['Question', 'Answer', 'Ref', 'P', 'R', 'F1', 'Time']
    write_results_to_csv(output_file, [headers])

    # Process each sample
    for i in range(num_samples):
        final_answer = ""
        question = df["query_en"][i]  # 用户输入的问题
        ref = [df["answer_en"][i]]

        print(f"\nquestion {i}")

        time0 = time.time()

        # Call BackTrack or RuleBase based on the method
        if method == "BackTrack":
            final_answer = BackTrack.back_track(question, max_pop, label_dict, driver, model, top_k)
        elif method == "RuleBase":
            final_answer = RuleBase.rule_base(question, max_pop, label_dict, driver, model, top_k)

        print(f"final_answer:\n{final_answer}")


        cand = [final_answer]

        # 判断要使用哪种指标
        if metric == "BERTScore":
            # Compute BERTScore
            P, R, F1 = score(cand, ref, lang="en", verbose=True, model_type="bert-large-uncased",
                             rescale_with_baseline=True)

        print(f"Precision: {P.mean():.4f}")
        print(f"Recall: {R.mean():.4f}")
        print(f"F1 Score: {F1.mean():.4f}")

        total_p += P.mean().item()
        total_r += R.mean().item()
        total_f1 += F1.mean().item()

        time1 = time.time()
        print(f"用时:{time1 - time0}")

        # Store per-sample results for CSV (removed 'Cand' field)
        results.append({
            'Question': question,
            'Answer': final_answer,
            'Ref': str(ref),
            'P': f"{P.mean():.4f}",
            'R': f"{R.mean():.4f}",
            'F1': f"{F1.mean():.4f}",
            'Time': f"{time1 - time0:.4f}"
        })

    # Calculate average scores
    avg_p = total_p / num_samples
    avg_r = total_r / num_samples
    avg_f1 = total_f1 / num_samples

    print(f"Average Precision: {avg_p:.4f}")
    print(f"Average Recall: {avg_r:.4f}")
    print(f"Average F1: {avg_f1:.4f}")

    # Append average scores to results and write them to the CSV
    results.append({
        'Question': 'Average',
        'Answer': '',
        'Ref': '',
        'P': f"{avg_p:.4f}",
        'R': f"{avg_r:.4f}",
        'F1': f"{avg_f1:.4f}",
        'Time': ''
    })

    write_results_to_csv(output_file, results)

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
    schema_text_path = "./data/IFLYTEC-NLP/GraphKnowledge/schema.txt"  # 所用知识图谱的关系定义文件，格式是：label:entity_name-relation-label:entity_name
    test_dataset = "./data/IFLYTEC-NLP/test200/200_QA_多文档.csv"  # 要进行测试的数据集路径，注意是csv格式的
    output_dir = "./output"  # 测试结果的存储路径，存储为csv文件，会根据上面的参数和当前时间命名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # 指定输出文件名中的时间戳
    metric = "BERTScore"  # 选择实验的指标，包括：BERTScore

    label_dict = back.build_label_dict(schema_text_path)
    df = pd.read_csv(test_dataset)
    output_file = create_output_file(output_dir, method, model, max_pop, top_k, timestamp)

    evaluate_and_save_results(df, method, max_pop, top_k, model, label_dict, driver, output_file, metric)
