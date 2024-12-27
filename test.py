from BackTrack import back, BackTrack, answer
from RuleBase import RuleBase
from neo4j import GraphDatabase
import time
import pandas as pd
from bert_score import score
import os
from datetime import datetime


def create_output_file(output_dir, method, model, max_pop, top_k, timestamp):
    """Create an output file with method, model, max_pop, and top_k in the file name to avoid overwriting."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_file = os.path.join(output_dir, f"evaluation_results_{method}_{model}_maxpop{max_pop}_topk{top_k}_{timestamp}.csv")
    return output_file


def write_results_to_csv(output_file, data):
    """Write the results to a CSV file."""
    df = pd.DataFrame(data)
    df.to_csv(output_file, mode='a', header=False, index=False)  # 始终禁用表头和行索引写入


def evaluate_and_save_results(df, method, max_pop, top_k, model, label_dict, label_description_path, entity_extract_example_path, driver, neo4j_database_name, output_file, metric, questions, reference_answers):
    """
    Evaluate the answers and save the results in the specified file.
    """
    total_p = 0
    total_r = 0
    total_f1 = 0
    num_samples = len(df)

    # Write headers to the file (only once at the start)
    if not os.path.exists(output_file):
        df_empty = pd.DataFrame(columns=['Question', 'Answer', 'Ref', 'P', 'R', 'F1', 'Time', 'SuccessFlag', 'Error'])
        df_empty.to_csv(output_file, mode='w', index=False)

    # Process each sample
    for i in range(num_samples):
        question = questions[i]  # 用户输入的问题
        ref = [reference_answers[i]]

        print(f"\n$$$$$$ Processing question {i + 1}/{num_samples}: $$$$$$")
        
        time0 = time.time()
        try:
            # Call BackTrack or RuleBase based on the method
            if method == "BackTrack":
                final_answer, success_excute_flag = BackTrack.back_track(question, max_pop, label_dict, label_description_path, entity_extract_example_path, driver, neo4j_database_name, model, top_k)
            elif method == "RuleBase":
                final_answer, success_excute_flag = RuleBase.rule_base(question, max_pop, label_dict, label_description_path, entity_extract_example_path, driver, neo4j_database_name, model, top_k)
            elif method == "Spark":
                final_answer = answer.generate_answer(question, "","spark")
                success_excute_flag = 0
            print(f"Final answer: \n{final_answer}")

            cand = [final_answer]

            # Calculate evaluation metrics
            if metric == "BERTScore":
                P, R, F1 = score(cand, ref, lang="en", verbose=True, model_type="bert-large-uncased", rescale_with_baseline=True)
                P = P.mean().item()
                R = R.mean().item()
                F1 = F1.mean().item()

            print(f"Precision: {P}")
            print(f"Recall: {R}")
            print(f"F1 Score: {F1}")

            total_p += P
            total_r += R
            total_f1 += F1

            error_message = ''  # No error occurred

        except Exception as e:
            # Handle exceptions and log error message
            final_answer = ''
            P, R, F1 = 0, 0, 0
            success_excute_flag = 0
            error_message = str(e)
            print(f"Error occurred: {error_message}")

        time1 = time.time()
        print(f"Time taken: {time1 - time0:.4f}s")

        # Write the current result to the CSV file
        result = {
            'Question': question,
            'Answer': final_answer,
            'Ref': str(ref),
            'P': f"{P:.4f}" if P else '',
            'R': f"{R:.4f}" if R else '',
            'F1': f"{F1:.4f}" if F1 else '',
            'Time': f"{time1 - time0:.4f}",
            'SuccessFlag': success_excute_flag,
            'Error': error_message
        }
        write_results_to_csv(output_file, [result])

    # Calculate and append average scores
    avg_p = total_p / num_samples
    avg_r = total_r / num_samples
    avg_f1 = total_f1 / num_samples

    print(f"\nAverage Precision: {avg_p:.4f}")
    print(f"Average Recall: {avg_r:.4f}")
    print(f"Average F1: {avg_f1:.4f}")

    average_result = {
        'Question': 'Average',
        'Answer': '',
        'Ref': '',
        'P': f"{avg_p:.4f}",
        'R': f"{avg_r:.4f}",
        'F1': f"{avg_f1:.4f}",
        'Time': '',
        'SuccessFlag': '',
        'Error': ''
    }
    write_results_to_csv(output_file, [average_result])

    print("\nEvaluation completed and results saved.")


if __name__ == "__main__":
    uri = "bolt://10.43.108.62:7687"  # Neo4j连接URI
    user = "neo4j"  # 用户名
    password = "12345678"  # 密码
    driver = GraphDatabase.driver(uri, auth=(user, password))  # 创建数据库连接
    output_dir = "./output"  # 测试结果的存储路径，存储为csv文件，会根据上面的参数和当前时间命名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # 指定输出文件名中的时间戳

    # 算法可供选择的参数
    method = "BackTrack"  # 选择要使用的方法，包括"BackTrack"倒推，"RuleBase"基于规则，"Spark"直接使用spark
    max_pop = 5  # 构建推理树时最大的推理跳数
    top_k = 5  # 如果一个实体满足next_label的邻居有多个，最多取top_k个
    generate_answer_model = "spark"  # 选择生成最终答案使用的模型。包括：spark, gpt-4o-mini（提取条件和目的就使用spark，因为便宜，而且效果也还不错）

    # 使用不同数据集需要修改的参数
    test_dataset = "./data/IFLYTEC-NLP/test200/200_QA_多文档.csv"  # 要进行测试的数据集路径，注意是csv格式的
    schema_text_path = "./data/IFLYTEC-NLP/GraphKnowledge/schema.txt"  # 所用知识图谱的关系定义文件，格式是：label-relation-label
    label_dict = back.build_label_dict(schema_text_path)
    label_description_path = "./data/IFLYTEC-NLP/GraphKnowledge/label_description.txt"
    entity_extract_example_path = "./data/IFLYTEC-NLP/GraphKnowledge/entity_extract_example.txt"
    neo4j_database_name = "neo4j"
    df = pd.read_csv(test_dataset)
    questions = df["query_en"]
    reference_answers = df["answer_en"]

    metric = "BERTScore"  # 选择实验的指标，包括：BERTScore
    output_file = create_output_file(output_dir, method, generate_answer_model, max_pop, top_k, timestamp)
    evaluate_and_save_results(df, method, max_pop, top_k, generate_answer_model, label_dict, label_description_path, entity_extract_example_path, driver, neo4j_database_name, output_file, metric, questions, reference_answers)
