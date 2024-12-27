from utils.LLM import openai, spark


def generate_answer(question, last_node_str = "", reasoning_path_str = "", model = "spark"):
    """
    输入：用户的问题，推理出的意图和目的实体
    处理过程：完善query，交给大模型
    输出：最终答案
    """

    query = f"""
        I am working on a knowledge-based question-answering task.
        The user has input the following question:
        "{question}"

        I can provide you with some reference content, where each set of content consists of two parts: conditions and objectives.
        - Conditions: The known information from the question.
        - Objectives: The goals that the question seeks to address.
        Below is the conditions and objectives:
        {last_node_str}
        
        I can also provide you the complete reasoning paths which maybe useful for you. I wish you cloud utilize your reasoning ability to answer users' question
        - Each path is described by nodes and edges in the following format: [Entity Type] Entity Name -> (Relation) [Entity Type] Entity Name -> ...
        - Each node includes [Entity Type] Entity Name.
        - Each edge is represented by an arrow ->, with the edge information enclosed in parentheses, e.g., (Relation).
        - Starting from the root node, the path is described step by step, including nodes and their relationships, until reaching the leaf node.
        Below is the reasoning paths:
        {reasoning_path_str}

        Please strictly follow the reference content to answer the question, applying logical reasoning as needed to generate the final answer.
        **Note**: The generated answer must not mention or disclose the existence of the reference content.
    """

    # OpenAI System role content
    system_content = """
    You are a helpful and knowledgeable assistant. Your task is to provide precise answers by performing logical reasoning based on the user's input and additional reference content. 
    Remember: You must not disclose or mention the existence of the reference content provided to you in your response.
    """

    # OpenAI User role content
    user_content = f"""
    I am working on a knowledge-based question-answering task. 
    The user has input the following question:
    "{question}"

    I can provide you with some reference content, where each set of content consists of two parts: conditions and objectives.
    - Conditions: The known information from the question.
    - Objectives: The goals that the question seeks to address.
    Below is the conditions and objectives:
    {last_node_str}
        
    I can also provide you the complete reasoning paths which maybe useful for you. I wish you cloud utilize your reasoning ability to answer users' question
    - Each path is described by nodes and edges in the following format: [Entity Type] Entity Name -> (Relation) [Entity Type] Entity Name -> ...
    - Each node includes [Entity Type] Entity Name.
    - Each edge is represented by an arrow ->, with the edge information enclosed in parentheses, e.g., (Relation).
    - Starting from the root node, the path is described step by step, including nodes and their relationships, until reaching the leaf node.
    Below is the reasoning paths:
    {reasoning_path_str}

    Please strictly follow the reference content to answer the question. Use logical reasoning if necessary to generate the final answer. 
    **Note**: The generated answer must not reveal or mention the existence of the reference content.
    """

    if model == "spark":
        print(f"query:\n{query}")
        return spark.spark_4_0(query)
    elif model == "gpt-4o-mini":
        print(f"system_content:\n{system_content}")
        print(f"user_content:\n{user_content}")
        return openai.gpt_4o_mini(system_content, user_content)
    else:
        return "没有这个模型。现在可选的模型有: spark, gpt-4o-mini"