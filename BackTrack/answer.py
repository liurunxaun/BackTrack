from utils.LLM import openai, spark


def generate_answer(question, reference="", model = "spark"):
    """
    输入：用户的问题，推理出的意图和目的实体
    处理过程：完善query，交给大模型
    输出：最终答案
    """

    query = f"""
    我正在进行一个知识问答任务。
    用户输入了以下问题：
    "{question}"
    
    我可以提供给你一些参考内容，每组内容包括两部分：条件和目的。
    - 条件：问题中的已知信息。
    - 目的：问题中想要解答的目标。

    以下是参考内容：
    {reference}

    请严格依据参考内容回答问题，并进行必要的逻辑推理，生成最终答案。
    **注意**：生成的答案中不得提及或透露参考内容的存在。请用英文回答。
    """

    # System role content
    system_content = """
    You are a helpful and knowledgeable assistant. Your task is to provide precise answers by performing logical reasoning based on the user's input and additional reference content. 
    Remember: You must not disclose or mention the existence of the reference content provided to you in your response.
    """

    # User role content
    user_content = f"""
    I am working on a knowledge-based question-answering task. 
    The user has input the following question:
    "{question}"

    I will provide you with some reference content. Each reference contains two parts: *conditions* and *goals*. 
    - Conditions: Information known from the user's question.
    - Goals: The specific objectives the question aims to answer.

    Here is the reference content:
    {reference}

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