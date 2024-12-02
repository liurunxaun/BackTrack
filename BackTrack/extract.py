from utils.LLM import spark as llm


def extract(question):
    """
    输入：用户的问题
    处理过程：使用大模型提取条件实体、目的实体、实体类型。
            处理大模型的答复，为condition_entity, condition_label, aim_entity, aim_label赋值
    输出：conditions:[entity, label], aims:[entity, label]
    """

    query = f"""
我在做一个知识图谱增强的知识问答系统，知识图谱由从论文中提取的要素组成，节点包括作者姓名、方法、数据集等。
用户的输入是：{question}
你需要从用户输入的问题中提取出条件实体及其类型，目的实体及其类型。
条件是问题中的已知信息，目的是问题中想要问的内容。

请从以下表格中选择实体及其类型：
- 作者（作者姓名）
- 标题（论文标题）
- 机构（作者所在机构）
- 领域（论文所属领域）
- 会议（论文发表的会议或期刊名称）
- 关键词（论文关键词）
- 问题（论文所解决的问题）
- 方法（论文中提出的方法）
- 模型（论文中使用的模型名称）
- 任务（论文针对的任务）
- 数据集（论文实验所使用的数据集）
- 创新点（论文的创新点）
- 指标（论文实验所使用的指标）
- id（论文在数据库中的id）

如果表格中没有匹配项，请使用"none"。条件实体和目的实体之间用"."分隔。

以下是几个示例：
示例1:
输入: Kausalgie发表过多少篇论文？
条件实体: Kausalgie（作者）
目的实体: papers（标题）
输出: Kausalgie,作者.papers,标题

示例2:
输入: machine translation领域有哪些数据集？
条件实体: machine translation（领域）
目的实体: dataset（数据集）
输出: machine translation,领域.dataset,数据集

示例3:
输入: SpellGCN的作者是谁？
条件实体: SpellGCN（标题）
目的实体: author（作者）
输出: SpellGCN,标题.author,作者

示例4:
输入: MindMap和KAG哪个更早？作者分别是谁？
条件实体: MindMap（方法）；KAG（方法）
目的实体: earlier（none）；author（作者）
输出: MindMap,方法;KAG,方法.earlier,none;author,作者

示例5:
输入: In the field of machine translation, what are the ways to obtain and generate appropriate data sets for low-resource situations?
条件实体: machine translation（领域）；low-resource situations（问题）；obtain and generate appropriate data sets（问题）
目的实体: ways（方法）
输出: machine translation,领域;low-resource situations,问题;obtain and generate appropriate data sets,问题.ways,方法

请注意：条件和目的实体之间用"."隔开，输出时不需要额外的标识符，只需输出实体类型。
"""

    response = llm.spark_4_0(query)
    print(f"大模型返回:\n{response}\n")

    conditions = []
    aims = []

    try:
        split = response.split(".")

        split1 = split[0].split(";")
        for item in split1:
            item_split = item.split(",")
            conditions.append([ item_split[0], item_split[1] ]) # 0是实体，1是实体类型

        split2 = split[1].split(";")
        for item in split2:
            item_split = item.split(",")
            aims.append([ item_split[0], item_split[1] ]) # 0是实体，1是实体类型
    except:
        return [],[]

    return conditions, aims