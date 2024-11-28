from utils import llm_api as llm


def extract(question):
    """
    输入：用户的问题
    处理过程：使用大模型提取条件实体、目的实体、实体类型。
            处理大模型的答复，为condition_entity, condition_label, aim_entity, aim_label赋值
    输出：conditions:[entity, label], aims:[entity, label]
    """

    query = f"""
    我做的是一个知识图谱增强的知识问答系统。知识图谱是由从论文中提取的要素组成的，节点包括作者姓名，方法，数据集等。
    用户的输入是：{question}
    你需要从用户输入的问题中提取出问题的条件实体及其类型，目的实体及其类型。
    条件是指问题中的已知条件。目的是指问题中想问的东西。
    或许从语义上来讲条件和目的会很复杂，但是你只需要从下表中选择,括号里是解释的内容,实体类型必须是这里面的原词:作者（作者姓名）, 标题（论文标题）, 机构（作者所在机构）, 领域（论文所属领域）, 会议（论文发表的会议或期刊名称）, 关键词（论文关键词）, 问题（论文所解决的问题）, 方法（论文中提出的方法）, 模型（论文中使用的模型的名称）, 任务（论文针对的任务）, 数据集（论文实验所使用的数据集）, 创新点（论文的创新点）, 指标（论文实验所使用的指标）, id（论文在数据库中的id）

    下面是几个例子：
    input:Kausalgie发表过多少篇论文？
    条件实体有:Kausalgie(作者)。目的实体有:papers(标题)
    output:Kausalgie,作者.papers,标题
    
    input:machine translation领域有哪些数据集？
    条件实体有:machine translation(领域)。目的实体有:dataset(数据集)
    output:machine translation,领域.dataset,数据集

    input:SpellGCN的作者是谁？
    条件实体有:SpellGCN (标题)。目的实体有:author(作者)
    output:SpellGCN,标题.author,作者

    input:MindMap和KAG哪个更早？作者分别是谁？
    条件实体有:MindMap (方法);KAG (方法)。目的实体有:earlier(none);author(作者)
    output:MindMap,方法;KAG,方法.earlier,none;author,作者
    
    input:In the field of machine translation, what are the ways to obtain and generate appropriate data sets for low-resource situations?
    条件实体有:machine translation(领域);low-resource situations(问题);obtain and generate appropriate data sets(问题)。目的实体有:ways(方法)
    output:machine translation,领域;low-resource situations,问题;obtain and generate appropriate data sets,问题.ways,方法

    括号里是实体类型。如果从表中找不到就写none
    输出是output:后面的。不需要写"output:"
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