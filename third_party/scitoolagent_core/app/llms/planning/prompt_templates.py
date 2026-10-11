def generate_plan_input_template(question, current_output, tool_name, input_type, output_type, previous_answer):
    return f"""
    You are the information extractor, and your task is to extract the incoming parameters from the currently provided information as input to the tool function.

    This is the reference information for you
    question: {question} 
    tool_name: {tool_name}
    input_type:{input_type}
    output_type: {output_type}
    current_output: {current_output}
    previous_answer: {previous_answer}

    current_output indicates the output of an existing tool, and previous_answer indicates the existing answer to the problem, tool_name is the name of the tool function that needs to be used, input_type is the input parameter type of the tool function, and output_type is the output type of the tool function.

    Before extracting parameters, you first need to check "input_type". If "input_type" contains "pair", such as "smiles pair", "protein sequence pair", it means that multiple parameters need to be extracted. Use between different parameters. '.' separated. If "input_type" does not contain "pair", such as "molecule name", "protein name", only one parameter can be extracted.

    Note: When "input_type" does not contain "pair", but the provided Description contains multiple parameters, you must strictly follow the rules, that is, you can only extract one of the parameters, but not multiple parameters. At this time, you need to make a judgment based on the results currently returned in Description. For example, if you analyze that there are parameters A and B, but the processing result of A already exists in Description, you only need to extract B at this time, otherwise extract A. In short, if "input_type" does not contain "pair", then only one of the unprocessed parameters needs to be extracted.

    You need to combine the input type of the tool function and extract the correct input parameters of the tool function from the description document. It is then given strictly as follows:
    Input:your answer

    Note that your answer should strictly match the input parameter type and need not contain other information. For example, if you need to extract information about smiles, your answer should be "input:CCO" and be sure not to include any additional information.
    """

def generate_output_template(question, output):
    return f"""
    Based on the information provided, you need to determine whether my problem has been solved and give an explanation.

    This is my question:
    question:{question}

    This is the reference content: 
    output:{output}

    Please give your answer.

    Your answer should contain two parts. The first part tells me whether the problem has been solved. If it has been solved, give the corresponding answer; if it has not been solved, give the reason.

    Your output should be given strictly in the following format:

    IsCompleted:Completed or Not Completed\n

    FinalAnswer:Your answer\n

    Note: Before answering, you need to think about whether the question involves Security issues. 
    If it does not involve security issues, you should answer normally. 
    Otherwise, you only need to reply the content of the Security Warning, and you do not need to reply other content. And the IsCompleted part should be Completed at this time
    """

def generate_tools_plan_template(question, tools_retrieved):
    return f"""
    You are a professional planner. Now you need to develop a plan chain to complete the task based on the task description and corresponding reference information I provide.

    And your answer should strictly follow the following format:
    - A detailed breakdown of each sub-task.
    - The name of the tool required for each sub-task.
    - A plan chain showing all tools in order, clearly distinguishing between tools used for similar sub-tasks.

    The tools and their sequence should be formatted as:
    Plan Chain:
    ['tool_A', 'tool_B', 'tool_C', ...]

    Note:
    1.If the same tool is required multiple times for different inputs, it should appear as many times as needed in the plan chain.
    2.Your answer must include "Plan Chain:". And the tools part of it must be from existing tools, not made up.
    3.No comments need to be added in the Plan Chain: 

    Next, start your attempt:
    This is the task question:
    {question}

    This is the reference information provided to you:
    {tools_retrieved}

    Please give your answer.
    """

def query_next_tool_template(question, triplet):
    return f"""
    Your task is to tell me what needs to be done to solve this problem based on the existing information.

    I will give you the problem to be solved:
    {question}

    The existing triple information:
    {triplet}

    You need to analyze the problem and the given information and tell me what needs to be done.

    Note:
    1. You only need to answer the steps that need to be done later. You do not need to include the existing triple information or reflect the thinking process. You only need to tell me what needs to be done. The answer should be as concise as possible.
    2. The existing triple information may be the first step to solve this problem, or it may be the subsequent step.
    3. The existing triple information may also be irrelevant to solving this problem. At this time, you need to think about what needs to be done.
    """
