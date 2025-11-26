from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableBranch, RunnablePassthrough, RunnableLambda
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()

parser = StrOutputParser()

prompt = PromptTemplate(template = "Generate an analysis of {subject}",
    input_variables = ["subject"])

chain1 = RunnableSequence(prompt, model, parser)

prompt2 = PromptTemplate(template = "generate a summary of {topic}", input_variables = ["topic"])


# RunnableBranch is a sort of if-else block of langchain where based on the success of a condition, a specific Runnable gets triggered
conditional_chain = RunnableBranch(
    (lambda x: len(x.split()) > 500, RunnableSequence(prompt2, model, parser)),
    (lambda x: len(x.split()) <= 500, RunnablePassthrough()),
    (RunnableLambda(lambda x:"could not calculate"))
)

final_chain = RunnableSequence(chain1, conditional_chain)

final_chain_result = final_chain.invoke({"subject": "Stanley Kubrik's A clockwork's orange"})

print(final_chain_result)