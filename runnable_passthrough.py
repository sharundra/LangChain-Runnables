from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnablePassthrough, RunnableParallel
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()

parser = StrOutputParser()

prompt = PromptTemplate(template = "Tell me a joke about {subject}",
    input_variables = ["subject"])

prompt2 = PromptTemplate(template = "Give me a serious fact about {topic}", input_variables = ["topic"])


# RunnablePassthrough is a Runnable that simply returns the input as it is.
chain1 = RunnableSequence(prompt, model, parser)
chain2 = RunnableParallel({
    "joke": RunnablePassthrough(),
    "fact": RunnableSequence(prompt2, model, parser)
})

final_chain = RunnableSequence(chain1, chain2)

result = final_chain.invoke({"subject": "Charlie Chaplin"})

print(result)