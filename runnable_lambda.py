from langchain_openai import ChatOpenAI 
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableLambda, RunnableParallel, RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()

prompt = PromptTemplate(template = "Tell me a joke about {subject}",
    input_variables = ["subject"])

parser = StrOutputParser()
chain1 = RunnableSequence(prompt, model, parser)


# RunnableLambda is used to convert normal functions into a Runnable so that they can be part of longer chain formations.
chain2 = RunnableParallel({
    "joke": RunnablePassthrough(),
    "count of words": RunnableLambda(lambda x: len(x.split()))
})

final_chain = RunnableSequence(chain1, chain2)

print(final_chain.invoke({"subject": "Mathematics"}))

