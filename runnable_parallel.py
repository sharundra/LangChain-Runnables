from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()

prompt = PromptTemplate(template = "Tell me a joke about {subject}",
    input_variables = ["subject"])

parser = StrOutputParser()

prompt2 = PromptTemplate(template = "Tell me a serious fact about {subject}", input_variables = ["subject"])

chain = RunnableParallel({
    "joke": RunnableSequence(prompt, model, parser),
    "fact": RunnableSequence(prompt2, model, parser)
})

result = chain.invoke({"subject": "Religion"})

print(result)