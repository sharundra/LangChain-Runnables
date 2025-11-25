from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence

load_dotenv()

model = ChatOpenAI()

prompt = PromptTemplate(template = "Tell me a joke about {subject}",
                        input_variables = ["subject"])

parser = StrOutputParser()

prompt2 = PromptTemplate(template = "Explain the {topic}", input_variables = ["topic"])

chain = RunnableSequence(prompt, model, parser, prompt2, model, parser)

result = chain.invoke({"subject": "Hollywood"})

print(result)
