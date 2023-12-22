from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
import os
from langchain.llms import HuggingFacePipeline

os.environ['HUGGINGFACEHUB_API_TOKEN'] = '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'

llm = HuggingFacePipeline.from_model_id(
    model_id="google/flan-t5-large",
    task="text2text-generation",
    model_kwargs={"temperature": 0.6, "max_length": 64},
)
# Template
title_template = PromptTemplate(input_variables=["topic"],
                                template="write me a youtube vidoe title about {topic}, just a title")
title_template_script = PromptTemplate(input_variables=["title"], template="write me a script about {title}")

# Memory
memory = ConversationBufferMemory(input_key='topic', memory_key="chat_history")

# Llms
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key="title", memory=memory)
script_chain = LLMChain(llm=llm, prompt=title_template_script, verbose=True, output_key="script", memory=memory)
sq_chain = SequentialChain(chains=[title_chain, script_chain], verbose=True, input_variables=['topic'],
                           output_variables=['title', 'script'])
sq_chain({'topic': "Deep Learning"})
print(memory.buffer)
