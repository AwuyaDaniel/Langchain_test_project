import os

os.environ['HUGGINGFACEHUB_API_TOKEN'] = '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'

from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain

llm = HuggingFacePipeline.from_model_id(
    model_id="google/flan-t5-large",
    task="text2text-generation",
    model_kwargs={"temperature": 0.6, "max_length": 64},
)

title_template = PromptTemplate(input_variables = ["topic"], template = "write me a youtube vidoe title about {topic}, just a title")
title_template_script = PromptTemplate(input_variables = ["title"], template = "write me a script about {title}")

title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True)
script_chain = LLMChain(llm=llm, prompt=title_template_script, verbose=True)

sq_chain = SimpleSequentialChain(chains=[title_chain, script_chain], verbose=True)

sq_chain.run("Deep Learning")
