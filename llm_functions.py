from dotenv import load_dotenv

from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from newspaper import Article

load_dotenv()

llm = ChatOpenAI(temperature=0, model = "gpt-3.5-turbo")

def get_text_from_url(url):
                article = Article(url)
                article.download()
                article.parse()
                text = article.text
                # print(llm.get_num_tokens(text))
                # print(text[0:1000])
                return article.text
            
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 2000,
    chunk_overlap = 500,
    length_function = len
)
    # chunks = text_splitter.split_text(text)
    chunks = text_splitter.create_documents([text])
    # print(chunks)
    return chunks

def get_text_summary(text_chunks):

    map_prompt = '''
    Write a summary the text below delimited by <>
    Rules - 
    - Keep the length of the summary to max 160 words.
    - Retain the name of person, designation and company in the summary.
    - Keep the summary in format <title>, <person interviewed, designation, company>: <text summary>

    Text to summarize - <{text}>
    '''
    map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])

    combine_prompt =  """
    Summarize the content delimited by <> into a single paragraph and output the content with schema <title>, <person interviewed, designation, company>: <text summary>.
    Example output is given below delimited by ///
    Rules - 
    - Output of the summary should in format <title>, <person interviewed, designation, company>: <text summary>
    - Keep the length of output summary to max 160 words.
    ///
    ‘Indian banking sector is set for a golden decade amid growing digicalisation’, Uday Kotak: MD Kotak Bank: details that the 
    financial sector landscape in India is at an interesting juncture, having witnessed global banking challenges and the 
    importance of stability. Despite past turbulence, India's financial sector has emerged stronger, but sustained growth 
    while prioritizing stability and sustainability is crucial.  He believes policymakers should focus on capacity-building 
    and though public sector banks have shown recovery, a larger capacity is needed to support India's growing economy. 
    He details the importance of technology in banking's future emphasizes that banks must balance legacy technology 
    with innovation. He cautions that risks and returns should be adequately priced with PLI scheme as companies to 
    scale, supporting India Inc.'s growth.
    ///
    Text to check on - <{text}>
    """
    combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["text"])

    # template = template = """
    # Please write 150 words summary of the following text
    # Keep name of person interviewed, designation, company and what they said in the summary.

    # {text}
    # """
    # prompt = PromptTemplate(input_variables = ["text"], template = template)
    # summary_chain1 = load_summarize_chain(llm, chain_type = "stuff", prompt = prompt)

    # all_text = ""
    # for chunks in text_chunks:    
    #     summ = summary_chain1.run(chunks)
    #     all_text += summ

    # text = get_text_chunks(all_text)

    summary_chain = load_summarize_chain(llm, chain_type = "map_reduce", 
                                        map_prompt=map_prompt_template, 
                                        combine_prompt=combine_prompt_template)
    
    summary = summary_chain.run(text_chunks)
    return summary
