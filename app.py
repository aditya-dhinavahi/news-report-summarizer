import streamlit as st
import time
from datetime import datetime
import pandas as pd

from gnews import GNews
from dotenv import load_dotenv

from langchain.callbacks import get_openai_callback

from llm_functions import get_text_from_url, get_text_chunks, get_text_summary

load_dotenv()

def main():
    
    st.set_page_config(page_title = "News article summarizer",
                       page_icon = ":newspaper:", layout = "wide")
 
    st.header("Summarize news articles")
    st.text('''This app helps you with following steps - \n
            1. Search for news articles from Google News using keywords. 
            2. Select articles to summarize from step 1.
            3. When satisfied, download summary of selected articles as a text file.''')
    
    google_search_help_url = "https://ahrefs.com/blog/google-advanced-search-operators/"
    st.write("Check out this [link](%s) for help with Google search operators" %google_search_help_url)

    form = st.form(key = "my_form")
    
    keywords = form.text_input("#### Enter keywords to search for", 
                                    value = 'interview (MD OR CEO OR CFO) after:2023-07-01 country:india -results')

    if 'stage' not in st.session_state:
        st.session_state.stage = 0

    def set_state(i):
        st.session_state.stage = i

    submit = form.form_submit_button('Search', on_click = set_state, args = [1])

    if st.session_state.stage == 1:
        response = pd.DataFrame()
        with st.spinner("Searching for articles..."):
            
            google_news = GNews(language='en', max_results = 100)
            # google_news.topic = "Business"
            
            @st.cache_data
            def get_news_func(_gnews_object, kwds):
                return _gnews_object.get_news(kwds)

            response = get_news_func(google_news, keywords)
            response = pd.DataFrame(response)

        response = response.loc[ :, ["published date", "publisher", "title", "url"]]
        
        response.columns = ["Date", "Publisher", "Title", "URL"]
        response["Publisher"] = response["Publisher"].apply(lambda x: x["title"])
        response["Date"] = [datetime.strptime(dt_string, "%a, %d %b %Y %H:%M:%S %Z") for dt_string in response["Date"]]
        response["Date"] = [datetime.strftime(dt, "%d-%m-%Y") for dt in response["Date"]]


        def make_clickable(row):
            text = row['Title']
            link = row["URL"]
            row["Title"] = f'<a target="_blank" href="{link}">{text}</a>'
            return row

        st.write('#### News articles found, select the ones you want to summarize')
    
        response = response.loc[:, ["Date", "Publisher", "Title", "URL"]]

        edited_df = response.copy()
        edited_df.insert(0, "Select", False)

        edited_df = st.data_editor(
            edited_df,
            hide_index=True,
            width = 2000,
            column_config={"Select": st.column_config.CheckboxColumn(required=True),
                            "URL": st.column_config.LinkColumn("URL", required=True)},
            disabled = response.columns,
            on_change = set_state,
            args = [2],           
        )

    if st.session_state.stage == 2:
            
        selected_rows = edited_df[edited_df.Select]
        selected_rows.reset_index(drop = True, inplace = True)
        # selected_rows = grid["selected_rows"]
        selected_rows = pd.DataFrame(selected_rows)
        selected_rows.reset_index(drop = True, inplace = True)

        st.write("Your selection:")
        st.dataframe(selected_rows, width = 1500, hide_index = True)

        summarize_button = st.button ("Click to summarize selected articles", on_click = set_state, args = [3])

    if st.session_state.stage == 3:

        st.write('#### Selected news items and their summaries')

        with get_openai_callback() as cb:
            with st.spinner("Summarizing articles..."):

                article_counter = 1
                output_text = ""
                for i in range(0, selected_rows.shape[0]):
                    article_header = str(article_counter) + ".  " + selected_rows.loc[i, "Title"]
                    st.write(article_header)
                    output_text = output_text + article_header + "\n"
                    try:
                        text = get_text_from_url(selected_rows.loc[i, "URL"])
                        text_chunks = get_text_chunks(text)
                        text_summary = get_text_summary(text_chunks)
                        output_text = output_text + str(text_summary) + "\n"
                        st.write(str(text_summary))
                        output_url = f"""[Click here to read in detail...]({selected_rows.loc[i, "URL"]})"""
                        output_text = output_text + output_url + "\n \n"
                        article_counter += 1
                    except Exception as e:
                        st.write("Error in summarizing article")
                        st.write(e)
                print(cb)    

        st.write("Thanks for using the app \n Total cost of GPT API calls : Rs. " + str(round(cb.total_cost*82, 0)))

        download = st.download_button(label = "Click to download summary", data = output_text, 
                                file_name = "news-summary.txt", mime = "text/plain", 
                                on_click = set_state, args = [4])
        
                
    if st.session_state.stage >= 4:
        st.write("Goodbye!")
                    

if __name__ == '__main__':
    main()