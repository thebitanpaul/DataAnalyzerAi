import streamlit as st
import pandas as pd
from ydata_profiling import ProfileReport
import tempfile
import base64
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import seaborn as sns
import matplotlib.pyplot as plt


# Function to read HTML content
def read_html(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    # Extract text from HTML
    text = soup.get_text(separator=" ")
    return text    


# Load environment variables
load_dotenv()


def main():
    st.title("Data Analyzer Ai")

    tab1, tab2, tab3 = st.tabs(["Data Profile", "Excel Sheet", "Pair Plots"])

    # File uploader
    st.sidebar.header("Upload Data")
    uploaded_file = st.sidebar.file_uploader("Choose a file", type=["xls", "xlsx", "csv"])

    if uploaded_file is None:
        st.info("Please upload a file of type: " + ", ".join(["xls", "xlsx", "csv"]) + " to start analysing your data.")
        st.image("waiting.jpg", use_column_width=True)
        return

    elif uploaded_file is not None:
        st.sidebar.caption("Uploaded file: " + uploaded_file.name)
        

        # Read uploaded file
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
            return

        # Generate ProfileReport if not already generated
        if "profile_report_path" not in st.session_state:
            with st.spinner('Generating Profile Report...'):
                profile = ProfileReport(df, correlations={"auto": {"calculate": False}}, missing_diagrams={"Heatmap": False})
                with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as temp_file:
                    profile.to_file(output_file=temp_file.name)
                    st.session_state.profile_report_path = temp_file.name
               

        # Get HTML content from the generated profile report
        with open(st.session_state.profile_report_path, 'r') as f:
            html_string = f.read()
            text = read_html(html_string)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            length_function=len
        )   

        # Process the text and create the documents list
        documents = text_splitter.split_text(text=text)

        # Vectorize the documents and create vectorstore
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_texts(documents, embedding=embeddings)

        st.session_state.processed_data = {
            "document_chunks": documents,
            "vectorstore": vectorstore,
        }

        # Load the Langchain chatbot
        llm = ChatOpenAI(temperature=0, max_tokens=1000, model_name="gpt-3.5-turbo")
        qa = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever())


        # Display Data Profile under "Data Profile" tab
        with tab1:
            st.title("Data Profile")
            # Add an outline around the HTML content
            styled_html = f'<div style="border: 2px solid orange; padding: 10px;">{html_string}</div>'
            st.components.v1.html(styled_html, height=800, scrolling=True)

        # Display empty Excel tab
        with tab2:
            st.title("Excel")
            st.dataframe(df)

        
        # Initialize Streamlit chat UI
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input(f"Ask ai anything regarding your data?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            result = qa({"question": prompt, "chat_history": [(message["role"], message["content"]) for message in st.session_state.messages]})

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = result["answer"]
                message_placeholder.markdown(full_response + "|")
            message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})  # Adjust height as needed

        # Download button for HTML report
        html_button_sidebar = st.sidebar.button("Download Report")
        if html_button_sidebar:
            progress_bar = st.sidebar.progress(0)
            progress_message = st.sidebar.empty()
            
            progress_message.text("Preparing report for download...")
            progress_bar.progress(10)
            temp_html_file = tempfile.NamedTemporaryFile(delete=False, suffix='.html')
            progress_bar.progress(30)
            temp_html_file.write(html_string.encode())
            progress_bar.progress(40)
            temp_html_file.close()
            progress_bar.progress(60)
            st.sidebar.markdown(f'<a href="data:text/html;base64,{base64.b64encode(open(temp_html_file.name, "rb").read()).decode()}" target="_blank" download="report.html">Click here to download</a>', unsafe_allow_html=True)
            progress_bar.progress(80)
            st.sidebar.info("☝︎ By clicking on this link you can download the report, and open it in your browser.\n\n Then you can press cntrl+p to save the report in a pdf file.")
            progress_bar.progress(100)
            progress_message.text("Download link generated successfully!")


        # Select only numeric columns for pair plot
        numeric_columns = df.select_dtypes(include=['number'])

        # Check for blank columns and exclude them
        non_blank_numeric_columns = numeric_columns.dropna(axis=1)  
        # Display Pair Plots under "Pair Plots" tab
        if not non_blank_numeric_columns.empty:
            with tab3:
                st.title("Pair Plots")
                with st.spinner('Generating Pair Plot...'):
                    try:
                        fig = sns.pairplot(non_blank_numeric_columns)
                        st.pyplot(fig)
                        st.markdown("Pair Plot Generated on these columns: \n\n")
                        st.write(non_blank_numeric_columns)
                    except Exception as e:
                        st.error(f"An error occurred while generating the pair plot: {e}")
                        st.stop()
        else:
            with tab3:
                st.warning("No numeric value column found for pair plot generation.")


if __name__ == "__main__":
    main()
