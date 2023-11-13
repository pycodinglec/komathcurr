import openai
import streamlit as st
import os
import pickle
import pandas as pd
import numpy as np
from scipy import spatial
import hashlib

openai.api_key = os.environ.get("OPENAI_API_KEY")
GPT_MODEL = 'gpt-4'
EMBEDDING_MODEL = 'text-embedding-ada-002'
EMBEDDING_FOLDER = 'documents_embed'

def initialize_conversation():
    with open("system_message.txt", "r", encoding="utf-8") as f:
        system_message = f.read()
    hello_message = '👋안녕하세요 선생님, 무엇을 도와드릴까요?'
    system = {'role':'system', 'content': system_message}
    hello = {'role':'assistant', 'content': hello_message}
    msgs = [system, hello]
    return msgs

def initialize_documents_embedding():
    all_data = []
    embedding_files = [file for file in os.listdir(EMBEDDING_FOLDER) if file.endswith('pkl')]
    for embedding_file in embedding_files:
        with open(f'{EMBEDDING_FOLDER}/{embedding_file}', 'rb') as f:
            df2 = pickle.load(f)
            all_data.append(df2)
    
    df = pd.concat(all_data, ignore_index=True)
    return df

def get_modified_prompt(original_prompt):
    if 'df' not in st.session_state:
        st.session_state['df'] = initialize_documents_embedding()
    df = st.session_state['df']
    response = openai.Embedding.create(
                model = EMBEDDING_MODEL,
                input = original_prompt,
    )
    prompt_embedding = response["data"][0]["embedding"]
    df['similarity'] = df['embedding'].apply(lambda x: np.nan_to_num(1 - spatial.distance.cosine(prompt_embedding, x), nan=0))
    df = df.sort_values(by='similarity', ascending=False)
    modified_prompt = f'''근거 자료를 줄 테니까 질문에 대답해. 만약 질문에 관련된 내용을 근거 자료에서 찾지 못하겠다면, '관련 내용을 찾을 수 없다'고 답하면 돼.
    질문: {original_prompt}
    근거 자료: {df.text[:15]}'''
    
    return [{'role': 'user', 'content': modified_prompt}]

def verify_password(input_password):
    correct_password_hash = os.getenv('PASSWORD_HASH')
    password_hash = hashlib.md5(input_password.encode()).hexdigest()
    return password_hash == correct_password_hash

def chatbot_page():
    st.title('KoMathCurr: 한국 수학 교육과정 질의 챗봇')
    st.caption('선생님을 위하여 수학 교육과정을 검색합니다.')
     
    if 'msgs' not in st.session_state:
        st.session_state['msgs'] = initialize_conversation()
    if 'df' not in st.session_state:
        st.session_state['df'] = initialize_documents_embedding()

    for msg in st.session_state['msgs'][1:]:
        with st.chat_message(msg['role']):
            st.markdown(msg['content'])
    
    if prompt:= st.chat_input("이곳에 요청을 입력"):
        st.session_state['msgs'].append({'role':'user', 'content': prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            modified_msgs = st.session_state['msgs'][:-1] + get_modified_prompt(st.session_state['msgs'][-1]['content'])
            responses = openai.ChatCompletion.create(
                model = GPT_MODEL,
                messages = modified_msgs,
                stream = True,
            )
            for response in responses:
                full_response += response.choices[0].delta.get("content", "")
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        st.session_state['msgs'].append({"role": "assistant", "content": full_response})

def main():
    with st.container():
        input_password = st.text_input("패스워드를 입력하세요", type="password")
        if st.button("로그인"):
            if verify_password(input_password):
                st.session_state['authenticated'] = True
            else:
                st.error("패스워드가 잘못되었습니다.")

    if st.session_state.get('authenticated', False):
        chatbot_page()


if __name__=='__main__':
    main()
