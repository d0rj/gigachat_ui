import gigachat
import streamlit as st
from langchain.chains import ConversationChain
from langchain.chat_models.gigachat import GigaChat
from langchain.memory import ConversationBufferMemory
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)


st.title("GigaChat")

giga_user = st.secrets["GIGACHAT_USER"]
giga_password = st.secrets["GIGACHAT_PASSWORD"]
giga_url = st.secrets["GIGACHAT_BASE_URL"]

with st.sidebar:
    models = (
        gigachat.GigaChat(
            verify_ssl_certs=False,
            user=giga_user,
            password=giga_password,
            base_url=giga_url,
        )
        .get_models()
        .data
    )
    model_name = st.selectbox("Модель", options=[_.id_ for _ in models])
    temperature = st.number_input(
        "Температура", min_value=0.001, max_value=10.0, value=1.0
    )
    top_p = st.number_input("top-p", min_value=0.0, max_value=1.0, value=0.5)
    max_tokens = st.number_input(
        "Максимум токенов в ответе", min_value=1, max_value=1024, value=200
    )
    system_prompt = st.text_area(
        "Системный промпт",
        value="Ты - дружелюбный голосовой ассистент ГигаЧат. Поддерживай приятный диалог с пользователем",
    )
    if st.button("Обновить настройки"):
        st.session_state.clear()

if "chain" not in st.session_state:
    llm = GigaChat(
        model=model_name,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        timeout=10.0,
        verbose=False,
        profanity=False,
        verify_ssl_certs=False,
        user=giga_user,
        password=giga_password,
        base_url=giga_url,
    )
    memory = ConversationBufferMemory(human_prefix="Пользователь", ai_prefix="ГигаЧат")
    chain = ConversationChain(
        llm=llm,
        memory=memory,
        prompt=ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(template=system_prompt),
                HumanMessagePromptTemplate.from_template(
                    template="{history}\nПользователь: {input}",
                    input_variables=["input", "history"],
                ),
            ]
        ),
        verbose=True,
    )
    st.session_state["chain"] = chain

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Новое сообщение..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        responce = st.session_state["chain"].invoke({"input": prompt})
        answer = responce["response"]
        message_placeholder.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
