import streamlit as st
from predict import predict_intent
st.set_page_config(page_title="chat-app")
st.title("ML powered chat app")
st.write("Chat with your data using a large language model (LLM)")


st.title("Simple Chatbot")

user_input = st.text_input("Enter Something...");

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if st.button("send"):
    if(user_input.strip()!=""):
        st.session_state.chat_history.append(("You", user_input))
        prediction = predict_intent(user_input)
        st.write("Prediction:", prediction)
        

        if prediction == "greet":
            response ="Hello! How can I assist you today?"
        elif prediction == "weather":
            response = "The weather is sunny with a chance of rain later in the day."
        else:
            response = "I'm not sure how to respond to that."
    
        st.session_state.chat_history.append(("Bot", response))
for sender, message in st.session_state.chat_history:
    if sender=="You":
        st.markdown(f"**You:** {message}")
    else:
        st.markdown(f"**Bot:** {message}")






    