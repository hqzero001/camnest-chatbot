import streamlit as st
from chatbot import chatbot_response  # dùng chatbot.py bạn đã có

st.set_page_config(page_title="CamNest Chatbot", page_icon="📸", layout="centered")

st.title("🤖 CamNest AI Chatbot")
st.markdown("Chào bạn! Tôi là trợ lý AI hỗ trợ tư vấn sản phẩm và dịch vụ của CamNest. Hãy đặt câu hỏi của bạn bên dưới.")

# Tạo session lưu lịch sử chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Hiển thị lịch sử
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Nhận tin nhắn người dùng
user_input = st.chat_input("Bạn muốn hỏi gì?")
if user_input:
    # Hiển thị tin nhắn người dùng
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Phản hồi từ chatbot
    with st.chat_message("assistant"):
        with st.spinner("Đang trả lời..."):
            reply = chatbot_response(user_input)
            st.markdown(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})
