import json
import torch
import numpy as np
import faiss
import joblib
from transformers import AutoTokenizer, AutoModel
import unicodedata
import re
from fuzzywuzzy import fuzz

# Load intent → response mapping
with open("../Data/intent_response_mapping.json", "r", encoding="utf-8") as f:
    INTENT_RESPONSE_MAP = json.load(f)

# Load dữ liệu sản phẩm
with open("../Data/processed/AI_processed_product_details_vn.json", "r", encoding="utf-8") as f:
    PRODUCT_DATA = json.load(f)

# Load PhoBERT
print("🔄 Đang tải PhoBERT...")
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
model = AutoModel.from_pretrained("vinai/phobert-base-v2")

def get_embedding(text: str) -> np.ndarray:
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy().astype('float32')

# Load FAISS index + metadata (FAQ Matching)
print("🔄 Đang tải FAISS index FAQ...")
faiss_index = faiss.read_index("../Models/faiss_index/faqs.index")

with open("../Data/embedded/AI_processed_faqs_knowledges_with_embeddings.json", "r", encoding="utf-8") as f:
    faq_data = json.load(f)

faq_questions = [item['original_item']['Câu hỏi'] for item in faq_data]
faq_answers = [item['original_item']['Câu trả lời'] for item in faq_data]

# Load Intent Classifier + Label Encoder
print("🔄 Đang tải mô hình phân loại Intent...")
intent_model = joblib.load("../Models/intent_classifier/mlp_intent_classifier.joblib")
label_encoder = joblib.load("../Models/intent_classifier/label_encoder.joblib")

def predict_intent(text: str):
    emb = get_embedding(text).reshape(1, -1)
    probs = intent_model.predict_proba(emb)[0]
    best_idx = np.argmax(probs)
    best_intent = label_encoder.inverse_transform([best_idx])[0]
    confidence = probs[best_idx]
    return best_intent, confidence

def search_faq(text: str, top_k=1):
    emb = get_embedding(text).reshape(1, -1)
    D, I = faiss_index.search(emb, top_k)
    results = []
    for idx in I[0]:
        results.append({
            "question": faq_questions[idx],
            "answer": faq_answers[idx]
        })
    return results

def normalize(text):
    text = unicodedata.normalize("NFKD", text).lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()

def normalize_product_name(name: str) -> str:
    name = unicodedata.normalize("NFKC", name).lower()
    name = re.sub(r"[^\w\s]", "", name)
    name = re.sub(r"\b(máy ảnh|kỹ thuật số|mirrorless|dslr|fullframe|body|kit|ống kính|lens|máy quay|only|with)\b", "", name)
    return re.sub(r"\s+", " ", name).strip()

def handle_ask_price(user_input: str) -> str:
    user_input_norm = normalize_product_name(user_input)
    best_score = 0
    best_product = None

    for product in PRODUCT_DATA:
        product_norm = product.get("normalized_name", "")
        score = fuzz.token_set_ratio(user_input_norm, product_norm)
        if score > best_score:
            best_score = score
            best_product = product

    if best_product and best_score >= 70:
        name = best_product.get("product_name", "Không rõ")
        price = best_product.get("price", "Không rõ")
        return f"Giá của {name} là {price}."
    return "Tôi chưa tìm thấy sản phẩm phù hợp. Bạn vui lòng kiểm tra lại tên nhé."

def handle_ask_compare(user_input: str) -> str:
    user_input_norm = normalize(user_input)
    matches = []
    for product in PRODUCT_DATA:
        name = product.get("normalized_name", "")
        if name in user_input_norm:
            matches.append(product)

    if len(matches) < 2:
        return "Vui lòng cung cấp ít nhất 2 tên sản phẩm để tôi so sánh giúp bạn."

    p1, p2 = matches[0], matches[1]
    COMPARE_FIELDS = {
        "price": "Giá",
        "sensor": "Cảm biến",
        "brand": "Hãng sản xuất",
        "color": "Màu sắc",
        "camera_type": "Loại máy ảnh",
        "focus_mode": "Chế độ lấy nét",
        "image_resolution": "Độ phân giải ảnh"
    }

    result = f"So sánh {p1['product_name']} và {p2['product_name']}:\n"
    for key, label in COMPARE_FIELDS.items():
        val1 = p1.get(key, "Không rõ")
        val2 = p2.get(key, "Không rõ")
        result += f"- {label}:\n  • {p1['product_name']}: {val1}\n  • {p2['product_name']}: {val2}\n"
    return result

def handle_ask_spec(user_input: str) -> str:
    user_input_norm = normalize(user_input)
    for product in PRODUCT_DATA:
        name = product.get("normalized_name", "")
        if name in user_input_norm:
            info = product.get("specifications", "Không rõ")
            return f"Thông số kỹ thuật của {product.get('product_name', '')}:\n{info}"
    return "Bạn vui lòng cung cấp tên sản phẩm cụ thể để tôi hiển thị thông số."

INTENT_FUNCTION_MAP = {
    "ask_price": handle_ask_price,
    "ask_compare": handle_ask_compare,
    "ask_spec": handle_ask_spec,
    "ask_feature": handle_ask_spec
}

def chatbot_response(user_input: str, intent_confidence_threshold=0.6, faq_score_threshold=0.75):
    # 1. Thử tìm câu hỏi phù hợp trong FAISS
    emb = get_embedding(user_input).reshape(1, -1)
    D, I = faiss_index.search(emb, 1)
    best_score = 1 - D[0][0] / 2  # FAISS dùng inner product / L2 → ta chuyển thành [0,1] cho dễ hiểu

    if best_score >= faq_score_threshold:
        idx = I[0][0]
        question = faq_questions[idx]
        answer = faq_answers[idx]
        print(f"📚 Match FAQ: \"{question}\" (Score: {best_score:.2f})")
        return answer

    # 2. Nếu không có match tốt, chuyển sang Intent Classifier
    intent, confidence = predict_intent(user_input)
    print(f"🧠 Intent: {intent} (Confidence: {confidence:.2f})")

    if confidence >= intent_confidence_threshold:
        if intent in INTENT_FUNCTION_MAP:
            return INTENT_FUNCTION_MAP[intent](user_input)
        elif intent in INTENT_RESPONSE_MAP:
            return INTENT_RESPONSE_MAP[intent]

    return INTENT_RESPONSE_MAP.get("fallback", "Tôi không hiểu ý bạn.")




if __name__ == "__main__":
    print("🤖 CamNest Chatbot sẵn sàng! (gõ 'exit' để thoát)\n")
    while True:
        user_input = input("👤 Bạn: ")
        if user_input.lower() in ['exit', 'quit']:
            break
        reply = chatbot_response(user_input)
        print(f"🤖 Bot: {reply}\n")
