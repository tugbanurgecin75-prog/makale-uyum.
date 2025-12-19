import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Makale - Hoca Uyumluluk Analizi",
    layout="centered"
)

st.title("📘 Makale ve Hoca Profil Benzerlik Analizi")
st.write("Hoca anahtar kelimeleri ile makale başlığı ve anahtar kelimeleri arasındaki uyumu hesaplar.")

@st.cache_resource
def load_model():
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

model = load_model()

hoca_keywords = st.text_area(
    "👨‍🏫 Hoca Anahtar Kelimeleri (virgülle ayır)",
    placeholder="machine learning, deep learning, computer vision, pose estimation, mmpose"
)

makale_title = st.text_input(
    "📄 Makale Başlığı",
    placeholder="A deep learning approach for human pose estimation"
)

makale_keywords = st.text_area(
    "🏷️ Makale Anahtar Kelimeleri (virgülle ayır)",
    placeholder="deep learning, pose estimation, cnn, mmpose"
)

if st.button("🔍 Benzerlik Hesapla"):
    if not hoca_keywords or not makale_title:
        st.warning("Lütfen hoca anahtar kelimeleri ve makale başlığını girin.")
    else:
        profile_text = " ".join([k.strip() for k in hoca_keywords.split(",")])
        article_text = makale_title + " " + " ".join([k.strip() for k in makale_keywords.split(",")])

        vec_profile = model.encode([profile_text])
        vec_article = model.encode([article_text])

        score = cosine_similarity(vec_article, vec_profile)[0][0]

        st.success(f"🔢 Benzerlik Skoru: {score:.4f}")

        if score > 0.75:
            st.write("🟢 **Çok yüksek uyum**")
        elif score > 0.5:
            st.write("🟡 **Orta düzey uyum**")
        else:
            st.write("🔴 **Düşük uyum**")
