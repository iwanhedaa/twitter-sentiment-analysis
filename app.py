import streamlit as st
import re
import numpy as np
import tensorflow as tf
import pickle
from parsivar import Normalizer as ParsivarNormalizer
from parsivar import Tokenizer as ParsivarTokenizer

# Load model and tokenizer
model = tf.keras.models.load_model("saved_lstm_model_with_callbacks.h5")
with open("tokenizer.pkl", "rb") as handle:
    tokenizer = pickle.load(handle)


# Emoji and keyword mappings
emoji_to_persian = {
    '😂': ' خنده ', '🤣': ' خنده ', '😊': ' لبخند ', '😍': ' عشق ', '❤️': ' قلب ', '💕': ' عشق ',
    '👍': ' لایک ', '🔥': ' آتش ', '🙏': ' تشکر ', '🌸': ' گل ', '🎉': ' جشن ',
    '💔': ' قلب_شکسته ', '😔': ' غمگین ', '😞': ' ناامید ', '🙁': ' ناراحت ',
    '🤔': ' تفکر ', '👎': ' دیسلایک ', '😠': ' عصبانی ', '😡': ' عصبانی ', '✅': ' تیک_سبز ',
    '❌': ' ضربدر ', '🌹': ' گل ',
    '😭': ' گریه '
}

positive_keywords_for_cry_emoji = [
    "زیبایی", "زیبا", "خوشبخت", "ناز", "خشکل", "خوش", "خر", "خوشحالی", "جشن", "شادی", "قشنگ",
    "شوق", "خوشحالم", "هیجان", "ذوق", "شاد", "مبارک", "تبریک",
    "عالی", "فوق_العاده", "محشر", "سورپرایز_خوب", "افتخار"
]
negative_keywords_for_cry_emoji = [
    "غم", "ناراحت", "اندوه", "دلتنگ", "افسوس", "متاسفم", "تسلیت", "درد", "داغ",
    "شکست", "باخت", "مصیبت", "فاجعه", "بدبخت", "مرگ", "فوت", "گریه_کنان", "ناراجتی"
]

contractions_dict = {
    "نمیدونم": "نمی دانم", "نمیخوام": "نمی خواهم", "میدونم": "می دانم",
    "نمیتونم": "نمی توانم", "میتونم": "می توانم", "میشه": "می شود",
    "نمیشه": "نمی شود", "خوبه": "خوب است", "بریم": "برویم",
    "اینجوری": "این جوری", "اینا": "این ها", "چیه": "چی است", "کیه": "کی است",
    "میرم": "می روم", "نمیام": "نمی آیم", "میام": "می آیم",
    "حالم خوبه": "حال من خوب است", "حال خوبی ندارم": "حال خوبی ندارم",
    "یه": "یک"
}

normalizer = ParsivarNormalizer()
tokenizer_p = ParsivarTokenizer()

# Preprocessing function
def preprocess_persian_tweet(text):
    if not isinstance(text, str):
        return ""

    for contraction, expansion in contractions_dict.items():
        text = text.replace(contraction, expansion)

    num_cry_emojis = text.count("😭")
    if num_cry_emojis > 0:
        has_positive = any(k in text for k in positive_keywords_for_cry_emoji)
        has_negative = any(k in text for k in negative_keywords_for_cry_emoji)

        replacement = " گریه "
        if has_positive and not has_negative:
            replacement = " اشک_شوق "
        elif has_negative and not has_positive:
            replacement = " گریه_شدید "
        text = text.replace("😭", replacement)

    for emoji_char, persian_keyword in emoji_to_persian.items():
        text = text.replace(emoji_char, persian_keyword)

    text = text.replace("_", " ")
    text = normalizer.normalize(text)
    text = re.sub(r"(.)\1{3,}", r"\1\1\1", text)
    text = re.sub(r"@[^\s]+", "", text)
    text = re.sub(r"#([^\s]+)", r"\1", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(
        r"[^\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\U0001EE00-\U0001EEFF\u200C\s\d_؟!?]",
        "", text, flags=re.UNICODE
    )

    text_to_token = text.replace("_", " ")
    tokens = tokenizer_p.tokenize_words(text_to_token)
    tokens = [t.strip() for t in tokens if len(t.strip()) >= 1]
    return " ".join(tokens)

# Streamlit UI
st.set_page_config(page_title="Persian Emotion Classifier", layout="centered")
st.title("💬 Persian Tweet Emotion Classifier")
st.markdown("این ابزار احساسات موجود در یک توییت فارسی را (بین شادی و غم) تشخیص می‌دهد.")

user_input = st.text_area("یک توییت فارسی وارد کنید:")

if st.button("پیش‌بینی احساس"):
    if not user_input.strip():
        st.warning("لطفاً ابتدا یک متن وارد کنید.")
    else:
        preprocessed = preprocess_persian_tweet(user_input)
        sequence = tokenizer.texts_to_sequences([preprocessed])
        padded = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=50)

        prediction = model.predict(padded)[0][0]
        label = "😢 غم" if prediction >= 0.5 else "😊 شادی"
        st.success(f"احساس شناسایی‌شده: **{label}**")
        st.write(f"📊 احتمال خروجی مدل: `{prediction:.4f}`")
