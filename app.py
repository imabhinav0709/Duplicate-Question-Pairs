import os
import streamlit as st
import helper
import pickle

st.set_page_config(page_title="Quora Duplicate Question Detector", page_icon="❓", layout="centered")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'model.pkl')

@st.cache_resource
def load_model():
    with open(model_path, 'rb') as f:
        return pickle.load(f)

model = load_model()

st.title('❓ Duplicate Question Pairs Detector')
st.markdown('Predict whether two Quora questions ask the same thing or share the same intent.')

with st.sidebar:
    st.header('⚙️ Detection Settings')
    threshold = st.slider(
        'Base Similarity Threshold',
        min_value=0.30,
        max_value=0.70,
        value=0.45,
        step=0.01,
        help='Quora datasets have ~37% positive duplicate prior. A baseline of ~0.45 balances duplicate detection.'
    )
    entity_check = st.checkbox('Enable Entity & Keyword Conflict Guard', value=True, help='Prevents template overlap false positives when distinct subjects (e.g. India vs USA, Java vs Python) are detected.')
    st.markdown('---')
    st.markdown('**Model Info:**\n- Classifier: Random Forest\n- Features: 22 Statistical (Token, Length, Fuzzy) + BoW')

q1 = st.text_input('Enter Question 1:', placeholder='e.g. what is capital of india')
q2 = st.text_input('Enter Question 2:', placeholder='e.g. what is current capital of india')

if st.button('🔍 Find Similarity', use_container_width=True):
    if not q1.strip() or not q2.strip():
        st.warning('Please enter both questions to compare.')
    else:
        with st.spinner('Analyzing question similarity and semantic intent...'):
            query = helper.query_point_creator(q1, q2)
            prob_dup = float(model.predict_proba(query)[0][1])
            kw_info = helper.analyze_keywords(q1, q2)
            
            # Determine if duplicate
            if entity_check and kw_info['has_subject_conflict']:
                is_duplicate = False
                reason = f"Distinct core subjects detected ({', '.join(kw_info['diff1'])} vs {', '.join(kw_info['diff2'])})."
            else:
                is_duplicate = prob_dup >= threshold
                reason = "Questions ask the same thing." if is_duplicate else "Different questions or intents."

            st.markdown('### 📊 Result')
            st.progress(prob_dup, text=f'Statistical Model Duplicate Probability: {prob_dup * 100:.1f}%')

            if is_duplicate:
                st.success(f'✅ **Duplicate Questions** (Confidence: {prob_dup * 100:.1f}%)')
            else:
                st.error(f'❌ **Not Duplicate** — {reason}')

            with st.expander('🔍 Detailed Keyword & Feature Breakdown'):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric('ML Model Score', f'{prob_dup * 100:.1f}%')
                    st.write('**Common Key Terms:**', ', '.join(kw_info['common']) if kw_info['common'] else 'None')
                with col2:
                    st.metric('Keyword Alignment', f"{kw_info['jaccard'] * 100:.1f}%")
                    st.write('**Q1 Unique Terms:**', ', '.join(kw_info['diff1']) if kw_info['diff1'] else 'None')
                    st.write('**Q2 Unique Terms:**', ', '.join(kw_info['diff2']) if kw_info['diff2'] else 'None')




