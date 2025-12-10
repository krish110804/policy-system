# -------------------------------------------------------------
# Policy Intelligence System - Final Full Version (policies2.json)
# + Automatic multi-category assignment (keyword + tags)
# -------------------------------------------------------------
import streamlit as st
import numpy as np
import pandas as pd
import json
import os
import io
from pathlib import Path
from datetime import datetime
from collections import Counter

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Summarizer (T5)
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Keyword Extraction
from keybert import KeyBERT

# Clustering + Visualization
import umap
from sklearn.cluster import KMeans
import plotly.express as px

# Voice Processing
import speech_recognition as sr
from gtts import gTTS

# ----------------------------
# Config
# ----------------------------
POLICY_JSON = "policies2.json"      # main dataset
EMBED_FILE = "embeddings.npy"
MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K_RECOMMEND = 6
ADMIN_PASSWORD = "admin123"  # CHANGE before final submission

st.set_page_config(page_title="Policy Intelligence System", layout="wide", page_icon="📘")

# ----------------------------
# Session-state defaults
# ----------------------------
if "page" not in st.session_state:
    st.session_state.page = "User"
if "admin_logged" not in st.session_state:
    st.session_state.admin_logged = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ----------------------------
# Category keywords map (used to auto-assign categories)
# You can edit/extend these keywords as needed.
# Each category maps to multiple keywords — if a policy contains any keyword or tag,
# that category will be assigned. Because you chose option B, we allow multiple categories.
# ----------------------------
CATEGORY_KEYWORDS = {
    "Education": ["scholarship", "school", "student", "college", "education", "fee", "tuition", "skill", "training", "course"],
    "Healthcare": ["health", "hospital", "medical", "treatment", "insurance", "clinic", "doctor", "medicine", "immuniz", "vaccin"],
    "Agriculture": ["farmer", "agriculture", "crop", "irrig", "fertil", "seeds", "farm", "msc", "agri"],
    "Women & Child": ["women", "mother", "child", "girl", "maternity", "female", "empower"],
    "Business & Entrepreneurship": ["startup", "msme", "loan", "credit", "subsidy", "business", "entrepreneur"],
    "Employment & Training": ["job", "employment", "placement", "hire", "skill development", "apprentice", "training"],
    "Housing": ["house", "housing", "pmay", "rental", "shelter"],
    "Social Security & Pension": ["pension", "widow", "senior", "old age", "social security", "pensioner"],
    "Finance & Banking": ["bank", "loan", "interest", "finance", "banking", "microfinance", "subsidy", "credit"],
    "Welfare & Public Services": ["welfare", "relief", "public", "service", "assist", "benefit"]
}

# Normalize keys by lowercasing keywords (done in matching)
for k, kws in CATEGORY_KEYWORDS.items():
    CATEGORY_KEYWORDS[k] = [w.lower() for w in kws]

# ----------------------------
# CSS
# ----------------------------
st.markdown("""
<style>
.card {
    background-color: #1e1e1e;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #333;
    margin-bottom: 12px;
    color: #fff;
}
.score-bar {
    height: 8px;
    border-radius: 4px;
    background: linear-gradient(90deg, #4CAF50, #8BC34A);
}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Utility: Load & validate policies file
# ----------------------------
@st.cache_data
def load_policies(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)

    # Ensure expected columns exist
    for col in ["title", "tags", "state", "categoryId", "summary", "eligibility", "benefits"]:
        if col not in df.columns:
            df[col] = ""

    # Normalize tags to list of lower strings
    df["tags"] = df["tags"].apply(lambda x: x if isinstance(x, list) else [])
    df["tags"] = df["tags"].apply(lambda tags: [str(t).lower() for t in tags])

    return df

# ----------------------------
# Auto-assign categories (multi-label) using simple keyword + tags rules
# NOTE: fallback to "General" (no Category <id> fallback) to avoid garbage buckets.
# ----------------------------
def auto_detect_categories_for_row(row):
    text = " ".join([
        str(row.get("title","")),
        str(row.get("summary","")),
        " ".join(row.get("tags", []))
    ]).lower()

    assigned = set()
    for cname, kwlist in CATEGORY_KEYWORDS.items():
        for kw in kwlist:
            if kw in text:
                assigned.add(cname)
                break

    # Fallback: do NOT create "Category <id>" buckets. Use a single 'General' label if nothing matched.
    if not assigned:
        assigned.add("General")

    return list(assigned)

# ----------------------------
# SBERT Model - cached
# ----------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer(MODEL_NAME)

# ----------------------------
# T5 Summarizer - cached
# ----------------------------
@st.cache_resource
def load_summarizer():
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    return tokenizer, model

summ_tokenizer, summ_model = load_summarizer()

def summarize_text(text, max_len=120):
    input_ids = summ_tokenizer.encode("summarize: " + text,
                                      return_tensors="pt",
                                      max_length=512,
                                      truncation=True)
    summary_ids = summ_model.generate(
        input_ids, max_length=max_len, min_length=40, num_beams=3,
        temperature=0.7, length_penalty=2.0, early_stopping=True
    )
    return summ_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# ----------------------------
# KeyBERT - cached
# ----------------------------
@st.cache_resource
def load_keybert():
    return KeyBERT(model=load_model())

kw_model = load_keybert()

def extract_keywords(text, top_n=8):
    keys = kw_model.extract_keywords(text, top_n=top_n)
    return [k for k, s in keys]

# ----------------------------
# Embedding helpers
# ----------------------------
def combine_fields(row):
    return " ".join([str(row.get("title","")), str(row.get("summary","")),
                     str(row.get("eligibility","")), str(row.get("benefits","")),
                     " ".join(row.get("tags", []))])

@st.cache_data
def get_or_create_embeddings(df_records, _model):
    if os.path.exists(EMBED_FILE):
        try:
            emb = np.load(EMBED_FILE)
            if emb.shape[0] == len(df_records):
                return emb
        except Exception:
            pass
    texts = df_records.apply(combine_fields, axis=1).tolist()
    emb = _model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    np.save(EMBED_FILE, emb)
    return emb

# ----------------------------
# Load data and models (initial)
# ----------------------------
df = load_policies(POLICY_JSON)

# Add/overwrite the auto categoryName multi-label column
df["categoryName"] = df.apply(auto_detect_categories_for_row, axis=1)

# Flatten category list for sidebar choices and charts
def get_all_categories_from_df(df_local):
    flat = []
    for cats in df_local["categoryName"].tolist():
        flat.extend(cats if isinstance(cats, list) else [cats])
    unique = sorted(list(set(flat)))
    return unique

all_categories = get_all_categories_from_df(df)

model = load_model()
embeddings = get_or_create_embeddings(df, _model=model)

# ----------------------------
# Sidebar navigation + filters (category filter is now multi-select)
# ----------------------------
st.sidebar.title("Navigation")
nav_choice = st.sidebar.radio("Go to:", ["User", "Admin Dashboard"])
st.session_state.page = nav_choice

st.sidebar.title("Filters")
state_sel = st.sidebar.selectbox("State", ["All"] + sorted(df["state"].dropna().unique().tolist()))
category_sel = st.sidebar.multiselect("Category (multi-select)", ["All"] + all_categories, default=["All"])
tag_list = sorted({tag for tags in df["tags"] for tag in tags})
tag_sel = st.sidebar.multiselect("Tags", tag_list)

# ----------------------------
# Helper: refresh embeddings (remove file and recompute)
# ----------------------------
def refresh_embeddings(df_ref):
    if os.path.exists(EMBED_FILE):
        try:
            os.remove(EMBED_FILE)
        except Exception:
            pass
    return get_or_create_embeddings(df_ref, _model=model)

# ----------------------------
# Filter dataset view (handle multi-category selection 'All' OR specific)
# ----------------------------
filtered_idx = df.index.tolist()
if state_sel != "All":
    filtered_idx = df[df["state"] == state_sel].index.intersection(filtered_idx)

# category_sel can contain "All" or list
if not (len(category_sel) == 1 and category_sel[0] == "All"):
    # keep policies where ANY of the selected categories appear in policy's categoryName list
    mask_cat = df["categoryName"].apply(lambda cats: any(c in cats for c in category_sel))
    filtered_idx = df[mask_cat].index.intersection(filtered_idx)

if tag_sel:
    mask = df["tags"].apply(lambda ts: all(t in ts for t in tag_sel))
    filtered_idx = df[mask].index.intersection(filtered_idx)

filtered_df = df.loc[filtered_idx]
filtered_embeddings = embeddings[filtered_idx]

# ----------------------------
# Header
# ----------------------------
st.title("📘 Policy Intelligence System")
st.write("Explore, analyse, compare, and interact with policies using AI.")

# ----------------------------
# ADMIN DASHBOARD
# ----------------------------
if st.session_state.page == "Admin Dashboard":
    st.title("👨‍💼 Admin Dashboard")
    if not st.session_state.admin_logged:
        st.subheader("🔐 Admin Login")
        pwd = st.text_input("Enter admin password:", type="password")
        if st.button("Login"):
            if pwd == ADMIN_PASSWORD:
                st.session_state.admin_logged = True
                st.success("Logged in as admin.")
                st.experimental_rerun()
            else:
                st.error("Wrong password.")
        st.stop()

    # Logged in
    st.success("Admin Mode — You are logged in.")
    if st.button("Logout"):
        st.session_state.admin_logged = False
        st.session_state.page = "User"
        st.experimental_rerun()

    tabs = st.tabs(["📊 Overview", "📂 Manage Policies", "➕ Add Policy", "🗑 Delete Policy", "🔁 Rebuild Embeddings"])

    # Overview - use categoryName exploded counts
    with tabs[0]:
        st.header("System Overview")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Policies", len(df))
        col2.metric("States Covered", int(df['state'].nunique()))
        col3.metric("Distinct Categories", len(get_all_categories_from_df(df)))
        st.write("---")
        st.write("State distribution")
        st.bar_chart(df["state"].value_counts())

        # build exploded category counts
        exploded = pd.DataFrame({"category": [c for lst in df["categoryName"] for c in lst]})
        st.write("Category distribution")
        st.bar_chart(exploded["category"].value_counts())

    # Manage Policies
    with tabs[1]:
        st.header("All Policies (preview)")
        st.dataframe(df, use_container_width=True, height=500)

    # Add policy
    with tabs[2]:
        st.header("Add a New Policy (JSON single object)")
        st.write("Upload a JSON file containing a single policy object with fields: title, state, categoryId, summary, eligibility, benefits, tags")
        up = st.file_uploader("Upload single-policy JSON", type=["json"], key="add_policy")
        if up is not None:
            try:
                new_policy = json.load(up)
                # normalize tags
                if "tags" in new_policy and isinstance(new_policy["tags"], list):
                    new_policy["tags"] = [str(t).lower() for t in new_policy["tags"]]
                else:
                    new_policy["tags"] = []
                new_policy_record = new_policy.copy()
                # auto detect categories
                new_policy_record["categoryName"] = auto_detect_categories_for_row(new_policy_record)
                new_df = pd.DataFrame([new_policy_record])
                df = pd.concat([df, new_df], ignore_index=True)
                # save
                with open(POLICY_JSON, "w", encoding="utf-8") as f:
                    json.dump(df.drop(columns=["categoryName"]).to_dict("records"), f, indent=2, ensure_ascii=False)
                st.success("Policy added. Regenerating embeddings (this can take a while)...")
                embeddings = refresh_embeddings(df)
                st.success("Embeddings rebuilt.")
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Failed to add policy: {e}")

    # Delete policy
    with tabs[3]:
        st.header("Delete Policy")
        if len(df) == 0:
            st.info("No policies to delete.")
        else:
            to_del = st.selectbox("Select policy to delete", df["title"].tolist())
            if st.button("Delete selected policy"):
                df = df[df["title"] != to_del].reset_index(drop=True)
                with open(POLICY_JSON, "w", encoding="utf-8") as f:
                    json.dump(df.drop(columns=["categoryName"]).to_dict("records"), f, indent=2, ensure_ascii=False)
                st.warning(f"Deleted: {to_del}. Regenerating embeddings (this can take a while)...")
                embeddings = refresh_embeddings(df)
                st.success("Embeddings rebuilt.")
                st.experimental_rerun()

    # Rebuild embeddings
    with tabs[4]:
        st.header("Rebuild Embeddings")
        st.write("Use this if you updated the JSON manually or uploaded many policies.")
        if st.button("Rebuild now"):
            embeddings = refresh_embeddings(df)
            st.success("Embeddings rebuilt.")
            st.experimental_rerun()

    st.stop()  # stop further UI for admin page

# ----------------------------
# SEARCH
# ----------------------------
st.markdown("---")
st.subheader("🔎 Search Policies")
query = st.text_input("Enter search query:", placeholder="e.g., loan for women entrepreneurs")
top_k = st.slider("Number of results", 1, 10, 5)
if st.button("Search"):
    if query.strip():
        q_vec = model.encode(query, convert_to_numpy=True)
        scores = cosine_similarity([q_vec], filtered_embeddings)[0]
        local_idx = scores.argsort()[::-1][:top_k]
        global_idx = np.array(filtered_idx)[local_idx]
        st.subheader("Search Results")
        for i, gi in enumerate(global_idx):
            row = df.iloc[gi]
            score = float(scores[local_idx[i]])
            cats = ", ".join(row.get("categoryName", []))
            st.markdown(f"""
                <div class="card">
                    <h3>{row['title']}</h3>
                    <p><b>Categories:</b> {cats}</p>
                    <p><b>Summary:</b> {row['summary']}</p>
                    <p><b>Eligibility:</b> {row['eligibility']}</p>
                    <p><b>Benefits:</b> {row['benefits']}</p>
                    <p><b>Score:</b> {score:.3f}</p>
                    <div class="score-bar" style="width:{score*100}%"></div>
                </div>
            """, unsafe_allow_html=True)

# ----------------------------
# RECOMMENDATIONS
# ----------------------------
st.markdown("---")
st.subheader("🎯 Recommend Similar Policies")
policy_choice = st.selectbox("Select a policy", df["title"].tolist())
if st.button("Recommend Similar"):
    idx = df[df["title"] == policy_choice].index[0]
    base_vec = embeddings[idx]
    sim = cosine_similarity([base_vec], embeddings)[0]
    top = sim.argsort()[::-1][1:TOP_K_RECOMMEND+1]
    st.write(f"Recommendations for **{policy_choice}**:")
    for i in top:
        st.write(f"- {df.iloc[i]['title']} — {sim[i]:.3f}")

# ----------------------------
# POLICY COMPARISON
# ----------------------------
st.markdown("---")
st.subheader("⚖️ Compare Two Policies")
colA, colB = st.columns(2)
policyA = colA.selectbox("Policy A", df["title"].tolist(), key="cmpA")
policyB = colB.selectbox("Policy B", df["title"].tolist(), key="cmpB")
if st.button("Compare Policies"):
    ia = df[df["title"] == policyA].index[0]
    ib = df[df["title"] == policyB].index[0]
    score = cosine_similarity([embeddings[ia]], [embeddings[ib]])[0][0]
    st.write(f"Similarity score: **{score:.3f}**")
    st.write("**Policy A Categories**")
    st.info(", ".join(df.loc[ia, "categoryName"]))
    st.write("**Policy A Summary**")
    st.info(df.loc[ia, "summary"])
    st.write("**Policy B Categories**")
    st.info(", ".join(df.loc[ib, "categoryName"]))
    st.write("**Policy B Summary**")
    st.info(df.loc[ib, "summary"])

# ----------------------------
# SUMMARIZATION
# ----------------------------
st.markdown("---")
st.subheader("📝 Summarize a Policy")
policy_sum = st.selectbox("Choose a policy to summarize", df["title"].tolist(), key="sum1")
if st.button("Generate Summary"):
    idx = df[df["title"] == policy_sum].index[0]
    full_text = df.loc[idx, "summary"] + " " + df.loc[idx, "benefits"] + " " + df.loc[idx, "eligibility"]
    summary = summarize_text(full_text)
    st.success(summary)

# ----------------------------
# KEYWORD EXTRACTION
# ----------------------------
st.markdown("---")
st.subheader("🔑 Extract Keywords")
policy_kw = st.selectbox("Choose policy for keywords", df["title"].tolist(), key="kw1")
if st.button("Extract Keywords"):
    idx = df[df["title"] == policy_kw].index[0]
    text = df.loc[idx, "summary"] + " " + df.loc[idx, "benefits"]
    keys = extract_keywords(text)
    st.write("Top keywords:")
    for k in keys:
        st.markdown(f"- **{k}**")

# ----------------------------
# CLUSTERING + UMAP
# ----------------------------
st.markdown("---")
st.subheader("📊 Policy Clustering (UMAP + KMeans)")
@st.cache_data
def cluster_embeddings(emb, n_clusters=6):
    return KMeans(n_clusters=n_clusters, random_state=42).fit_predict(emb)
@st.cache_data
def reduce_umap(emb):
    return umap.UMAP(n_neighbors=10, min_dist=0.2, metric="cosine").fit_transform(emb)
if st.button("Generate Clusters"):
    labels = cluster_embeddings(embeddings)
    umap_points = reduce_umap(embeddings)
    df_viz = pd.DataFrame({"x": umap_points[:,0], "y": umap_points[:,1], "title": df["title"], "cluster": labels})
    fig = px.scatter(df_viz, x="x", y="y", color="cluster", hover_name="title", title="Policy Clusters", width=900, height=600)
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# AI Eligibility Matcher
# ----------------------------
st.markdown("---")
st.header("🔎 AI Eligibility Matcher — Best Matches for You")
user_state = st.selectbox("Your State", options=["Any"] + sorted(df['state'].dropna().unique().tolist()))
user_age = st.number_input("Age", min_value=1, max_value=120, value=25)
user_gender = st.selectbox("Gender", ["Male", "Female", "Other"])
user_income = st.number_input("Annual Household Income (INR)", min_value=0, value=50000)
user_cat = st.selectbox("Social Category", ["Any", "GEN", "OBC", "SC", "ST", "EWS"])
user_disability = st.selectbox("Disability", ["No", "Yes"])
col1, col2 = st.columns([1,0.4])
with col1:
    top_k_matches = st.slider("How many top matches to show", 3, 20, 7)
with col2:
    min_sem = st.slider("Min semantic sim", 0.1, 0.8, 0.28)
if st.button("Find Best Matches"):
    def build_profile_text(state, age, gender, income, category, disability):
        parts = [
            f"State: {state}" if state and state != "Any" else "State: any",
            f"Age: {age}", f"Gender: {gender}",
            f"Annual household income (INR): {income}",
            f"Category: {category}" if category and category != "Any" else "Category: any",
            f"Disability: {disability}"
        ]
        return "I am a person with the following profile: " + "; ".join(parts) + ". I want to find government schemes I qualify for."
    def rule_score(policy_row, user_profile):
        s = 0.0
        text = (str(policy_row.get("eligibility","")) + " " + str(policy_row.get("summary",""))).lower()
        if user_profile["state"] != "Any":
            if str(policy_row.get("state","")).strip().lower() == user_profile["state"].strip().lower():
                s += 0.35
        else:
            s += 0.05
        if "women" in text or "female" in text or "girl" in text:
            if user_profile["gender"].lower() == "female": s += 0.2
            else: s -= 0.05
        cat = user_profile["category"].lower()
        if cat != "any":
            if cat in text: s += 0.2
            else: s -= 0.05
        if ("disability" in text or "divyang" in text or "disabled" in text):
            if user_profile["disability"].lower() == "yes": s += 0.2
            else: s -= 0.05
        if any(k in text for k in ["below poverty", "below rs", "income below", "annual income less", "income up to"]):
            if user_profile["income"] <= 250000: s += 0.15
            else: s -= 0.05
        return max(0.0, min(1.0, s))
    def ai_match_policies(df_local, model_local, emb_local, user_profile, top_k=7, min_semantic=0.25, weight_semantic=0.7):
        profile_text = build_profile_text(user_profile["state"], user_profile["age"], user_profile["gender"], user_profile["income"], user_profile["category"], user_profile["disability"])
        q_vec = model_local.encode(profile_text, convert_to_numpy=True)
        sims = cosine_similarity([q_vec], emb_local)[0]
        candidate_idx = np.where(sims >= min_semantic)[0]
        if len(candidate_idx) == 0:
            candidate_idx = np.argsort(sims)[::-1][:150]
        results = []
        for i in candidate_idx:
            row = df_local.iloc[i]
            sem = float(sims[i])
            rscore = rule_score(row, user_profile)
            final = weight_semantic * sem + (1 - weight_semantic) * rscore
            results.append({"idx": i, "title": row.get("title",""), "summary": row.get("summary",""), "eligibility": row.get("eligibility",""), "state": row.get("state",""), "tags": row.get("tags", []), "semantic": sem, "rule": rscore, "final": final})
        return sorted(results, key=lambda x: x["final"], reverse=True)[:top_k], q_vec
    user_profile = {"state": user_state, "age": user_age, "gender": user_gender, "income": user_income, "category": user_cat, "disability": user_disability}
    with st.spinner("Computing matches..."):
        matches, profile_vec = ai_match_policies(df, model, embeddings, user_profile, top_k=top_k_matches, min_semantic=min_sem)
    if not matches:
        st.warning("No matches found. Try lowering min semantic sim or increase top_k.")
    else:
        best = matches[0]
        st.markdown("## 🥇 Best Match")
        st.markdown(f"**{best['title']}** — Score: **{int(best['final']*100)}%**")
        st.info(best["summary"])
        st.write("**Eligibility:**", best["eligibility"] if best["eligibility"] else "_Not provided_")
        if len(matches) > 1:
            st.markdown("## 🔎 Other Matches")
            for m in matches[1:]:
                st.markdown(f"### {m['title']} — {int(m['final']*100)}%")
                st.info(m["summary"])
                st.write("Eligibility:", m["eligibility"])

# ----------------------------
# VOICE + TEXT CHATBOT
# ----------------------------
st.markdown("---")
st.subheader("🤖 Smart Policy Chatbot (Text + Voice)")

# ensure chat_history in session
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.write("### 🎤 Speak or upload audio (optional)")
try:
    audio_file = st.audio_input("Record your voice query:")
except Exception:
    audio_file = st.file_uploader("Upload audio file (wav) instead", type=["wav","mp3"])

voice_text = ""
if audio_file is not None:
    st.info("Processing audio...")
    recognizer = sr.Recognizer()
    try:
        if hasattr(audio_file, "read"):
            audio_bytes = audio_file.read()
            audio_buf = io.BytesIO(audio_bytes)
            with sr.AudioFile(audio_buf) as source:
                audio_data = recognizer.record(source)
        else:
            with sr.AudioFile(audio_file) as source:
                audio_data = recognizer.record(source)
        voice_text = recognizer.recognize_google(audio_data)
        st.success(f"Recognized: {voice_text}")
    except Exception as e:
        st.error(f"Could not transcribe audio: {e}")

user_query = st.text_input("Or type your question here:")
if voice_text:
    user_query = voice_text

def smart_policy_chatbot(query, top_k=3):
    q_vec = model.encode(query, convert_to_numpy=True)
    sims = cosine_similarity([q_vec], embeddings)[0]
    top_idx = sims.argsort()[::-1][:top_k]
    refs = df.iloc[top_idx]
    context = ""
    for _, row in refs.iterrows():
        context += f"Title: {row['title']}. Summary: {row['summary']}. Eligibility: {row['eligibility']}. Benefits: {row['benefits']}. "
    prompt = f"Based on the context, answer clearly and helpfully.\n\nContext: {context}\n\nQuestion: {query}\nAnswer:"
    input_ids = summ_tokenizer.encode(prompt, return_tensors="pt", truncation=True)
    output_ids = summ_model.generate(input_ids, max_length=140, min_length=40, num_beams=4, temperature=0.7, no_repeat_ngram_size=2)
    answer = summ_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    final = answer.split("Answer:")[-1].strip()
    return final, refs

if st.button("Ask Chatbot"):
    if user_query and user_query.strip():
        with st.spinner("AI is thinking..."):
            reply, used_refs = smart_policy_chatbot(user_query)
        st.session_state.chat_history.append(("You", user_query))
        st.session_state.chat_history.append(("Bot", reply))
        # voice output
        try:
            tts = gTTS(reply, lang="en")
            tmp_mp3 = "bot_voice.mp3"
            tts.save(tmp_mp3)
            with open(tmp_mp3, "rb") as f:
                st.audio(f.read(), format="audio/mp3")
        except Exception:
            st.info("Audio output unavailable on this environment.")

# show chat history (last 10)
st.markdown("### 💬 Conversation")
for sender, msg in st.session_state.chat_history[-10:]:
    if sender == "You":
        st.markdown(f"**🧑 You:** {msg}")
    else:
        st.markdown(f"**🤖 AI:** {msg}")

# referenced policies (from most recent query) - guard existence
if "used_refs" in locals() and isinstance(used_refs, pd.DataFrame):
    st.markdown("### 📄 Policies Referenced")
    for _, row in used_refs.iterrows():
        st.markdown(f"""
        <div class="card">
            <h4>{row['title']}</h4>
            <p>{row['summary'][:200]}...</p>
        </div>
        """, unsafe_allow_html=True)

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.write("Built with ❤️ by Krish ")
st.write(f"Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
