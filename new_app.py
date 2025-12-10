# app.py - 合併版本（版面與功能用新版, 模型與情緒計算用你的 SentenceTransformer）

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import plotly.express as px
import plotly.graph_objects as go
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import os

# ================================
# 頁面設置
# ================================
st.set_page_config(
    page_title="🎵 智能音樂推薦系統",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定義 CSS 樣式（沿用別人的模板）
st.markdown("""
<style>
    /* 主容器 */
    .main {
        background-color: #FFFFFF;
        color: #000000;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    /* 標題 */
    .app-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(45deg, #FF6B6B, #FFE66D, #1DD1A1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 30px;
        padding: 20px;
    }

    /* 卡片樣式 */
    .song-card {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(0, 0, 0, 0.05);
        transition: transform 0.3s ease;
    }

    .song-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
    }

    /* 情緒標籤 */
    .emotion-tag {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        margin: 2px 5px;
        font-weight: 500;
    }

    /* 按鈕樣式 */
    .stButton>button {
        background: linear-gradient(45deg, #FF6B6B, #FFE66D);
        color: #333;
        border: none;
        border-radius: 25px;
        padding: 12px 30px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
    }

    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 20px rgba(255, 107, 107, 0.4);
    }

    /* 情緒顏色 class（如果 key 不對就當預設顏色用） */
    .joy { background-color: #FFE66D; color: #333; }
    .sad { background-color: #54A0FF; color: white; }
    .angry { background-color: #FF6B6B; color: white; }
    .fear_anxiety { background-color: #5F27CD; color: white; }
    .calm { background-color: #1DD1A1; color: white; }
    .disgust { background-color: #8395A7; color: white; }

</style>
""", unsafe_allow_html=True)

# ================================
# 情緒圖標和名稱映射（如果你的 emotion_centers key 不一樣, 會 fallback）
# ================================
EMOTION_ICONS = {
    'joy': '😊',
    'sad': '😢',
    'angry': '😠',
    'fear_anxiety': '😨',
    'calm': '😌',
    'disgust': '🤢'
}

EMOTION_NAMES_ZH = {
    'joy': '快樂幸福',
    'sad': '悲傷憂愁',
    'angry': '憤怒生氣',
    'calm': '平靜放鬆',
    'fear_anxiety': '害怕焦慮',
    'disgust': '厭惡反感'
}

# ================================
# 檔案檢查
# ================================
def check_files():
    required_files = [
        "song_bert_vectors.npy",
        "songs_meta.csv",
        "emotion_centers.pkl"
    ]
    missing = [f for f in required_files if not os.path.exists(f)]
    return missing

# ================================
# 0. Cache：一次載入全部資源（用你的 SentenceTransformer 流程）
# ================================
@st.cache_resource(show_spinner=False)
def load_resources():
    """
    載入 SentenceTransformer 模型、歌曲向量、歌曲 meta、情緒中心,
    並建立 emotion_list 與 song_emotion_matrix。
    """
    # 檢查檔案
    missing = check_files()
    if missing:
        return None

    # 1. 載入模型
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    # 2. 載入歌曲向量
    X = np.load("song_bert_vectors.npy")

    # 3. 載入 meta 與情緒中心
    df = pd.read_csv("songs_meta.csv")
    with open("emotion_centers.pkl", "rb") as f:
        emotion_centers = pickle.load(f)

    # 4. 建立 emotion_list 與歌曲情緒矩陣
    emotion_list = list(emotion_centers.keys())
    # 如果有缺欄位會丟錯, 可以視需要加 try
    song_emotion_matrix = df[[f"emo_{e}" for e in emotion_list]].values

    return model, X, df, emotion_centers, emotion_list, song_emotion_matrix

# ================================
# 文本前處理（完全用你原本那套：clean + 停用詞）
# ================================
STOP_WORDS = set([
    "的","是","了","我","你","他","她","它","們","在","就","也","很","都",
    "而","與","及","著","啦","吧","啊","呀","嘛",
    "但","卻","又","再","還","讓","給","對","把","被"
])

EN_STOP = set(ENGLISH_STOP_WORDS).union({
    "oh", "yeah", "baby", "la", "na", "woo", "hey",
    "ha", "ah", "mm", "ooh", "whoa"
})

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|www\S+|\S+@\S+", " ", text)
    text = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def remove_stopwords(text: str) -> str:
    words = text.split()
    cleaned = []
    for w in words:
        if w.lower() in EN_STOP:
            continue
        if w in STOP_WORDS:
            continue
        cleaned.append(w)
    return " ".join(cleaned)

def preprocess(text: str) -> str:
    text = clean_text(text)
    text = remove_stopwords(text)
    return text

# ================================
# 情緒分析（用你的 analyze_query 邏輯）
# ================================
def analyze_emotions(text, model, emotion_centers):
    """
    回傳：
    - emotion_scores：情緒分數 dict（不壓 0–1, 完全照 cosine）
    - q_vec：query 的 BERT 向量（1D numpy array）
    """
    clean = preprocess(text)
    if not clean:
        return {}, None

    q_vec = model.encode([clean], normalize_embeddings=True)[0]

    emotion_scores = {}
    for emo, center in emotion_centers.items():
        sim = cosine_similarity(
            q_vec.reshape(1, -1),
            center.reshape(1, -1)
        )[0][0]
        emotion_scores[emo] = sim

    return emotion_scores, q_vec

# ================================
# 推薦系統（語意 + 情緒融合, 用你的邏輯）
# ================================
def get_recommendations(query_vector,
                        query_emotions,
                        df,
                        X,
                        emotion_centers,
                        emotion_list,
                        song_emotion_matrix,
                        top_k=10,
                        semantic_weight=0.7,
                        emotion_weight=0.3):
    """
    query_vector: query 的 SentenceTransformer 向量
    query_emotions: analyze_emotions 得到的 emotion_scores dict
    emotion_list: 情緒維度順序, 與 song_emotion_matrix 對齊
    song_emotion_matrix: df 對應的情緒矩陣
    """
    if query_vector is None:
        return pd.DataFrame()

    # 語意相似度
    semantic_scores = cosine_similarity(
        query_vector.reshape(1, -1),
        X
    )[0]

    # 情緒相似度（完全照你原本的做法）
    q_emo_vec = np.array([query_emotions.get(e, 0) for e in emotion_list]).reshape(1, -1)
    emotion_scores = cosine_similarity(q_emo_vec, song_emotion_matrix)[0]

    # 融合分數
    final_scores = semantic_weight * semantic_scores + emotion_weight * emotion_scores

    result_df = df.copy()
    result_df["semantic_score"] = semantic_scores
    result_df["emotion_score"] = emotion_scores
    result_df["final_score"] = final_scores
    result_df = result_df.sort_values("final_score", ascending=False).head(top_k)
    result_df["rank"] = range(1, len(result_df) + 1)

    return result_df

# ================================
# 繪圖函數
# ================================
def create_emotion_radar_chart(emotion_scores):
    if not emotion_scores:
        return go.Figure()
    categories = [EMOTION_NAMES_ZH.get(emo, emo) for emo in emotion_scores.keys()]
    values = list(emotion_scores.values())

    fig = go.Figure(data=go.Scatterpolar(
        r=values,
        theta=categories,
        fill="toself",
        line=dict(color="#FF6B6B"),
        fillcolor="rgba(255, 107, 107, 0.3)"
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[min(0, min(values)), max(0, max(values))]  # 允許負值
            )
        ),
        showlegend=False,
        title="情緒分析雷達圖",
        title_font_size=16,
        height=400
    )
    return fig

def create_emotion_bar_chart(emotion_scores):
    if not emotion_scores:
        return go.Figure()

    emotion_data = {
        "情緒": [EMOTION_NAMES_ZH.get(emo, emo) for emo in emotion_scores.keys()],
        "分數": list(emotion_scores.values()),
        "圖標": [EMOTION_ICONS.get(emo, "🎵") for emo in emotion_scores.keys()]
    }
    df_emotions = pd.DataFrame(emotion_data).sort_values("分數", ascending=True)

    fig = px.bar(
        df_emotions,
        x="分數", y="情緒",
        orientation="h",
        text="圖標",
        color="分數",
        color_continuous_scale="RdYlBu_r"
    )
    fig.update_layout(
        title="情緒分數分佈",
        yaxis_title="",
        xaxis_title="情緒強度（cosine）",
        showlegend=False,
        height=400
    )
    fig.update_traces(
        textposition="outside",
        marker_line_width=0
    )
    return fig

# ================================
# 主應用
# ================================
def main():
    st.markdown('<h1 class="app-title">🎧 智能音樂情緒推薦系統</h1>', unsafe_allow_html=True)

    # 檔案檢查
    missing = check_files()
    if missing:
        st.error(f"❌ 缺少必要的檔案: {', '.join(missing)}")
        st.info("請先準備好向量與情緒中心檔案。")
        return

    # 載入資源
    with st.spinner("🎵 正在載入音樂推薦系統..."):
        resources = load_resources()

    if resources is None:
        st.error("無法載入系統資源。")
        return

    model, X, df, emotion_centers, emotion_list, song_emotion_matrix = resources

    # 側邊欄
    with st.sidebar:
        st.markdown("## ⚙️ 系統設定")

        st.markdown("### 推薦參數")
        top_k = st.slider("推薦數量", 5, 20, 10, 1)

        col1, col2 = st.columns(2)
        with col1:
            semantic_weight = st.slider("語意權重", 0.0, 1.0, 0.7, 0.1)
        with col2:
            emotion_weight = 1.0 - semantic_weight
            st.metric("情緒權重", f"{emotion_weight:.1f}")

        st.markdown("---")

        st.markdown("### 📊 系統資訊")
        st.metric("歌曲總數", len(df))
        st.metric("情緒維度", len(emotion_centers))
        st.metric("聚類數量", df["cluster"].nunique() if "cluster" in df.columns else 0)

        st.markdown("---")
        st.markdown("### 🎭 情緒維度")
        for emo in emotion_centers.keys():
            icon = EMOTION_ICONS.get(emo, "🎵")
            name = EMOTION_NAMES_ZH.get(emo, emo)
            st.markdown(f"{icon} **{name}**")

    # 分頁
    tab1, tab2, tab3 = st.tabs(["🎯 音樂推薦", "📊 情緒分析", "🎵 歌曲瀏覽"])
          
    # ========== Tab1: 音樂推薦 ==========
    with tab1:
        st.markdown("## 🎯 智能音樂推薦")

        query = st.text_area(
            "請描述你的心情、情境或想說的話：",
            placeholder="例如：上班好累, 想聽療癒一點的歌...",
            height=100
        )

        example_queries = [
            "快樂的愛情故事",
            "上班好累好疲憊",
            "失戀後很難過",
            "睡前想聽放鬆的歌",
            "想振奮精神",
            "想一邊讀書一邊聽的歌"
        ]
        st.markdown("**示例查詢:**")
        cols = st.columns(3)

        for i, example in enumerate(example_queries):
            col = cols[i % 3]
            if col.button(example, key=f"example_{i}"):
                st.session_state.query = example
                st.rerun()

        # if "query" in st.session_state and not query.strip():
        #     query = st.session_state.query

        if st.button("🎧 開始推薦", type="primary", use_container_width=True):
            if query.strip():
                with st.spinner("🔍 正在分析你的心情並尋找最佳歌曲..."):
                    emotion_scores, query_vector = analyze_emotions(query, model, emotion_centers)

                    if not emotion_scores:
                        st.warning("無法分析輸入文本, 請試試其他描述。")
                        return

                    recommendations = get_recommendations(
                        query_vector,
                        emotion_scores,
                        df, X,
                        emotion_centers,
                        emotion_list,
                        song_emotion_matrix,
                        top_k=top_k,
                        semantic_weight=semantic_weight,
                        emotion_weight=emotion_weight
                    )

                    if recommendations.empty:
                        st.warning("無法產生推薦結果, 請檢查資料。")
                        return

                    # 情緒分析區塊
                    st.markdown("### 🎭 你的心情分析")

                    top_emotions = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)[:3]
                    cols_top = st.columns(3)
                    for idx, (emo, score) in enumerate(top_emotions):
                        with cols_top[idx]:
                            icon = EMOTION_ICONS.get(emo, "🎵")
                            name = EMOTION_NAMES_ZH.get(emo, emo)
                            st.metric(f"{icon} {name}", f"{score:.3f}")

                    col_chart1, col_chart2 = st.columns(2)
                    with col_chart1:
                        fig_radar = create_emotion_radar_chart(emotion_scores)
                        st.plotly_chart(fig_radar, use_container_width=True)
                    with col_chart2:
                        fig_bar = create_emotion_bar_chart(emotion_scores)
                        st.plotly_chart(fig_bar, use_container_width=True)

                    # 推薦歌曲列表
                    st.markdown(f"### 🎵 為你推薦的 {len(recommendations)} 首歌")

                    for _, row in recommendations.iterrows():
                        # 取該首歌的情緒分數（若有 emo_xxx 欄位）
                        song_emotions = {}
                        for emo in emotion_list:
                            col_name = f"emo_{emo}"
                            if col_name in row:
                                song_emotions[emo] = row[col_name]
                        top_song_emotions = sorted(song_emotions.items(), key=lambda x: x[1], reverse=True)[:2]

                        emotion_tags_html = ""
                        for emo, score in top_song_emotions:
                            icon = EMOTION_ICONS.get(emo, "🎵")
                            name_short = EMOTION_NAMES_ZH.get(emo, emo)[:2]
                            css_class = emo
                            emotion_tags_html += (
                                f'<span class="emotion-tag {css_class}" '
                                f'title="{emo}: {score:.3f}">{icon} {name_short}</span>'
                            )

                        cluster_name = row.get("cluster_name", "未分類")

                        st.markdown(f"""
                        <div class="song-card">
                            <div style="display: flex; justify-content: space-between; align-items: start;">
                                <div style="flex: 1;">
                                    <h3 style="margin: 0; color: #333;">{row['歌曲']}</h3>
                                    <p style="margin: 5px 0; color: #666;">🎤 {row['歌手']}</p>
                                    <p style="margin: 5px 0; color: #888;">🏷️ {cluster_name}</p>
                                    <div style="margin: 10px 0;">
                                        {emotion_tags_html}
                                    </div>
                                </div>
                                <div style="text-align: right; min-width: 120px;">
                                    <div style="background: linear-gradient(45deg, #1DD1A1, #FFE66D);
                                                padding: 8px 15px;
                                                border-radius: 20px;
                                                color: white;
                                                font-weight: bold;">
                                        {row['final_score']:.3f}
                                    </div>
                                    <p style="margin: 5px 0; font-size: 0.8rem; color: #888;">
                                        語意: {row['semantic_score']:.3f}<br>
                                        情緒: {row['emotion_score']:.3f}
                                    </p>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    # 下載推薦結果
                    csv = recommendations.to_csv(index=False, encoding="utf-8-sig")
                    st.download_button(
                        label="📥 下載推薦結果",
                        data=csv,
                        file_name="音樂推薦結果.csv",
                        mime="text/csv"
                    )
            else:
                st.warning("請先輸入一些文字來描述你的心情。")

    # ========== Tab2: 全體情緒分析 ==========
    with tab2:
        st.markdown("## 📊 情緒分析儀表板")

        if df.empty:
            st.warning("沒有可用的歌曲資料。")
        else:
            st.markdown("### 歌曲情緒分佈")
            emotion_cols = [f"emo_{e}" for e in emotion_list if f"emo_{e}" in df.columns]
            if not emotion_cols:
                st.warning("找不到情緒欄位 emo_xxx。")
            else:
                avg_emotions = df[emotion_cols].mean().sort_values(ascending=False)
                emotion_dist_data = {
                    "情緒": [
                        EMOTION_NAMES_ZH.get(col.replace("emo_", ""), col.replace("emo_", ""))
                        for col in avg_emotions.index
                    ],
                    "平均分數": avg_emotions.values,
                    "圖標": [
                        EMOTION_ICONS.get(col.replace("emo_", ""), "🎵")
                        for col in avg_emotions.index
                    ]
                }
                df_emotion_dist = pd.DataFrame(emotion_dist_data)
                fig_dist = px.bar(
                    df_emotion_dist,
                    x="情緒", y="平均分數",
                    color="平均分數",
                    color_continuous_scale="Viridis",
                    text="圖標"
                )
                fig_dist.update_layout(
                    title="歌曲情緒平均分佈",
                    xaxis_title="情緒類型",
                    yaxis_title="平均分數",
                    height=500
                )
                st.plotly_chart(fig_dist, use_container_width=True)

            # 聚類情緒熱力圖
            if "cluster" in df.columns and "cluster_name" in df.columns and emotion_cols:
                st.markdown("### 聚類情緒分析")
                cluster_emotions = []
                for cid in sorted(df["cluster"].unique()):
                    cdata = df[df["cluster"] == cid]
                    cname = cdata["cluster_name"].iloc[0] if len(cdata) > 0 else f"聚類{cid}"
                    for emo in emotion_list:
                        col_name = f"emo_{emo}"
                        if col_name in cdata.columns:
                            cluster_emotions.append({
                                "聚類名稱": cname,
                                "情緒": EMOTION_NAMES_ZH.get(emo, emo),
                                "平均分數": cdata[col_name].mean()
                            })
                if cluster_emotions:
                    df_cluster_emotions = pd.DataFrame(cluster_emotions)
                    heatmap_data = df_cluster_emotions.pivot_table(
                        index="聚類名稱",
                        columns="情緒",
                        values="平均分數"
                    )
                    fig_heatmap = px.imshow(
                        heatmap_data,
                        color_continuous_scale="RdYlBu_r",
                        aspect="auto"
                    )
                    fig_heatmap.update_layout(
                        title="聚類情緒熱力圖",
                        height=400
                    )
                    st.plotly_chart(fig_heatmap, use_container_width=True)

    # ========== Tab3: 歌曲瀏覽 ==========
    with tab3:
        st.markdown("## 🎵 歌曲瀏覽與搜尋")

        if df.empty:
            st.warning("沒有可用的歌曲資料。")
        else:
            col_search, col_filter1, col_filter2 = st.columns(3)
            with col_search:
                search_term = st.text_input("搜尋歌曲或歌手", "")

            with col_filter1:
                if "cluster_name" in df.columns:
                    cluster_options = ["全部"] + sorted(df["cluster_name"].dropna().unique().tolist())
                    selected_cluster = st.selectbox("選擇歌曲類別", cluster_options)
                else:
                    selected_cluster = "全部"
                    st.info("沒有分類欄位 cluster_name。")

            with col_filter2:
                emotion_display_names = ["全部"] + [
                    EMOTION_NAMES_ZH.get(e, e) for e in emotion_list
                ]
                selected_emotion = st.selectbox("選擇主要情緒", emotion_display_names)

            # 篩選
            fdf = df.copy()
            if search_term:
                fdf = fdf[
                    fdf["歌曲"].astype(str).str.contains(search_term, case=False, na=False) |
                    fdf["歌手"].astype(str).str.contains(search_term, case=False, na=False)
                ]
            if selected_cluster != "全部" and "cluster_name" in fdf.columns:
                fdf = fdf[fdf["cluster_name"] == selected_cluster]
            if selected_emotion != "全部":
                emo_en = None
                for e in emotion_list:
                    if EMOTION_NAMES_ZH.get(e, e) == selected_emotion:
                        emo_en = e
                        break
                if emo_en and f"emo_{emo_en}" in fdf.columns:
                    fdf = fdf.sort_values(f"emo_{emo_en}", ascending=False)

            st.markdown(f"### 找到 {len(fdf)} 首歌曲")
            if len(fdf) == 0:
                st.info("沒有符合條件的歌曲。")
            else:
                page_size = 20
                total_pages = max(1, (len(fdf) - 1) // page_size + 1)
                if total_pages > 1:
                    page = st.number_input("頁碼", min_value=1, max_value=total_pages, value=1)
                else:
                    page = 1
                start_idx = (page - 1) * page_size
                end_idx = min(page * page_size, len(fdf))

                for _, row in fdf.iloc[start_idx:end_idx].iterrows():
                    song_emotions = {}
                    for emo in emotion_list:
                        col_name = f"emo_{emo}"
                        if col_name in row:
                            song_emotions[emo] = row[col_name]
                    top_emotions = sorted(song_emotions.items(), key=lambda x: x[1], reverse=True)[:3]

                    emotion_tags_html = ""
                    for emo, score in top_emotions:
                        if score > 0.2:
                            icon = EMOTION_ICONS.get(emo, "🎵")
                            name_short = EMOTION_NAMES_ZH.get(emo, emo)[:2]
                            css_class = emo
                            emotion_tags_html += (
                                f'<span class="emotion-tag {css_class}" '
                                f'title="{emo}: {score:.3f}">{icon} {name_short}</span>'
                            )

                    cluster_name = row.get("cluster_name", "未分類")

                    st.markdown(f"""
                    <div class="song-card">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <h4 style="margin: 0; color: #333;">{row['歌曲']}</h4>
                                <p style="margin: 5px 0; color: #666;">🎤 {row['歌手']}</p>
                                <p style="margin: 5px 0; color: #888;">
                                    <span style="background: #e0e0e0; padding: 2px 8px; border-radius: 10px;">
                                        {cluster_name}
                                    </span>
                                </p>
                            </div>
                            <div style="text-align: right;">
                                {emotion_tags_html}
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                if total_pages > 1:
                    st.caption(f"顯示第 {start_idx + 1}–{end_idx} 首, 共 {len(fdf)} 首, 共 {total_pages} 頁。")

if __name__ == "__main__":
    main()