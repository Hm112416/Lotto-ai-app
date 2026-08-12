import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import time
import os
import pickle
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

st.set_page_config(page_title="Lotto AI Master Heavy Pro", layout="wide", page_icon="⚙️")

def get_connection():
    return sqlite3.connect('lotto_master.db')

def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS draws (
        draw_id INTEGER PRIMARY KEY,
        draw_date TEXT,
        ball1 INTEGER, ball2 INTEGER, ball3 INTEGER, ball4 INTEGER, ball5 INTEGER, ball6 INTEGER,
        bonus INTEGER
    )
    ''')
    conn.commit()
    conn.close()

init_db()

EXCEL_PATH = 'lotto_final_2439.xlsx'

if os.path.exists(EXCEL_PATH):
    conn = get_connection()
    cursor = conn.cursor()
    df_excel = pd.read_excel(EXCEL_PATH)
    cursor.execute("SELECT COUNT(*) FROM draws")
    db_count = cursor.fetchone()[0]
    
    if db_count < len(df_excel):
        cursor.execute("DELETE FROM draws")
        df_excel['Date'] = pd.to_datetime(df_excel['Date'], errors='coerce').dt.strftime('%Y-%m-%d')
        df_excel['Date'] = df_excel['Date'].fillna('2024-01-01')
        
        for idx, row in df_excel.iterrows():
            cursor.execute('''
            INSERT OR REPLACE INTO draws (draw_id, draw_date, ball1, ball2, ball3, ball4, ball5, ball6, bonus)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                int(row['Draw']), 
                str(row['Date']), 
                int(row['Ball 1']), 
                int(row['Ball 2']), 
                int(row['Ball 3']), 
                int(row['Ball 4']), 
                int(row['Ball 5']), 
                int(row['Ball 6']), 
                int(row['Bonus']) if 'Bonus' in row and pd.notnull(row['Bonus']) else 0
            ))
        conn.commit()
    conn.close()

def load_draws_df():
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM draws ORDER BY draw_id ASC", conn)
    conn.close()
    main_cols = ['ball1', 'ball2', 'ball3', 'ball4', 'ball5', 'ball6', 'bonus']
    for col in main_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(1).astype(int)
    return df

df_all = load_draws_df()

def build_advanced_features(df):
    num_draws = len(df)
    matrix = np.zeros((num_draws, 42), dtype=int)
    main_cols = ['ball1', 'ball2', 'ball3', 'ball4', 'ball5', 'ball6']
    
    for i, row in enumerate(df[main_cols].values):
        for ball in row:
            b_idx = int(ball) - 1
            if 0 <= b_idx < 42:
                matrix[i, b_idx] = 1
                
    X, y = [], []
    lag_short = 5
    lag_med = 15
    lag_long = 30
    
    for i in range(lag_long, num_draws):
        freq_s = matrix[i-lag_short:i].sum(axis=0)
        freq_m = matrix[i-lag_med:i].sum(axis=0)
        freq_l = matrix[i-lag_long:i].sum(axis=0)
        
        delays = np.zeros(42)
        for b in range(42):
            sub = matrix[:i, b]
            pos = np.where(sub == 1)[0]
            delays[b] = (i - pos[-1]) if len(pos) > 0 else lag_long
            
        features = np.concatenate([freq_s, freq_m, freq_l, delays, matrix[i-1], matrix[i-2]])
        X.append(features)
        y.append(matrix[i])
        
    return np.array(X), np.array(y), matrix

def clean_list(arr):
    return [int(x) for x in arr]

MODEL_FILE = "best_lotto_model.pkl"

st.sidebar.title("🔥 Lotto AI Heavy Engine")
st.sidebar.markdown(f"**عدد السحوبات:** `{len(df_all)}` سحبة")

if os.path.exists(MODEL_FILE):
    try:
        with open(MODEL_FILE, 'rb') as f:
            saved_data = pickle.load(f)
            st.sidebar.success(f"🏆 **النموذج الأفضل المحفوظ:**\n\nأعلى دقة: `{saved_data.get('score', 0):.2f}%`")
    except:
        st.sidebar.info("لم يتم حفظ نموذج أفضل بعد.")
else:
    st.sidebar.info("لا يوجد نموذج ذكي محفوظ حالياً.")

st.sidebar.markdown("---")

menu = st.sidebar.radio("القائمة الرئيسية", [
    "📊 تحليل الكثافة والفجوات",
    "🕰️ آلة الزمن واختبار النموذج المحفوظ",
    "⚙️ مشغل التدريب وحفظ الأفضل",
    "🎫 توليد القائمة الذهبية والبطاقات"
])

# 1. Analytics
if menu == "📊 تحليل الكثافة والفجوات":
    st.header("📊 تحليل الفجوات ودورة الغياب (Gaps Tracker)")
    main_cols = ['ball1', 'ball2', 'ball3', 'ball4', 'ball5', 'ball6']
    all_balls = df_all[main_cols].values
    
    delays = {}
    for ball in range(1, 43):
        last_seen = 0
        for idx, row in enumerate(all_balls):
            if ball in row:
                last_seen = idx
        delays[ball] = len(all_balls) - 1 - last_seen
        
    df_delays = pd.DataFrame(list(delays.items()), columns=['الرقم', 'عدد السحوبات منذ آخر ظهور']).sort_values('عدد السحوبات منذ آخر ظهور', ascending=False)
    
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("⚠️ الأرقام الأكثر غياباً (المتأخرة جداً)")
        st.dataframe(df_delays.head(10))
    with c2:
        st.subheader("🔥 الأرقام الأكثر نشاطاً حالياً")
        st.dataframe(df_delays.tail(10))

# 2. Time Machine
elif menu == "🕰️ آلة الزمن واختبار النموذج المحفوظ":
    st.header("🕰️ آلة الزمن للتحقق من دقة النموذج المحفوظ")
    max_d = int(df_all['draw_id'].max())
    target_draw = st.number_input("اختبر سحبة من الماضي", min_value=100, max_value=max_d, value=2000 if max_d>=2000 else max_d)
    
    if st.button("🚀 تشغيل المحاكاة"):
        with st.spinner("جاري التحقق..."):
            past_df = df_all[df_all['draw_id'] < target_draw]
            actual_row = df_all[df_all['draw_id'] == target_draw]
            actual_balls = clean_list(sorted(actual_row[['ball1', 'ball2', 'ball3', 'ball4', 'ball5', 'ball6']].values[0]))
            
            X, y, matrix = build_advanced_features(past_df)
            
            if os.path.exists(MODEL_FILE):
                with open(MODEL_FILE, 'rb') as f:
                    saved_data = pickle.load(f)
                    clf = saved_data['model']
            else:
                clf = RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42)
                clf.fit(X, y)
                
            last_i = len(past_df)
            freq_s = matrix[last_i-5:last_i].sum(axis=0)
            freq_m = matrix[last_i-15:last_i].sum(axis=0)
            freq_l = matrix[last_i-30:last_i].sum(axis=0)
            delays = np.zeros(42)
            for b in range(42):
                sub = matrix[:last_i, b]
                pos = np.where(sub == 1)[0]
                delays[b] = (last_i - pos[-1]) if len(pos) > 0 else 30
            
            X_test = np.concatenate([freq_s, freq_m, freq_l, delays, matrix[-1], matrix[-2]]).reshape(1, -1)
            probas = clf.predict_proba(X_test)
            probs = np.zeros(42)
            for idx in range(42):
                if probas[idx].shape[1] > 1:
                    probs[idx] = probas[idx][0, 1]
                    
            top_6 = clean_list(sorted(np.argsort(probs)[-6:] + 1))
            top_12 = clean_list(sorted(np.argsort(probs)[-12:] + 1))
            top_18 = clean_list(sorted(np.argsort(probs)[-18:] + 1))
            
            h6 = len(set(actual_balls).intersection(set(top_6)))
            h12 = len(set(actual_balls).intersection(set(top_12)))
            h18 = len(set(actual_balls).intersection(set(top_18)))
            
            st.success(f"🎯 **الأرقام الحقيقية للسحبة {target_draw}:** `{actual_balls}`")
            st.write(f"🔹 أفضل 6 أرقام: `{top_6}` ⬅️ أصابت **{h6}**")
            st.write(f"🔹 التغطية بـ 12 رقم: `{top_12}` ⬅️ أصابت **{h12}** من 6")
            st.write(f"🔥 **التغطية بـ 18 رقم (الذهبية):** `{top_18}` ⬅️ أصابت **{h18}** من 6! 🎉")

# 3. Training Engine
elif menu == "⚙️ مشغل التدريب وحفظ الأفضل":
    st.header("⚡ مشغل التدريب وحفظ النموذج ذو الدقة الأعلى تلقائياً")
    st.info("النماذج الآن لا تضيع! أي جولة تحقق نتيجة أعلى من السابق سيتم حفظها تلقائياً بملف النخبة `best_lotto_model.pkl`.")
    
    run_loop = st.checkbox("🔥 ابدأ البحث وتحديث النموذج الأفضل")
    
    if run_loop:
        st_log = st.empty()
        chart_holder = st.empty()
        
        X, y, _ = build_advanced_features(df_all)
        scores = []
        
        best_score = 0.0
        if os.path.exists(MODEL_FILE):
            try:
                with open(MODEL_FILE, 'rb') as f:
                    best_score = pickle.load(f).get('score', 0.0)
            except:
                best_score = 0.0
                
        iteration = 0
        while run_loop:
            iteration += 1
            n_est = np.random.randint(100, 250)
            m_depth = np.random.randint(8, 22)
            
            clf = RandomForestClassifier(n_estimators=n_est, max_depth=m_depth)
            clf.fit(X[:-60], y[:-60])
            
            # True accuracy estimation over last 60 draws
            preds = clf.predict_proba(X[-60:])
            hits = 0
            for idx_draw in range(60):
                p_row = np.zeros(42)
                for b_idx in range(42):
                    if preds[b_idx].shape[1] > 1:
                        p_row[b_idx] = preds[b_idx][idx_draw, 1]
                t18 = set(np.argsort(p_row)[-18:] + 1)
                actual_b = set(np.where(y[-60:][idx_draw] == 1)[0] + 1)
                hits += len(t18.intersection(actual_b))
                
            cur_score = (hits / (60 * 6)) * 100
            scores.append(cur_score)
            
            is_new_best = False
            if cur_score > best_score:
                best_score = cur_score
                is_new_best = True
                with open(MODEL_FILE, 'wb') as f:
                    pickle.dump({'model': clf, 'score': best_score}, f)
                    
            msg = f"### 🔄 جولة: `{iteration}` | أشجار: `{n_est}` | عمق: `{m_depth}` | الدقة الحالية: **{cur_score:.2f}%** | 🏆 الأفضل: **{best_score:.2f}%**"
            if is_new_best:
                msg += " 👑 **(تم حفظ كسر رقم قياسي جديد!)**"
            st_log.markdown(msg)
            
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(scores[-30:], color='teal', marker='o')
            ax.axhline(best_score, color='gold', linestyle='--', label='Record High')
            ax.set_title("مراقبة الأداء وحفظ الأرقام القياسية")
            chart_holder.pyplot(fig)
            time.sleep(1)

# 4. Golden Coverage
elif menu == "🎫 توليد القائمة الذهبية والبطاقات":
    st.header("🎫 توليد القائمة الذهبية والبطاقات (مستخرجة من أفضل نموذج)")
    
    if st.button("👑 استخراج من النموذج القياسي المحفوظ"):
        if os.path.exists(MODEL_FILE):
            with open(MODEL_FILE, 'rb') as f:
                s_data = pickle.load(f)
                clf = s_data['model']
                rec_score = s_data['score']
            st.success(f"تم تحميل النموذج القياسي المعتمد برقم قياسي: **{rec_score:.2f}%**")
        else:
            X, y, _ = build_advanced_features(df_all)
            clf = RandomForestClassifier(n_estimators=180, max_depth=14, random_state=42)
            clf.fit(X, y)
            st.warning("لم يتم العثور على نموذج قياسي سابق، تم تدريب نموذج افتراضي.")
            
        X, y, matrix = build_advanced_features(df_all)
        last_i = len(df_all)
        freq_s = matrix[last_i-5:last_i].sum(axis=0)
        freq_m = matrix[last_i-15:last_i].sum(axis=0)
        freq_l = matrix[last_i-30:last_i].sum(axis=0)
        delays = np.zeros(42)
        for b in range(42):
            sub = matrix[:last_i, b]
            pos = np.where(sub == 1)[0]
            delays[b] = (last_i - pos[-1]) if len(pos) > 0 else 30
            
        X_next = np.concatenate([freq_s, freq_m, freq_l, delays, matrix[-1], matrix[-2]]).reshape(1, -1)
        probas = clf.predict_proba(X_next)
        probs = np.zeros(42)
        for idx in range(42):
            if probas[idx].shape[1] > 1:
                probs[idx] = probas[idx][0, 1]
                
        top_18 = clean_list(sorted(np.argsort(probs)[-18:] + 1))
        st.success(f"👑 **الـ 18 رقم الذهبية النهائية المستخرجة:**\n\n`{top_18}`")
        st.markdown("---")
        
        tickets = []
        attempts = 0
        np.random.seed(42)
        while len(tickets) < 6 and attempts < 2000:
            attempts += 1
            combo = clean_list(sorted(np.random.choice(top_18, 6, replace=False)))
            
            c_sum = sum(combo)
            odds = sum(1 for x in combo if x % 2 != 0)
            diffs = [combo[k+1] - combo[k] for k in range(5)]
            has_triple_seq = any(diffs[k]==1 and diffs[k+1]==1 for k in range(4))
            
            if 105 <= c_sum <= 155 and 2 <= odds <= 4 and not has_triple_seq:
                if combo not in tickets:
                    tickets.append(combo)
                    
        st.subheader("🎯 البطاقات الـ 6 المستخرجة والمفلترة للبطاقات:")
        for idx, t in enumerate(tickets, 1):
            st.info(f"🎟️ **البطاقة {idx}:** `{t}`  |  المجموع: **{sum(t)}** | فردي/زوجي: **{sum(1 for x in t if x%2!=0)}/{6-sum(1 for x in t if x%2!=0)}**")
