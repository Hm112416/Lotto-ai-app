import sqlite3
import pandas as pd
import numpy as np
import os
import pickle
import time
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

EXCEL_PATH = 'lotto_final_2439.xlsx'
MODEL_FILE = "best_lotto_model.pkl"

def get_connection():
    return sqlite3.connect('lotto_master.db')

def load_draws_df():
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM draws ORDER BY draw_id ASC", conn)
    conn.close()
    main_cols = ['ball1', 'ball2', 'ball3', 'ball4', 'ball5', 'ball6', 'bonus']
    for col in main_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(1).astype(int)
    return df

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
        
    return np.array(X), np.array(y)

def run_continuous_training():
    print("🚀 بدء محرك التدريب التراكمي المستمر 24/24...")
    
    best_score = 0.0
    if os.path.exists(MODEL_FILE):
        try:
            with open(MODEL_FILE, 'rb') as f:
                best_score = pickle.load(f).get('score', 0.0)
            print(f"🏆 الرقم القياسي السابق المحفوظ: {best_score:.2f}%")
        except:
            best_score = 0.0

    df = load_draws_df()
    X, y = build_advanced_features(df)
    
    iteration = 0
    while True:
        iteration += 1
        n_est = np.random.randint(100, 300)
        m_depth = np.random.randint(8, 25)
        
        # اختيار عشوائي بين الخوارزميتين للتنويع
        if np.random.rand() > 0.5:
            clf = RandomForestClassifier(n_estimators=n_est, max_depth=m_depth)
        else:
            clf = RandomForestClassifier(n_estimators=n_est, max_depth=m_depth, criterion='entropy')
            
        clf.fit(X[:-60], y[:-60])
        
        # تقييم الدقة بصرامة على آخر 60 سحبة
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
        
        if cur_score > best_score:
            best_score = cur_score
            with open(MODEL_FILE, 'wb') as f:
                pickle.dump({'model': clf, 'score': best_score, 'time': time.strftime("%Y-%m-%d %H:%M")}, f)
            print(f"🔥 [جولة {iteration}] كسر رقم قياسي جديد! الدقة: {best_score:.2f}% وتم حفظ النموذج!")
        else:
            print(f"🔄 [جولة {iteration}] الدقة: {cur_score:.2f}% | الأفضل حتى الآن: {best_score:.2f}%")
            
        time.sleep(2)

if __name__ == "__main__":
    run_continuous_training()
