import sqlite3
import numpy as np
import os
import pickle
import time
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd # بنستخدم pandas هون فقط للتحويل لمرة واحدة

EXCEL_FILE = 'lotto_final_2439.xlsx'
DB_FILE = 'lotto_master.db'
MODEL_FILE = "best_lotto_model.pkl"

def initialize_db():
    if not os.path.exists(DB_FILE):
        print("📥 جاري استيراد البيانات من الإكسل إلى قاعدة البيانات...")
        df = pd.read_excel(EXCEL_FILE)
        # التأكد من أسماء الأعمدة (تعديل حسب ملفك إذا لزم)
        df.columns = ['draw_id', 'ball1', 'ball2', 'ball3', 'ball4', 'ball5', 'ball6', 'bonus']
        conn = sqlite3.connect(DB_FILE)
        df.to_sql('draws', conn, if_exists='replace', index=False)
        conn.close()
        print("✅ تم إنشاء قاعدة البيانات بنجاح!")

def load_draws_numpy():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT ball1, ball2, ball3, ball4, ball5, ball6 FROM draws ORDER BY draw_id ASC")
    rows = cursor.fetchall()
    conn.close()
    return np.array(rows)

# [نفس دوال الشبكة العصبية والتدريب السابقة...]
# (يمكنك نسخ دوال LottoNeuralNet و build_features و run_continuous_training من الرد السابق)

if __name__ == "__main__":
    initialize_db() # الخطوة الجديدة
    run_continuous_training()
