import sqlite3
import numpy as np
import os
import pickle
import time
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

CSV_FILE = 'lotto_final_2439.csv'
DB_FILE = 'lotto_master.db'
MODEL_FILE = "best_lotto_model.pkl"

def initialize_db():
    if not os.path.exists(DB_FILE):
        print("📥 جاري إنشاء قاعدة البيانات من الملف المرفق...")
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE IF NOT EXISTS draws (draw_id INTEGER PRIMARY KEY AUTOINCREMENT, ball1 INT, ball2 INT, ball3 INT, ball4 INT, ball5 INT, ball6 INT, bonus INT)")
        
        # محاولة قراءة الملف سواء كان Excel أو CSV
        df = None
        try:
            df = pd.read_excel(CSV_FILE)
        except Exception:
            try:
                df = pd.read_csv(CSV_FILE)
            except Exception as e:
                print(f"❌ فشل قراءة الملف: {e}")

        if df is not None:
            count = 0
            for _, row in df.iterrows():
                try:
                    # قراءة أرقام الكرات الـ 6 والـ Bonus
                    b1 = int(row['Ball 1'])
                    b2 = int(row['Ball 2'])
                    b3 = int(row['Ball 3'])
                    b4 = int(row['Ball 4'])
                    b5 = int(row['Ball 5'])
                    b6 = int(row['Ball 6'])
                    bonus = int(row['Bonus']) if 'Bonus' in row and pd.notnull(row['Bonus']) else 0
                    
                    cursor.execute("INSERT INTO draws (ball1, ball2, ball3, ball4, ball5, ball6, bonus) VALUES (?,?,?,?,?,?,?)", 
                                   (b1, b2, b3, b4, b5, b6, bonus))
                    count += 1
                except Exception:
                    continue
                    
            conn.commit()
            print(f"✅ تم استيراد {count} سحبة بنجاح!")
        else:
            print("⚠️ تحذير: تعذر استخراج البيانات من الملف.")
        conn.close()

def load_draws_numpy():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT ball1, ball2, ball3, ball4, ball5, ball6 FROM draws ORDER BY draw_id ASC")
    rows = cursor.fetchall()
    conn.close()
    return np.array(rows)

class LottoNeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LottoNeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, 42)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return self.sigmoid(out)

def run_continuous_training():
    print("🚀 بدء محرك التعلم العميق (PyTorch)")
    initialize_db()
    
    draws = load_draws_numpy()
    if len(draws) < 30:
        print(f"❌ البيانات غير كافية للتدريب (تم العثور على {len(draws)} سحبة فقط).")
        return

    print(f"📊 جاري تدريب الشبكة العصبية على {len(draws)} سحبة...")
    num_draws = len(draws)
    matrix = np.zeros((num_draws, 42), dtype=float)
    for i, row in enumerate(draws):
        for ball in row:
            if 0 < ball <= 42:
                matrix[i, ball-1] = 1.0

    X, y = [], []
    for i in range(30, num_draws):
        features = np.concatenate([
            matrix[i-5:i].sum(axis=0)/5.0,
            matrix[i-15:i].sum(axis=0)/15.0,
            matrix[i-30:i].sum(axis=0)/30.0,
            matrix[i-1]
        ])
        X.append(features)
        y.append(matrix[i])
    
    X_tensor = torch.tensor(np.array(X, dtype=np.float32))
    y_tensor = torch.tensor(np.array(y, dtype=np.float32))
    
    model = LottoNeuralNet(X_tensor.shape[1], 128)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    
    best_loss = float('inf')
    iteration = 0
    
    while True:
        iteration += 1
        model.train()
        optimizer.zero_grad()
        output = model(X_tensor)
        loss = criterion(output, y_tensor)
        loss.backward()
        optimizer.step()
        
        cur_loss = loss.item()
        if cur_loss < best_loss:
            best_loss = cur_loss
            with open(MODEL_FILE, 'wb') as f:
                pickle.dump({'model_state': model.state_dict(), 'loss': best_loss}, f)
            print(f"🔥 [جولة {iteration}] كسر رقم قياسي! الخسارة (Loss): {best_loss:.4f}")
        elif iteration % 10 == 0:
            print(f"🔄 [جولة {iteration}] الخسارة الحالية: {cur_loss:.4f} | الأفضل: {best_loss:.4f}")
            
        time.sleep(0.5)

if __name__ == "__main__":
    run_continuous_training()
