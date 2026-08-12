import sqlite3
import numpy as np
import os
import pickle
import time
import csv
import torch
import torch.nn as nn
import torch.optim as optim

CSV_FILE = 'lotto_data.csv'
DB_FILE = 'lotto_master.db'
MODEL_FILE = "best_lotto_model.pkl"

def initialize_db():
    if not os.path.exists(DB_FILE):
        print("📥 جاري إنشاء قاعدة البيانات من ملف الـ CSV...")
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE IF NOT EXISTS draws (draw_id INTEGER, ball1 INT, ball2 INT, ball3 INT, ball4 INT, ball5 INT, ball6 INT, bonus INT)")
        
        if os.path.exists(CSV_FILE):
            # تجربة قراءة الملف بترميزات مختلفة لتفادي خطأ Decode
            encodings = ['utf-8', 'utf-8-sig', 'latin1', 'cp1256']
            lines = None
            for enc in encodings:
                try:
                    with open(CSV_FILE, 'r', encoding=enc, errors='ignore') as f:
                        lines = f.readlines()
                    break
                except Exception:
                    continue
            
            if lines:
                reader = csv.reader(lines)
                next(reader, None) # تخطي رأس الجدول
                count = 0
                for row in reader:
                    if len(row) >= 7: # التأكد من وجود الأرقام
                        try:
                            # تنظيف البيانات واستخراج الأرقام فقط
                            clean_row = [int(float(str(x).strip())) for x in row[:8] if str(x).strip().isdigit()]
                            if len(clean_row) >= 7:
                                if len(clean_row) == 7:
                                    clean_row.append(0) # إضافة bonus افتراضي إذا غير موجود
                                cursor.execute("INSERT INTO draws VALUES (?,?,?,?,?,?,?,?)", clean_row[:8])
                                count += 1
                        except ValueError:
                            continue
                conn.commit()
                print(f"✅ تم استيراد {count} سحبة بنجاح!")
            else:
                print("❌ فشل قراءة ملف الـ CSV.")
        else:
            print("⚠️ تحذير: ملف lotto_data.csv غير موجود.")
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
    print("🚀 بدء محرك التعلم العميق (PyTorch) - وضع بدون مكتبات خارجية")
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
