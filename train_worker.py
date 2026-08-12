import sqlite3
import numpy as np
import os
import pickle
import time
import csv
import re
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
            encodings = ['utf-8', 'utf-8-sig', 'latin1', 'cp1256']
            lines = None
            for enc in encodings:
                try:
                    with open(CSV_FILE, 'r', encoding=enc, errors='ignore') as f:
                        lines = f.readlines()
                    if lines:
                        break
                except Exception:
                    continue
            
            if lines:
                count = 0
                for line in lines:
                    # استخراج كل الأرقام من السطر بغض النظر عن الفواصل (comma, semicolon, spaces)
                    nums = [int(n) for n in re.findall(r'\b\d+\b', line)]
                    
                    # إذا كان السطر يحتوي على 7 أرقام أو أكثر (draw_id + 6 balls)
                    if len(nums) >= 7:
                        draw_id = nums[0]
                        balls = nums[1:7]
                        bonus = nums[7] if len(nums) >= 8 else 0
                        
                        # التأكد من أن الكرات أرقامها ضمن نطاق اللوتو المعقول
                        if all(0 < b <= 42 for b in balls):
                            cursor.execute("INSERT INTO draws VALUES (?,?,?,?,?,?,?,?)", [draw_id] + balls + [bonus])
                            count += 1
                            
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
