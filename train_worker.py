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
            with open(CSV_FILE, 'r') as f:
                reader = csv.reader(f)
                next(reader) # تخطي العنوان
                for row in reader:
                    cursor.execute("INSERT INTO draws VALUES (?,?,?,?,?,?,?,?)", row)
            conn.commit()
            print("✅ تم استيراد البيانات بنجاح!")
        else:
            print("⚠️ تحذير: ملف lotto_data.csv غير موجود، سيتم تشغيل المحرك بقاعدة بيانات فارغة.")
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
        return self.sigmoid(self.fc3(self.relu(self.fc2(self.relu(self.fc1(x))))))

def run_continuous_training():
    print("🚀 بدء محرك التعلم العميق (PyTorch) - وضع بدون مكتبات خارجية")
    initialize_db()
    
    draws = load_draws_numpy()
    if len(draws) < 30:
        print("❌ البيانات غير كافية للتدريب (أقل من 30 سحبة).")
        return

    # بناء الميزات (Features)
    num_draws = len(draws)
    matrix = np.zeros((num_draws, 42), dtype=float)
    for i, row in enumerate(draws):
        for ball in row:
            if 0 < ball <= 42:
                matrix[i, ball-1] = 1.0

    X, y = [], []
    for i in range(30, num_draws):
        features = np.concatenate([matrix[i-5:].sum(axis=0)/5, matrix[i-15:].sum(axis=0)/15, matrix[i-30:].sum(axis=0)/30, matrix[i-1]])
        X.append(features)
        y.append(matrix[i])
    
    X_tensor = torch.tensor(np.array(X, dtype=np.float32))
    y_tensor = torch.tensor(np.array(y, dtype=np.float32))
    
    model = LottoNeuralNet(X_tensor.shape[1], 128)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    
    while True:
        optimizer.zero_grad()
        output = model(X_tensor)
        loss = criterion(output, y_tensor)
        loss.backward()
        optimizer.step()
        
        print(f"🔄 جاري التدريب... الخسارة (Loss): {loss.item():.4f}")
        time.sleep(2)

if __name__ == "__main__":
    run_continuous_training()
