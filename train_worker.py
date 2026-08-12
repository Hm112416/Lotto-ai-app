import sqlite3
import numpy as np
import os
import pickle
import time
import torch
import torch.nn as nn
import torch.optim as optim

MODEL_FILE = "best_lotto_model.pkl"

def get_connection():
    return sqlite3.connect('lotto_master.db')

def load_draws_numpy():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT ball1, ball2, ball3, ball4, ball5, ball6 FROM draws ORDER BY draw_id ASC")
    rows = cursor.fetchall()
    conn.close()
    
    # تحويل السحوبات مباشرة إلى NumPy Array
    draws = []
    for r in rows:
        draws.append([int(b) for b in r])
    return np.array(draws)

def build_features(draws):
    num_draws = len(draws)
    matrix = np.zeros((num_draws, 42), dtype=float)
    
    for i, row in enumerate(draws):
        for ball in row:
            b_idx = int(ball) - 1
            if 0 <= b_idx < 42:
                matrix[i, b_idx] = 1.0
                
    X, y = [], []
    lag_s, lag_m, lag_l = 5, 15, 30
    
    for i in range(lag_l, num_draws):
        freq_s = matrix[i-lag_s:i].sum(axis=0) / lag_s
        freq_m = matrix[i-lag_m:i].sum(axis=0) / lag_m
        freq_l = matrix[i-lag_l:i].sum(axis=0) / lag_l
        
        delays = np.zeros(42)
        for b in range(42):
            pos = np.where(matrix[:i, b] == 1)[0]
            delays[b] = (i - pos[-1]) if len(pos) > 0 else lag_l
        delays = delays / 42.0
        
        features = np.concatenate([freq_s, freq_m, freq_l, delays, matrix[i-1], matrix[i-2]])
        X.append(features)
        y.append(matrix[i])
        
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

class LottoNeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LottoNeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, 42)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return self.sigmoid(out)

def run_continuous_training():
    print("🚀 بدء محرك التعلم العميق (PyTorch Deep Learning) 24/24...")
    
    best_score = 0.0
    if os.path.exists(MODEL_FILE):
        try:
            with open(MODEL_FILE, 'rb') as f:
                best_score = pickle.load(f).get('score', 0.0)
            print(f"🏆 الرقم القياسي السابق: {best_score:.2f}%")
        except:
            best_score = 0.0

    draws = load_draws_numpy()
    X, y = build_features(draws)
    
    X_tensor = torch.tensor(X)
    y_tensor = torch.tensor(y)
    
    X_train, y_train = X_tensor[:-60], y_tensor[:-60]
    X_val, y_val = X_tensor[-60:], y_tensor[-60:]
    
    input_dim = X.shape[1]
    iteration = 0
    
    while True:
        iteration += 1
        hidden_size = int(np.random.choice([64, 128, 256]))
        lr = float(np.random.choice([0.005, 0.001, 0.0005]))
        
        model = LottoNeuralNet(input_dim, hidden_size)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        model.train()
        for epoch in range(40):
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
            
        model.eval()
        with torch.no_grad():
            preds_val = model(X_val).numpy()
            y_val_np = y_val.numpy()
            
            hits = 0
            for idx in range(60):
                p_row = preds_val[idx]
                top18 = set(np.argsort(p_row)[-18:] + 1)
                actual = set(np.where(y_val_np[idx] == 1)[0] + 1)
                hits += len(top18.intersection(actual))
                
            cur_score = (hits / (60 * 6)) * 100
            
            if cur_score > best_score:
                best_score = cur_score
                save_data = {
                    'model_state': model.state_dict(),
                    'hidden_size': hidden_size,
                    'input_dim': input_dim,
                    'score': best_score
                }
                with open(MODEL_FILE, 'wb') as f:
                    pickle.dump(save_data, f)
                print(f"🔥 [جولة {iteration}] كسر رقم قياسي جديد! الدقة: {best_score:.2f}% (تم حفظ نموذج PyTorch)")
            else:
                if iteration % 5 == 0:
                    print(f"🔄 [جولة {iteration}] الدقة الحالية: {cur_score:.2f}% | الأفضل: {best_score:.2f}%")
                    
        time.sleep(0.5)

if __name__ == "__main__":
    run_continuous_training()
