"""
Lab 5 Example 1: การสร้าง Training Loop พื้นฐาน
เรียนรู้หลักการ training neural network และการใช้งาน optimizer
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

print("=" * 60)
print("Lab 5 Example 1: Training Loop พื้นฐาน")
print("=" * 60)

# 1. สร้าง Simple Neural Network
class SimpleClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleClassifier, self).__init__()
        
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size // 2)
        self.layer3 = nn.Linear(hidden_size // 2, num_classes)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.dropout(x)
        x = F.relu(self.layer2(x))
        x = self.dropout(x)
        x = self.layer3(x)
        return x

# 2. สร้างข้อมูลตัวอย่างสำหรับ classification
def create_sample_data(num_samples=1000, input_size=10, num_classes=3):
    """สร้างข้อมูลตัวอย่างสำหรับทดสอบ"""
    
    # สร้างข้อมูล features
    X = torch.randn(num_samples, input_size)
    
    # สร้าง labels แบบง่าย (based on sum of first few features)
    feature_sum = X[:, :3].sum(dim=1)
    y = torch.zeros(num_samples, dtype=torch.long)
    
    # กำหนด class ตาม threshold
    y[feature_sum < -1] = 0
    y[(feature_sum >= -1) & (feature_sum < 1)] = 1  
    y[feature_sum >= 1] = 2
    
    return X, y

# 3. Basic Training Loop Function
def basic_training_loop(model, train_loader, optimizer, criterion, num_epochs=10):
    """Training loop พื้นฐาน"""
    
    model.train()  # ตั้งโหมด training
    loss_history = []
    
    print("เริ่มการ training...")
    print("-" * 40)
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            # 1. Reset gradients
            optimizer.zero_grad()
            
            # 2. Forward pass
            output = model(data)
            
            # 3. คำนวณ loss
            loss = criterion(output, target)
            
            # 4. Backward pass
            loss.backward()
            
            # 5. Update weights
            optimizer.step()
            
            # เก็บสถิติ
            total_loss += loss.item()
            num_batches += 1
            
            # แสดงผล progress
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        # คำนวณ average loss ของ epoch
        avg_loss = total_loss / num_batches
        loss_history.append(avg_loss)
        
        print(f'Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}')
        print("-" * 40)
    
    return loss_history

# 4. ทดสอบการใช้งาน
print("\n1. สร้างข้อมูลและ model")
print("-" * 30)

# สร้างข้อมูล
X_train, y_train = create_sample_data(num_samples=800, input_size=10, num_classes=3)
print(f"Training data shape: X={X_train.shape}, y={y_train.shape}")

# แสดงการกระจายของ classes
unique, counts = torch.unique(y_train, return_counts=True)
print(f"Class distribution: {dict(zip(unique.tolist(), counts.tolist()))}")

# สร้าง DataLoader
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
print(f"Number of batches: {len(train_loader)}")

# สร้าง model
model = SimpleClassifier(input_size=10, hidden_size=64, num_classes=3)
print(f"\nModel architecture:")
print(model)

# นับจำนวน parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params}")
print(f"Trainable parameters: {trainable_params}")

# 5. ทดสอบ Optimizers ต่างๆ
print("\n2. ทดสอบ Optimizers ต่างๆ")
print("-" * 35)

optimizers_config = [
    ("SGD", optim.SGD(model.parameters(), lr=0.01)),
    ("Adam", optim.Adam(model.parameters(), lr=0.001)),
    ("RMSprop", optim.RMSprop(model.parameters(), lr=0.001))
]

criterion = nn.CrossEntropyLoss()

# ทดสอบแต่ละ optimizer
for opt_name, optimizer in optimizers_config:
    print(f"\n--- Testing {opt_name} Optimizer ---")
    
    # Reset model weights
    def reset_weights(m):
        if isinstance(m, nn.Linear):
            m.reset_parameters()
    
    model.apply(reset_weights)
    
    # Training
    loss_history = basic_training_loop(
        model, train_loader, optimizer, criterion, num_epochs=5
    )
    
    print(f"{opt_name} final loss: {loss_history[-1]:.4f}")

# 6. ทดสอบ Learning Rates ต่างๆ
print("\n3. ทดสอบ Learning Rates ต่างๆ")
print("-" * 35)

learning_rates = [0.1, 0.01, 0.001, 0.0001]
lr_results = {}

for lr in learning_rates:
    print(f"\n--- Testing Learning Rate: {lr} ---")
    
    # Reset model
    model.apply(reset_weights)
    
    # สร้าง optimizer ใหม่ด้วย lr ใหม่
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training
    loss_history = basic_training_loop(
        model, train_loader, optimizer, criterion, num_epochs=3
    )
    
    lr_results[lr] = loss_history
    print(f"LR {lr} final loss: {loss_history[-1]:.4f}")

# 7. Step-by-step Training Process
print("\n4. Step-by-step Training Process")
print("-" * 35)

# Reset model
model.apply(reset_weights)
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("แสดงขั้นตอนการ training แบบละเอียด:")

# ใช้ batch เดียวเพื่อดูขั้นตอน
sample_batch = next(iter(train_loader))
data, target = sample_batch

print(f"Batch shape: {data.shape}, Target shape: {target.shape}")

# Before training
model.eval()
with torch.no_grad():
    initial_output = model(data)
    initial_loss = criterion(initial_output, target)
    print(f"Initial loss: {initial_loss.item():.4f}")

# Training step by step
model.train()

for step in range(5):
    print(f"\n--- Training Step {step + 1} ---")
    
    # 1. Zero gradients
    print("1. Clearing previous gradients...")
    optimizer.zero_grad()
    
    # 2. Forward pass
    print("2. Forward pass...")
    output = model(data)
    
    # 3. Compute loss
    print("3. Computing loss...")
    loss = criterion(output, target)
    print(f"   Loss: {loss.item():.4f}")
    
    # 4. Backward pass
    print("4. Backward pass (computing gradients)...")
    loss.backward()
    
    # แสดง gradient ของ layer แรก
    first_layer_grad = model.layer1.weight.grad
    if first_layer_grad is not None:
        grad_norm = first_layer_grad.norm().item()
        print(f"   Gradient norm (layer1): {grad_norm:.4f}")
    
    # 5. Update weights
    print("5. Updating weights...")
    optimizer.step()
    
    # แสดงผลการเปลี่ยนแปลง
    with torch.no_grad():
        new_output = model(data)
        new_loss = criterion(new_output, target)
        print(f"   New loss: {new_loss.item():.4f}")

# 8. Loss Function Comparison
print("\n5. เปรียบเทียบ Loss Functions")
print("-" * 35)

loss_functions = [
    ("CrossEntropyLoss", nn.CrossEntropyLoss()),
    ("NLLLoss", nn.NLLLoss()),  # ต้องใช้กับ log_softmax
]

# สำหรับ NLLLoss เราต้องแก้ model output
class ModifiedClassifier(SimpleClassifier):
    def forward(self, x):
        x = super().forward(x)
        return F.log_softmax(x, dim=1)  # สำหรับ NLLLoss

for loss_name, loss_fn in loss_functions:
    print(f"\n--- Testing {loss_name} ---")
    
    if loss_name == "NLLLoss":
        test_model = ModifiedClassifier(input_size=10, hidden_size=64, num_classes=3)
    else:
        test_model = SimpleClassifier(input_size=10, hidden_size=64, num_classes=3)
    
    optimizer = optim.Adam(test_model.parameters(), lr=0.001)
    
    # Training สั้นๆ
    loss_history = basic_training_loop(
        test_model, train_loader, optimizer, loss_fn, num_epochs=2
    )
    
    print(f"{loss_name} final loss: {loss_history[-1]:.4f}")

# 9. Gradient Visualization
print("\n6. การติดตาม Gradients")
print("-" * 30)

def track_gradients(model, data, target, criterion):
    """ติดตาม gradients ของแต่ละ layer"""
    
    model.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    
    gradient_info = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            gradient_info[name] = grad_norm
    
    return gradient_info, loss.item()

# ติดตาม gradients
model.apply(reset_weights)
sample_data, sample_target = next(iter(train_loader))

grad_info, loss_val = track_gradients(model, sample_data, sample_target, criterion)

print(f"Current loss: {loss_val:.4f}")
print("Gradient norms:")
for param_name, grad_norm in grad_info.items():
    print(f"  {param_name}: {grad_norm:.6f}")

print("\n" + "=" * 60)
print("สรุป Example 1: Training Loop พื้นฐาน")
print("=" * 60)
print("✓ สร้าง training loop พื้นฐาน")
print("✓ ทดสอบ optimizers ต่างๆ (SGD, Adam, RMSprop)")
print("✓ ทดสอบ learning rates ต่างๆ")
print("✓ แสดงขั้นตอน training แบบละเอียด")
print("✓ เปรียบเทียบ loss functions")
print("✓ ติดตาม gradients")
print("=" * 60)