"""
Lab 5 Example 2: Backward Propagation และ Gradient Updates
เรียนรู้หลักการ backpropagation, gradient computation และ optimization techniques
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

print("=" * 60)
print("Lab 5 Example 2: Backward Propagation และ Gradient Updates")
print("=" * 60)

# 1. Manual Backpropagation Example
class ManualBackpropNet(nn.Module):
    """Network สำหรับแสดงการทำ backpropagation แบบ manual"""
    
    def __init__(self, input_size, hidden_size, output_size):
        super(ManualBackpropNet, self).__init__()
        
        # สร้าง parameters แบบ manual
        self.W1 = nn.Parameter(torch.randn(input_size, hidden_size) * 0.1)
        self.b1 = nn.Parameter(torch.zeros(hidden_size))
        self.W2 = nn.Parameter(torch.randn(hidden_size, output_size) * 0.1)
        self.b2 = nn.Parameter(torch.zeros(output_size))
        
        # เก็บ intermediate values สำหรับ manual backprop
        self.z1 = None
        self.a1 = None
        self.z2 = None
        
    def forward(self, x):
        # Forward pass พร้อมเก็บ intermediate values
        self.z1 = torch.matmul(x, self.W1) + self.b1  # Linear transformation
        self.a1 = torch.relu(self.z1)                  # ReLU activation
        self.z2 = torch.matmul(self.a1, self.W2) + self.b2  # Output layer
        
        return self.z2
    
    def manual_backward(self, x, y_true, y_pred):
        """Manual backpropagation implementation"""
        
        batch_size = x.size(0)
        
        # Compute loss gradient (for MSE loss)
        dL_dz2 = 2 * (y_pred - y_true) / batch_size
        
        # Gradients for output layer
        dL_dW2 = torch.matmul(self.a1.t(), dL_dz2)
        dL_db2 = torch.sum(dL_dz2, dim=0)
        
        # Backpropagate to hidden layer
        dL_da1 = torch.matmul(dL_dz2, self.W2.t())
        
        # ReLU derivative
        dL_dz1 = dL_da1 * (self.z1 > 0).float()
        
        # Gradients for hidden layer
        dL_dW1 = torch.matmul(x.t(), dL_dz1)
        dL_db1 = torch.sum(dL_dz1, dim=0)
        
        return {
            'dL_dW1': dL_dW1,
            'dL_db1': dL_db1,
            'dL_dW2': dL_dW2,
            'dL_db2': dL_db2
        }

# 2. Gradient Comparison Function
def compare_gradients(model, x, y):
    """เปรียบเทียบ gradients จาก automatic กับ manual backprop"""
    
    # Automatic backpropagation
    model.zero_grad()
    y_pred = model(x)
    loss = F.mse_loss(y_pred, y)
    loss.backward()
    
    # เก็บ automatic gradients
    auto_grads = {
        'dL_dW1': model.W1.grad.clone(),
        'dL_db1': model.b1.grad.clone(),
        'dL_dW2': model.W2.grad.clone(),
        'dL_db2': model.b2.grad.clone()
    }
    
    # Manual backpropagation
    y_pred_manual = model(x)  # Forward pass อีกครั้ง
    manual_grads = model.manual_backward(x, y, y_pred_manual)
    
    # เปรียบเทียบ
    print("Gradient Comparison (Automatic vs Manual):")
    print("-" * 50)
    
    for key in auto_grads:
        auto_grad = auto_grads[key]
        manual_grad = manual_grads[key]
        
        # คำนวณความแตกต่าง
        diff = torch.abs(auto_grad - manual_grad).max().item()
        relative_diff = diff / (torch.abs(auto_grad).max().item() + 1e-8)
        
        print(f"{key}:")
        print(f"  Max absolute difference: {diff:.8f}")
        print(f"  Max relative difference: {relative_diff:.8f}")
        print(f"  Are close? {torch.allclose(auto_grad, manual_grad, atol=1e-6)}")

# 3. Gradient Visualization
class GradientTracker:
    """Class สำหรับติดตาม gradients ระหว่าง training"""
    
    def __init__(self):
        self.gradient_history = {}
        
    def track_gradients(self, model, step):
        """เก็บ gradient norms ของแต่ละ parameter"""
        
        if step not in self.gradient_history:
            self.gradient_history[step] = {}
            
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                self.gradient_history[step][name] = grad_norm
    
    def plot_gradient_flow(self):
        """วาดกราฟแสดงการไหลของ gradients"""
        
        if not self.gradient_history:
            print("No gradient history to plot")
            return
            
        steps = list(self.gradient_history.keys())
        param_names = list(self.gradient_history[steps[0]].keys())
        
        plt.figure(figsize=(12, 6))
        
        for param_name in param_names:
            grad_norms = [self.gradient_history[step][param_name] for step in steps]
            plt.plot(steps, grad_norms, label=param_name, marker='o')
        
        plt.xlabel('Training Step')
        plt.ylabel('Gradient Norm')
        plt.title('Gradient Flow During Training')
        plt.legend()
        plt.grid(True)
        plt.yscale('log')
        plt.show()

# 4. Advanced Gradient Techniques
class GradientClippingDemo(nn.Module):
    """Demo การใช้ gradient clipping"""
    
    def __init__(self, input_size, hidden_size, output_size):
        super(GradientClippingDemo, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        
    def forward(self, x):
        return self.layers(x)

def training_with_gradient_clipping(model, train_loader, optimizer, criterion, 
                                  max_norm=1.0, num_epochs=5):
    """Training loop พร้อม gradient clipping"""
    
    gradient_tracker = GradientTracker()
    
    model.train()
    step = 0
    
    for epoch in range(num_epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # Track gradients before clipping
            gradient_tracker.track_gradients(model, step)
            
            # Gradient clipping
            grad_norm_before = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            
            optimizer.step()
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}, '
                      f'Grad Norm: {grad_norm_before:.4f}')
            
            step += 1
    
    return gradient_tracker

# 5. Optimizer Comparison with Gradients
def detailed_optimizer_comparison():
    """เปรียบเทียบ optimizers แบบละเอียด"""
    
    # สร้างข้อมูลทดสอบ
    X = torch.randn(100, 5)
    y = torch.randn(100, 1)
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=20)
    
    optimizers_config = [
        ("SGD", lambda params: optim.SGD(params, lr=0.01)),
        ("SGD+Momentum", lambda params: optim.SGD(params, lr=0.01, momentum=0.9)),
        ("Adam", lambda params: optim.Adam(params, lr=0.001)),
        ("AdaGrad", lambda params: optim.Adagrad(params, lr=0.01)),
        ("RMSprop", lambda params: optim.RMSprop(params, lr=0.001))
    ]
    
    results = {}
    
    for opt_name, opt_fn in optimizers_config:
        print(f"\n--- Testing {opt_name} ---")
        
        # สร้าง model ใหม่
        model = nn.Sequential(
            nn.Linear(5, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )
        
        optimizer = opt_fn(model.parameters())
        criterion = nn.MSELoss()
        
        # Training
        model.train()
        losses = []
        gradient_norms = []
        
        for epoch in range(10):
            epoch_loss = 0
            epoch_grad_norm = 0
            
            for data, target in loader:
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                
                # คำนวณ gradient norm
                total_norm = 0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                grad_norm = total_norm ** (1. / 2)
                
                optimizer.step()
                
                epoch_loss += loss.item()
                epoch_grad_norm += grad_norm
            
            avg_loss = epoch_loss / len(loader)
            avg_grad_norm = epoch_grad_norm / len(loader)
            
            losses.append(avg_loss)
            gradient_norms.append(avg_grad_norm)
        
        results[opt_name] = {
            'losses': losses,
            'gradient_norms': gradient_norms,
            'final_loss': losses[-1]
        }
        
        print(f"Final loss: {losses[-1]:.6f}")
        print(f"Final gradient norm: {gradient_norms[-1]:.6f}")
    
    return results

# 6. Learning Rate Scheduling
def learning_rate_scheduling_demo():
    """Demo การใช้ learning rate scheduling"""
    
    # สร้างข้อมูลและ model
    X = torch.randn(200, 8)
    y = torch.randint(0, 3, (200,))
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32)
    
    model = nn.Sequential(
        nn.Linear(8, 16),
        nn.ReLU(),
        nn.Linear(16, 3)
    )
    
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # ทดสอบ schedulers ต่างๆ
    schedulers = [
        ("StepLR", optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)),
        ("ExponentialLR", optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)),
        ("CosineAnnealingLR", optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10))
    ]
    
    for scheduler_name, scheduler in schedulers:
        print(f"\n--- Testing {scheduler_name} ---")
        
        # Reset optimizer learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.01
            
        lr_history = []
        loss_history = []
        
        for epoch in range(15):
            # Training
            model.train()
            epoch_loss = 0
            
            for data, target in loader:
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            # Update learning rate
            current_lr = optimizer.param_groups[0]['lr']
            lr_history.append(current_lr)
            loss_history.append(epoch_loss / len(loader))
            
            if scheduler_name != "No Scheduler":
                scheduler.step()
            
            if epoch % 3 == 0:
                print(f"Epoch {epoch}: LR = {current_lr:.6f}, Loss = {loss_history[-1]:.4f}")
        
        print(f"Final LR: {lr_history[-1]:.6f}")
        print(f"Final Loss: {loss_history[-1]:.4f}")

# ทดสอบการใช้งาน
print("\n1. Manual vs Automatic Backpropagation")
print("-" * 45)

# สร้างข้อมูลทดสอบ
torch.manual_seed(42)
x_test = torch.randn(5, 3)
y_test = torch.randn(5, 2)

# สร้าง model และทดสอบ
manual_model = ManualBackpropNet(input_size=3, hidden_size=4, output_size=2)
compare_gradients(manual_model, x_test, y_test)

print("\n2. Gradient Clipping Demo")
print("-" * 25)

# สร้างข้อมูลสำหรับ gradient clipping demo
X_clip = torch.randn(160, 6)
y_clip = torch.randint(0, 2, (160,))
clip_dataset = TensorDataset(X_clip, y_clip)
clip_loader = DataLoader(clip_dataset, batch_size=32)

clip_model = GradientClippingDemo(input_size=6, hidden_size=20, output_size=2)
optimizer = optim.SGD(clip_model.parameters(), lr=0.1)  # High LR to create large gradients
criterion = nn.CrossEntropyLoss()

print("Training with gradient clipping (max_norm=1.0):")
tracker = training_with_gradient_clipping(
    clip_model, clip_loader, optimizer, criterion, max_norm=1.0, num_epochs=3
)

print("\n3. Detailed Optimizer Comparison")
print("-" * 35)

optimizer_results = detailed_optimizer_comparison()

# แสดงสรุปผล
print("\nOptimizer Comparison Summary:")
print("-" * 30)
for opt_name, results in optimizer_results.items():
    print(f"{opt_name:15} | Final Loss: {results['final_loss']:.6f}")

print("\n4. Learning Rate Scheduling")
print("-" * 30)

learning_rate_scheduling_demo()

print("\n5. Gradient Analysis")
print("-" * 20)

# วิเคราะห์ gradients แบบละเอียด
analysis_model = nn.Sequential(
    nn.Linear(4, 8),
    nn.ReLU(),
    nn.Linear(8, 4),
    nn.ReLU(),
    nn.Linear(4, 1)
)

X_analysis = torch.randn(10, 4)
y_analysis = torch.randn(10, 1)

optimizer = optim.Adam(analysis_model.parameters(), lr=0.001)
criterion = nn.MSELoss()

print("Gradient analysis for 5 training steps:")
for step in range(5):
    optimizer.zero_grad()
    output = analysis_model(X_analysis)
    loss = criterion(output, y_analysis)
    loss.backward()
    
    print(f"\nStep {step + 1}:")
    print(f"  Loss: {loss.item():.6f}")
    
    # วิเคราะห์ gradients แต่ละ layer
    for name, param in analysis_model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_mean = param.grad.mean().item()
            grad_std = param.grad.std().item()
            
            print(f"  {name}:")
            print(f"    Gradient norm: {grad_norm:.6f}")
            print(f"    Gradient mean: {grad_mean:.6f}")
            print(f"    Gradient std:  {grad_std:.6f}")
    
    optimizer.step()

print("\n" + "=" * 60)
print("สรุป Example 2: Backward Propagation และ Gradient Updates")
print("=" * 60)
print("✓ Manual backpropagation implementation")
print("✓ เปรียบเทียบ automatic vs manual gradients")
print("✓ Gradient tracking และ visualization")
print("✓ Gradient clipping techniques")
print("✓ เปรียบเทียบ optimizers แบบละเอียด")
print("✓ Learning rate scheduling")
print("✓ การวิเคราะห์ gradients อย่างละเอียด")
print("=" * 60)