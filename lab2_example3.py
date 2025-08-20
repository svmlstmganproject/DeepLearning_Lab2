"""
Lab 5 Example 3: การ Monitor Loss และ Metrics
เรียนรู้การติดตามประสิทธิภาพของ model ระหว่าง training
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time

print("=" * 60)
print("Lab 5 Example 3: การ Monitor Loss และ Metrics")
print("=" * 60)

# 1. Comprehensive Metrics Tracker
class MetricsTracker:
    """Class สำหรับติดตาม metrics ต่างๆ ระหว่าง training"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset ค่า metrics ทั้งหมด"""
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        self.learning_rates = []
        self.epoch_times = []
        self.gradient_norms = []
        self.custom_metrics = {}
    
    def update_train_metrics(self, loss, accuracy, lr=None, grad_norm=None):
        """อัพเดท training metrics"""
        self.train_losses.append(loss)
        self.train_accuracies.append(accuracy)
        if lr is not None:
            self.learning_rates.append(lr)
        if grad_norm is not None:
            self.gradient_norms.append(grad_norm)
    
    def update_val_metrics(self, loss, accuracy):
        """อัพเดท validation metrics"""
        self.val_losses.append(loss)
        self.val_accuracies.append(accuracy)
    
    def add_custom_metric(self, name, value):
        """เพิ่ม custom metric"""
        if name not in self.custom_metrics:
            self.custom_metrics[name] = []
        self.custom_metrics[name].append(value)
    
    def add_epoch_time(self, time_taken):
        """เพิ่มเวลาที่ใช้ในแต่ละ epoch"""
        self.epoch_times.append(time_taken)
    
    def plot_metrics(self, figsize=(15, 10)):
        """วาดกราฟแสดง metrics ต่างๆ"""
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        
        # Loss plot
        epochs = range(1, len(self.train_losses) + 1)
        axes[0, 0].plot(epochs, self.train_losses, 'b-', label='Train Loss')
        if self.val_losses:
            axes[0, 0].plot(epochs, self.val_losses, 'r-', label='Val Loss')
        axes[0, 0].set_title('Loss Over Time')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy plot
        axes[0, 1].plot(epochs, self.train_accuracies, 'b-', label='Train Acc')
        if self.val_accuracies:
            axes[0, 1].plot(epochs, self.val_accuracies, 'r-', label='Val Acc')
        axes[0, 1].set_title('Accuracy Over Time')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Learning rate plot
        if self.learning_rates:
            axes[0, 2].plot(epochs, self.learning_rates, 'g-')
            axes[0, 2].set_title('Learning Rate')
            axes[0, 2].set_xlabel('Epoch')
            axes[0, 2].set_ylabel('Learning Rate')
            axes[0, 2].set_yscale('log')
            axes[0, 2].grid(True)
        
        # Gradient norms plot
        if self.gradient_norms:
            axes[1, 0].plot(epochs, self.gradient_norms, 'm-')
            axes[1, 0].set_title('Gradient Norms')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Gradient Norm')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True)
        
        # Epoch time plot
        if self.epoch_times:
            axes[1, 1].plot(epochs, self.epoch_times, 'c-')
            axes[1, 1].set_title('Training Time per Epoch')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Time (seconds)')
            axes[1, 1].grid(True)
        
        # Custom metrics plot
        if self.custom_metrics:
            metric_names = list(self.custom_metrics.keys())
            for i, metric_name in enumerate(metric_names[:4]):  # แสดงสูงสุด 4 metrics
                color = ['orange', 'purple', 'brown', 'pink'][i % 4]
                axes[1, 2].plot(epochs, self.custom_metrics[metric_name], 
                              color=color, label=metric_name)
            axes[1, 2].set_title('Custom Metrics')
            axes[1, 2].set_xlabel('Epoch')
            axes[1, 2].set_ylabel('Value')
            axes[1, 2].legend()
            axes[1, 2].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def get_summary(self):
        """สรุปผล metrics"""
        summary = {
            'best_train_loss': min(self.train_losses) if self.train_losses else None,
            'best_train_acc': max(self.train_accuracies) if self.train_accuracies else None,
            'best_val_loss': min(self.val_losses) if self.val_losses else None,
            'best_val_acc': max(self.val_accuracies) if self.val_accuracies else None,
            'total_training_time': sum(self.epoch_times) if self.epoch_times else None,
            'avg_epoch_time': np.mean(self.epoch_times) if self.epoch_times else None
        }
        return summary

# 2. Advanced Model with Detailed Metrics
class DetailedClassifier(nn.Module):
    """Classifier พร้อมการคำนวณ metrics แบบละเอียด"""
    
    def __init__(self, input_size, hidden_sizes, num_classes, dropout_rate=0.2):
        super(DetailedClassifier, self).__init__()
        
        layers = []
        current_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(current_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            current_size = hidden_size
        
        layers.append(nn.Linear(current_size, num_classes))
        
        self.network = nn.Sequential(*layers)
        self.num_classes = num_classes
    
    def forward(self, x):
        return self.network(x)
    
    def predict_proba(self, x):
        """คำนวณ probability"""
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=1)
        return probabilities
    
    def predict(self, x):
        """ทำนาย class"""
        with torch.no_grad():
            logits = self.forward(x)
            predictions = torch.argmax(logits, dim=1)
        return predictions

# 3. Comprehensive Evaluation Function
def evaluate_model(model, data_loader, criterion, device='cpu'):
    """ประเมิน model แบบครอบคลุม"""
    
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    all_probabilities = []
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            
            # เก็บ predictions และ targets
            predictions = torch.argmax(output, dim=1)
            probabilities = F.softmax(output, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # คำนวณ metrics
    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(all_targets, all_predictions) * 100
    
    # Multi-class metrics
    precision = precision_score(all_targets, all_predictions, average='weighted') * 100
    recall = recall_score(all_targets, all_predictions, average='weighted') * 100
    f1 = f1_score(all_targets, all_predictions, average='weighted') * 100
    
    metrics = {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'predictions': all_predictions,
        'targets': all_targets,
        'probabilities': all_probabilities
    }
    
    return metrics

# 4. Advanced Training Loop with Monitoring
def advanced_training_loop(model, train_loader, val_loader, optimizer, criterion, 
                         scheduler=None, num_epochs=20, device='cpu', early_stopping_patience=5):
    """Advanced training loop พร้อม monitoring แบบครบถ้วน"""
    
    model = model.to(device)
    tracker = MetricsTracker()
    
    # Early stopping variables
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    print("Starting advanced training...")
    print("=" * 50)
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
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
            
            # Calculate training accuracy
            predictions = torch.argmax(output, dim=1)
            train_correct += (predictions == target).sum().item()
            train_total += target.size(0)
            train_loss += loss.item()
        
        # Validation phase
        val_metrics = evaluate_model(model, val_loader, criterion, device)
        
        # Calculate averages
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = (train_correct / train_total) * 100
        
        # Update learning rate
        current_lr = optimizer.param_groups[0]['lr']
        if scheduler is not None:
            scheduler.step()
        
        # Record epoch time
        epoch_time = time.time() - epoch_start_time
        
        # Update tracker
        tracker.update_train_metrics(avg_train_loss, train_accuracy, current_lr, grad_norm)
        tracker.update_val_metrics(val_metrics['loss'], val_metrics['accuracy'])
        tracker.add_epoch_time(epoch_time)
        tracker.add_custom_metric('precision', val_metrics['precision'])
        tracker.add_custom_metric('recall', val_metrics['recall'])
        tracker.add_custom_metric('f1_score', val_metrics['f1_score'])
        
        # Print progress
        print(f"Epoch {epoch+1:3d}/{num_epochs}")
        print(f"  Train: Loss={avg_train_loss:.4f}, Acc={train_accuracy:.2f}%")
        print(f"  Val:   Loss={val_metrics['loss']:.4f}, Acc={val_metrics['accuracy']:.2f}%")
        print(f"  Val:   Prec={val_metrics['precision']:.2f}%, Rec={val_metrics['recall']:.2f}%, F1={val_metrics['f1_score']:.2f}%")
        print(f"  LR={current_lr:.6f}, GradNorm={grad_norm:.4f}, Time={epoch_time:.2f}s")
        print("-" * 50)
        
        # Early stopping check
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            
        if patience_counter >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            print(f"Best validation loss: {best_val_loss:.4f}")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return tracker, model

# 5. สร้างข้อมูลตัวอย่าง
def create_classification_dataset(num_samples=2000, input_size=20, num_classes=4):
    """สร้างข้อมูลสำหรับ classification"""
    
    # สร้าง features
    X = torch.randn(num_samples, input_size)
    
    # สร้าง labels ที่มี pattern
    weights = torch.randn(input_size, num_classes)
    logits = torch.matmul(X, weights)
    y = torch.argmax(logits, dim=1)
    
    # เพิ่ม noise
    noise_mask = torch.rand(num_samples) < 0.1  # 10% noise
    if noise_mask.sum() > 0:
        y[noise_mask] = torch.randint(0, num_classes, (noise_mask.sum(),))
    
    return X, y

# 6. Real-time Monitoring
class RealTimeMonitor:
    """Real-time monitoring ระหว่าง training"""
    
    def __init__(self, update_frequency=10):
        self.update_frequency = update_frequency
        self.step_count = 0
        self.running_loss = 0.0
        self.running_accuracy = 0.0
        
    def update(self, loss, accuracy):
        """อัพเดท running metrics"""
        self.step_count += 1
        self.running_loss += loss
        self.running_accuracy += accuracy
        
        if self.step_count % self.update_frequency == 0:
            avg_loss = self.running_loss / self.update_frequency
            avg_accuracy = self.running_accuracy / self.update_frequency
            
            print(f"  Step {self.step_count}: Running Loss={avg_loss:.4f}, Running Acc={avg_accuracy:.2f}%")
            
            # Reset
            self.running_loss = 0.0
            self.running_accuracy = 0.0

# ทดสอบการใช้งาน
print("\n1. สร้างข้อมูลและ model")
print("-" * 30)

# สร้างข้อมูล
X, y = create_classification_dataset(num_samples=2000, input_size=20, num_classes=4)

# แบ่งข้อมูล train/val
split_idx = int(0.8 * len(X))
X_train, X_val = X[:split_idx], X[split_idx:]
y_train, y_val = y[:split_idx], y[split_idx:]

print(f"Train set: {X_train.shape}, {y_train.shape}")
print(f"Val set: {X_val.shape}, {y_val.shape}")

# สร้าง DataLoaders
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# สร้าง model
model = DetailedClassifier(
    input_size=20, 
    hidden_sizes=[64, 32], 
    num_classes=4, 
    dropout_rate=0.3
)

print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

print("\n2. Advanced Training with Monitoring")
print("-" * 40)

# Setup optimizer และ scheduler
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3, verbose=True
)
criterion = nn.CrossEntropyLoss()

# Training
tracker, trained_model = advanced_training_loop(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    criterion=criterion,
    scheduler=scheduler,
    num_epochs=25,
    early_stopping_patience=7
)

print("\n3. Training Results Summary")
print("-" * 30)

summary = tracker.get_summary()
for key, value in summary.items():
    if value is not None:
        if 'time' in key:
            print(f"{key}: {value:.2f} seconds")
        elif 'acc' in key:
            print(f"{key}: {value:.2f}%")
        else:
            print(f"{key}: {value:.4f}")

print("\n4. Final Model Evaluation")
print("-" * 30)

# ประเมินผล final model
final_train_metrics = evaluate_model(trained_model, train_loader, criterion)
final_val_metrics = evaluate_model(trained_model, val_loader, criterion)

print("Final Training Metrics:")
for key, value in final_train_metrics.items():
    if key not in ['predictions', 'targets', 'probabilities']:
        if 'accuracy' in key or 'precision' in key or 'recall' in key or 'f1' in key:
            print(f"  {key}: {value:.2f}%")
        else:
            print(f"  {key}: {value:.4f}")

print("\nFinal Validation Metrics:")
for key, value in final_val_metrics.items():
    if key not in ['predictions', 'targets', 'probabilities']:
        if 'accuracy' in key or 'precision' in key or 'recall' in key or 'f1' in key:
            print(f"  {key}: {value:.2f}%")
        else:
            print(f"  {key}: {value:.4f}")

# วิเคราะห์ confusion matrix
from collections import Counter

print("\n5. Class Distribution Analysis")
print("-" * 35)

train_class_dist = Counter(final_train_metrics['targets'])
val_class_dist = Counter(final_val_metrics['targets'])

print("Training set class distribution:")
for class_id, count in sorted(train_class_dist.items()):
    percentage = (count / len(final_train_metrics['targets'])) * 100
    print(f"  Class {class_id}: {count} samples ({percentage:.1f}%)")

print("\nValidation set class distribution:")
for class_id, count in sorted(val_class_dist.items()):
    percentage = (count / len(final_val_metrics['targets'])) * 100
    print(f"  Class {class_id}: {count} samples ({percentage:.1f}%)")

print("\n" + "=" * 60)
print("สรุป Example 3: การ Monitor Loss และ Metrics")
print("=" * 60)
print("✓ MetricsTracker สำหรับติดตาม metrics ครบถ้วน")
print("✓ การประเมิน model แบบละเอียด")
print("✓ Advanced training loop พร้อม monitoring")
print("✓ Early stopping และ best model saving")
print("✓ Real-time monitoring")
print("✓ การวิเคราะห์ผลลัพธ์แบบครบถ้วน")
print("=" * 60)