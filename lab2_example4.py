"""
Lab 5 Example 4: Validation Loop Implementation
เรียนรู้การสร้าง validation loop ที่ครบถ้วน และเทคนิค validation ต่างๆ
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("Lab 5 Example 4: Validation Loop Implementation")
print("=" * 60)

# 1. Advanced Validation Framework
class ValidationFramework:
    """Framework สำหรับ validation ที่ครบถ้วน"""
    
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.validation_history = defaultdict(list)
        
    def validate_epoch(self, val_loader, criterion, return_details=False):
        """ทำ validation หนึ่ง epoch"""
        
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        all_outputs = []
        batch_losses = []
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass
                output = self.model(data)
                loss = criterion(output, target)
                
                # Collect results
                total_loss += loss.item()
                batch_losses.append(loss.item())
                
                predictions = torch.argmax(output, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_outputs.extend(output.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(val_loader)
        accuracy = np.mean(np.array(all_predictions) == np.array(all_targets)) * 100
        
        results = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'predictions': all_predictions,
            'targets': all_targets,
            'outputs': all_outputs,
            'batch_losses': batch_losses
        }
        
        if return_details:
            results.update(self._calculate_detailed_metrics(all_targets, all_predictions, all_outputs))
        
        return results
    
    def _calculate_detailed_metrics(self, targets, predictions, outputs):
        """คำนวณ metrics แบบละเอียด"""
        
        from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
        
        # Basic metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            targets, predictions, average='weighted', zero_division=0
        )
        
        # Per-class metrics
        per_class_precision, per_class_recall, per_class_f1, support = precision_recall_fscore_support(
            targets, predictions, average=None, zero_division=0
        )
        
        # Confidence metrics
        outputs_array = np.array(outputs)
        probabilities = np.exp(outputs_array) / np.sum(np.exp(outputs_array), axis=1, keepdims=True)
        max_probs = np.max(probabilities, axis=1)
        
        confidence_stats = {
            'mean_confidence': np.mean(max_probs),
            'std_confidence': np.std(max_probs),
            'min_confidence': np.min(max_probs),
            'max_confidence': np.max(max_probs)
        }
        
        return {
            'precision': precision * 100,
            'recall': recall * 100,
            'f1_score': f1 * 100,
            'per_class_precision': per_class_precision * 100,
            'per_class_recall': per_class_recall * 100,
            'per_class_f1': per_class_f1 * 100,
            'support': support,
            'confidence_stats': confidence_stats
        }
    
    def cross_validation(self, dataset, k_folds=5, criterion=nn.CrossEntropyLoss(), 
                        optimizer_fn=None, num_epochs=10):
        """K-fold cross validation"""
        
        print(f"Starting {k_folds}-fold cross validation...")
        
        fold_results = []
        dataset_size = len(dataset)
        fold_size = dataset_size // k_folds
        
        for fold in range(k_folds):
            print(f"\nFold {fold + 1}/{k_folds}")
            print("-" * 30)
            
            # Split data
            start_idx = fold * fold_size
            end_idx = start_idx + fold_size if fold < k_folds - 1 else dataset_size
            
            # Create validation set for current fold
            indices = list(range(dataset_size))
            val_indices = indices[start_idx:end_idx]
            train_indices = indices[:start_idx] + indices[end_idx:]
            
            train_subset = torch.utils.data.Subset(dataset, train_indices)
            val_subset = torch.utils.data.Subset(dataset, val_indices)
            
            train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=64, shuffle=False)
            
            # Reset model
            self._reset_model_weights()
            
            # Setup optimizer
            if optimizer_fn is None:
                optimizer = optim.Adam(self.model.parameters(), lr=0.001)
            else:
                optimizer = optimizer_fn(self.model.parameters())
            
            # Train for this fold
            fold_train_history = []
            fold_val_history = []
            
            for epoch in range(num_epochs):
                # Training
                train_loss, train_acc = self._train_epoch(train_loader, optimizer, criterion)
                fold_train_history.append({'loss': train_loss, 'accuracy': train_acc})
                
                # Validation
                val_results = self.validate_epoch(val_loader, criterion, return_details=True)
                fold_val_history.append(val_results)
                
                if epoch % 5 == 0:
                    print(f"  Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_results['loss']:.4f}, Val Acc={val_results['accuracy']:.2f}%")
            
            # Store fold results
            final_val_results = fold_val_history[-1]
            fold_results.append({
                'fold': fold + 1,
                'final_val_loss': final_val_results['loss'],
                'final_val_accuracy': final_val_results['accuracy'],
                'final_val_f1': final_val_results['f1_score'],
                'train_history': fold_train_history,
                'val_history': fold_val_history
            })
            
            print(f"  Fold {fold + 1} Final: Val Loss={final_val_results['loss']:.4f}, Val Acc={final_val_results['accuracy']:.2f}%")
        
        # Calculate cross-validation statistics
        cv_stats = self._calculate_cv_statistics(fold_results)
        
        return fold_results, cv_stats
    
    def _train_epoch(self, train_loader, optimizer, criterion):
        """ทำ training หนึ่ง epoch"""
        
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for data, target in train_loader:
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            predictions = torch.argmax(output, dim=1)
            correct += (predictions == target).sum().item()
            total += target.size(0)
        
        avg_loss = total_loss / len(train_loader)
        accuracy = (correct / total) * 100
        
        return avg_loss, accuracy
    
    def _reset_model_weights(self):
        """Reset model weights"""
        def reset_weights(m):
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                m.reset_parameters()
        
        self.model.apply(reset_weights)
    
    def _calculate_cv_statistics(self, fold_results):
        """คำนวณสถิติจาก cross-validation"""
        
        val_losses = [fold['final_val_loss'] for fold in fold_results]
        val_accuracies = [fold['final_val_accuracy'] for fold in fold_results]
        val_f1_scores = [fold['final_val_f1'] for fold in fold_results]
        
        stats = {
            'mean_val_loss': np.mean(val_losses),
            'std_val_loss': np.std(val_losses),
            'mean_val_accuracy': np.mean(val_accuracies),
            'std_val_accuracy': np.std(val_accuracies),
            'mean_val_f1': np.mean(val_f1_scores),
            'std_val_f1': np.std(val_f1_scores),
            'min_val_accuracy': np.min(val_accuracies),
            'max_val_accuracy': np.max(val_accuracies)
        }
        
        return stats

# 2. Hold-out Validation
class HoldoutValidator:
    """Hold-out validation implementation"""
    
    def __init__(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
    
    def split_dataset(self, dataset):
        """แบ่งข้อมูลเป็น train/val/test"""
        
        total_size = len(dataset)
        train_size = int(self.train_ratio * total_size)
        val_size = int(self.val_ratio * total_size)
        test_size = total_size - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        return train_dataset, val_dataset, test_dataset
    
    def validate_with_holdout(self, model, dataset, criterion, optimizer_fn, num_epochs=20):
        """ทำ holdout validation"""
        
        print("Starting holdout validation...")
        
        # Split dataset
        train_dataset, val_dataset, test_dataset = self.split_dataset(dataset)
        
        print(f"Dataset split - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        # Setup training
        optimizer = optimizer_fn(model.parameters())
        validator = ValidationFramework(model)
        
        # Training with validation
        train_history = []
        val_history = []
        
        best_val_loss = float('inf')
        best_model_state = None
        
        for epoch in range(num_epochs):
            # Training
            train_loss, train_acc = validator._train_epoch(train_loader, optimizer, criterion)
            train_history.append({'loss': train_loss, 'accuracy': train_acc})
            
            # Validation
            val_results = validator.validate_epoch(val_loader, criterion, return_details=True)
            val_history.append(val_results)
            
            # Save best model
            if val_results['loss'] < best_val_loss:
                best_val_loss = val_results['loss']
                best_model_state = model.state_dict().copy()
            
            if epoch % 5 == 0:
                print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_results['loss']:.4f}, Val Acc={val_results['accuracy']:.2f}%")
        
        # Load best model and test
        model.load_state_dict(best_model_state)
        test_results = validator.validate_epoch(test_loader, criterion, return_details=True)
        
        return {
            'train_history': train_history,
            'val_history': val_history,
            'test_results': test_results,
            'best_val_loss': best_val_loss
        }

# 3. Bootstrap Validation
class BootstrapValidator:
    """Bootstrap validation implementation"""
    
    def __init__(self, n_bootstrap=10, sample_ratio=0.8):
        self.n_bootstrap = n_bootstrap
        self.sample_ratio = sample_ratio
    
    def bootstrap_validate(self, model, dataset, criterion, optimizer_fn, num_epochs=15):
        """ทำ bootstrap validation"""
        
        print(f"Starting bootstrap validation with {self.n_bootstrap} iterations...")
        
        bootstrap_results = []
        
        for bootstrap_iter in range(self.n_bootstrap):
            print(f"\nBootstrap iteration {bootstrap_iter + 1}/{self.n_bootstrap}")
            
            # Create bootstrap sample
            dataset_size = len(dataset)
            sample_size = int(self.sample_ratio * dataset_size)
            
            # Sample with replacement
            indices = torch.randint(0, dataset_size, (sample_size,))
            bootstrap_dataset = torch.utils.data.Subset(dataset, indices)
            
            # Split bootstrap sample
            val_size = sample_size // 5  # 20% for validation
            train_size = sample_size - val_size
            
            train_subset, val_subset = random_split(
                bootstrap_dataset, [train_size, val_size],
                generator=torch.Generator().manual_seed(bootstrap_iter)
            )
            
            # Create data loaders
            train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=64, shuffle=False)
            
            # Reset model and train
            validator = ValidationFramework(model)
            validator._reset_model_weights()
            
            optimizer = optimizer_fn(model.parameters())
            
            # Short training
            for epoch in range(num_epochs):
                train_loss, train_acc = validator._train_epoch(train_loader, optimizer, criterion)
                if epoch == num_epochs - 1:  # Last epoch
                    val_results = validator.validate_epoch(val_loader, criterion, return_details=True)
            
            bootstrap_results.append({
                'iteration': bootstrap_iter + 1,
                'final_val_loss': val_results['loss'],
                'final_val_accuracy': val_results['accuracy'],
                'final_val_f1': val_results['f1_score']
            })
            
            print(f"  Final: Val Loss={val_results['loss']:.4f}, Val Acc={val_results['accuracy']:.2f}%")
        
        # Calculate bootstrap statistics
        bootstrap_stats = self._calculate_bootstrap_stats(bootstrap_results)
        
        return bootstrap_results, bootstrap_stats
    
    def _calculate_bootstrap_stats(self, results):
        """คำนวณสถิติจาก bootstrap"""
        
        val_losses = [r['final_val_loss'] for r in results]
        val_accuracies = [r['final_val_accuracy'] for r in results]
        val_f1_scores = [r['final_val_f1'] for r in results]
        
        def confidence_interval(values, confidence=0.95):
            sorted_values = np.sort(values)
            n = len(sorted_values)
            alpha = 1 - confidence
            lower_idx = int(alpha/2 * n)
            upper_idx = int((1 - alpha/2) * n)
            return sorted_values[lower_idx], sorted_values[upper_idx]
        
        stats = {
            'mean_val_loss': np.mean(val_losses),
            'std_val_loss': np.std(val_losses),
            'loss_ci': confidence_interval(val_losses),
            'mean_val_accuracy': np.mean(val_accuracies),
            'std_val_accuracy': np.std(val_accuracies),
            'accuracy_ci': confidence_interval(val_accuracies),
            'mean_val_f1': np.mean(val_f1_scores),
            'std_val_f1': np.std(val_f1_scores),
            'f1_ci': confidence_interval(val_f1_scores)
        }
        
        return stats

# 4. Model Selection Framework
class ModelSelector:
    """Framework สำหรับเลือก model ที่ดีที่สุด"""
    
    def __init__(self, models_config, validation_method='holdout'):
        self.models_config = models_config
        self.validation_method = validation_method
        self.results = {}
    
    def compare_models(self, dataset, criterion=nn.CrossEntropyLoss(), num_epochs=15):
        """เปรียบเทียบ models ต่างๆ"""
        
        print(f"Comparing {len(self.models_config)} models using {self.validation_method} validation...")
        print("=" * 60)
        
        for model_name, model_config in self.models_config.items():
            print(f"\nEvaluating model: {model_name}")
            print("-" * 40)
            
            model = model_config['model']
            optimizer_fn = model_config['optimizer_fn']
            
            if self.validation_method == 'holdout':
                validator = HoldoutValidator()
                results = validator.validate_with_holdout(
                    model, dataset, criterion, optimizer_fn, num_epochs
                )
                self.results[model_name] = {
                    'method': 'holdout',
                    'test_accuracy': results['test_results']['accuracy'],
                    'test_loss': results['test_results']['loss'],
                    'test_f1': results['test_results']['f1_score'],
                    'best_val_loss': results['best_val_loss']
                }
                
            elif self.validation_method == 'cross_validation':
                validator = ValidationFramework(model)
                fold_results, cv_stats = validator.cross_validation(
                    dataset, k_folds=5, criterion=criterion, 
                    optimizer_fn=optimizer_fn, num_epochs=num_epochs
                )
                self.results[model_name] = {
                    'method': 'cross_validation',
                    'mean_val_accuracy': cv_stats['mean_val_accuracy'],
                    'std_val_accuracy': cv_stats['std_val_accuracy'],
                    'mean_val_loss': cv_stats['mean_val_loss'],
                    'std_val_loss': cv_stats['std_val_loss']
                }
            
            elif self.validation_method == 'bootstrap':
                validator = BootstrapValidator(n_bootstrap=8)
                bootstrap_results, bootstrap_stats = validator.bootstrap_validate(
                    model, dataset, criterion, optimizer_fn, num_epochs
                )
                self.results[model_name] = {
                    'method': 'bootstrap',
                    'mean_val_accuracy': bootstrap_stats['mean_val_accuracy'],
                    'std_val_accuracy': bootstrap_stats['std_val_accuracy'],
                    'accuracy_ci': bootstrap_stats['accuracy_ci']
                }
        
        return self.results
    
    def print_comparison_summary(self):
        """แสดงสรุปการเปรียบเทียบ"""
        
        if not self.results:
            print("No results to display. Run compare_models() first.")
            return
        
        print("\n" + "=" * 60)
        print("MODEL COMPARISON SUMMARY")
        print("=" * 60)
        
        # Sort by performance metric
        if self.validation_method == 'holdout':
            sorted_results = sorted(
                self.results.items(), 
                key=lambda x: x[1]['test_accuracy'], 
                reverse=True
            )
            print(f"{'Model':<20} {'Test Acc':<12} {'Test Loss':<12} {'Test F1':<12}")
            print("-" * 60)
            for model_name, results in sorted_results:
                print(f"{model_name:<20} {results['test_accuracy']:<12.2f} {results['test_loss']:<12.4f} {results['test_f1']:<12.2f}")
                
        elif self.validation_method == 'cross_validation':
            sorted_results = sorted(
                self.results.items(), 
                key=lambda x: x[1]['mean_val_accuracy'], 
                reverse=True
            )
            print(f"{'Model':<20} {'Mean Val Acc':<15} {'Std Val Acc':<15} {'Mean Val Loss':<15}")
            print("-" * 70)
            for model_name, results in sorted_results:
                print(f"{model_name:<20} {results['mean_val_accuracy']:<15.2f} {results['std_val_accuracy']:<15.2f} {results['mean_val_loss']:<15.4f}")
                
        elif self.validation_method == 'bootstrap':
            sorted_results = sorted(
                self.results.items(), 
                key=lambda x: x[1]['mean_val_accuracy'], 
                reverse=True
            )
            print(f"{'Model':<20} {'Mean Acc':<12} {'Std Acc':<12} {'95% CI':<20}")
            print("-" * 70)
            for model_name, results in sorted_results:
                ci_str = f"[{results['accuracy_ci'][0]:.2f}, {results['accuracy_ci'][1]:.2f}]"
                print(f"{model_name:<20} {results['mean_val_accuracy']:<12.2f} {results['std_val_accuracy']:<12.2f} {ci_str:<20}")

# 5. สร้างข้อมูลและ models สำหรับทดสอบ
def create_sample_models():
    """สร้าง models ตัวอย่างสำหรับเปรียบเทียบ"""
    
    models_config = {
        'Simple_MLP': {
            'model': nn.Sequential(
                nn.Linear(20, 32),
                nn.ReLU(),
                nn.Linear(32, 4)
            ),
            'optimizer_fn': lambda params: optim.Adam(params, lr=0.001)
        },
        
        'Deep_MLP': {
            'model': nn.Sequential(
                nn.Linear(20, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 4)
            ),
            'optimizer_fn': lambda params: optim.Adam(params, lr=0.001, weight_decay=1e-4)
        },
        
        'BatchNorm_MLP': {
            'model': nn.Sequential(
                nn.Linear(20, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Linear(32, 4)
            ),
            'optimizer_fn': lambda params: optim.SGD(params, lr=0.01, momentum=0.9)
        }
    }
    
    return models_config

def create_complex_dataset(num_samples=1500, input_size=20, num_classes=4):
    """สร้างข้อมูลที่ซับซ้อนกว่าเดิม"""
    
    torch.manual_seed(42)
    
    # สร้าง features แต่ละ class
    samples_per_class = num_samples // num_classes
    X_list = []
    y_list = []
    
    for class_id in range(num_classes):
        # สร้าง pattern ที่แตกต่างกันแต่ละ class
        if class_id == 0:
            # Class 0: high values in first half
            class_X = torch.cat([
                torch.randn(samples_per_class, input_size // 2) + 2,
                torch.randn(samples_per_class, input_size // 2) - 1
            ], dim=1)
        elif class_id == 1:
            # Class 1: high values in second half
            class_X = torch.cat([
                torch.randn(samples_per_class, input_size // 2) - 1,
                torch.randn(samples_per_class, input_size // 2) + 2
            ], dim=1)
        elif class_id == 2:
            # Class 2: alternating pattern
            class_X = torch.randn(samples_per_class, input_size)
            class_X[:, ::2] += 1.5  # even indices
            class_X[:, 1::2] -= 1.5  # odd indices
        else:
            # Class 3: random but with specific variance
            class_X = torch.randn(samples_per_class, input_size) * 0.5
        
        X_list.append(class_X)
        y_list.append(torch.full((samples_per_class,), class_id, dtype=torch.long))
    
    X = torch.cat(X_list)
    y = torch.cat(y_list)
    
    # Shuffle
    perm = torch.randperm(num_samples)
    X = X[perm]
    y = y[perm]
    
    return X, y

# ทดสอบการใช้งาน
print("\n1. สร้างข้อมูลและ models")
print("-" * 30)

# สร้างข้อมูล
X, y = create_complex_dataset(num_samples=1500, input_size=20, num_classes=4)
dataset = TensorDataset(X, y)

print(f"Dataset: {X.shape}, Classes: {torch.unique(y).tolist()}")

# สร้าง models
models_config = create_sample_models()
print(f"Models to compare: {list(models_config.keys())}")

print("\n2. Holdout Validation")
print("-" * 25)

# ทดสอบ holdout validation
selector_holdout = ModelSelector(models_config, validation_method='holdout')
holdout_results = selector_holdout.compare_models(dataset, num_epochs=12)
selector_holdout.print_comparison_summary()

print("\n3. Cross Validation")
print("-" * 20)

# ทดสอบ cross validation (ใช้ epochs น้อยกว่าเพื่อความเร็ว)
selector_cv = ModelSelector(models_config, validation_method='cross_validation')
cv_results = selector_cv.compare_models(dataset, num_epochs=8)
selector_cv.print_comparison_summary()

print("\n4. Bootstrap Validation")
print("-" * 25)

# ทดสอบ bootstrap validation
selector_bootstrap = ModelSelector(models_config, validation_method='bootstrap')
bootstrap_results = selector_bootstrap.compare_models(dataset, num_epochs=10)
selector_bootstrap.print_comparison_summary()

print("\n5. Detailed Validation Analysis")
print("-" * 35)

# ทำ detailed validation สำหรับ model ที่ดีที่สุด
best_model_name = max(holdout_results.items(), key=lambda x: x[1]['test_accuracy'])[0]
best_model = models_config[best_model_name]['model']

print(f"Detailed analysis for best model: {best_model_name}")

# Reset และสร้าง detailed validation
validator = ValidationFramework(best_model)

# Create simple train/val split
dataset_size = len(dataset)
train_size = int(0.8 * dataset_size)
val_size = dataset_size - train_size

train_dataset, val_dataset = random_split(
    dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Train และ validate
validator._reset_model_weights()
optimizer = models_config[best_model_name]['optimizer_fn'](best_model.parameters())
criterion = nn.CrossEntropyLoss()

print(f"Training {best_model_name} for detailed analysis...")

for epoch in range(15):
    train_loss, train_acc = validator._train_epoch(train_loader, optimizer, criterion)
    
    if epoch % 5 == 4:  # Every 5 epochs
        val_results = validator.validate_epoch(val_loader, criterion, return_details=True)
        
        print(f"\nEpoch {epoch+1} Detailed Results:")
        print(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
        print(f"  Val: Loss={val_results['loss']:.4f}, Acc={val_results['accuracy']:.2f}%")
        print(f"  Val: Precision={val_results['precision']:.2f}%, Recall={val_results['recall']:.2f}%, F1={val_results['f1_score']:.2f}%")
        
        # Confidence statistics
        conf_stats = val_results['confidence_stats']
        print(f"  Confidence: Mean={conf_stats['mean_confidence']:.3f}, Std={conf_stats['std_confidence']:.3f}")

# Final detailed validation
final_val_results = validator.validate_epoch(val_loader, criterion, return_details=True)

print(f"\nFinal Detailed Validation Results:")
print(f"Overall Metrics:")
print(f"  Accuracy: {final_val_results['accuracy']:.2f}%")
print(f"  Precision: {final_val_results['precision']:.2f}%")
print(f"  Recall: {final_val_results['recall']:.2f}%")
print(f"  F1-Score: {final_val_results['f1_score']:.2f}%")

print(f"\nPer-Class Metrics:")
for i in range(len(final_val_results['per_class_precision'])):
    print(f"  Class {i}: Prec={final_val_results['per_class_precision'][i]:.1f}%, "
          f"Rec={final_val_results['per_class_recall'][i]:.1f}%, "
          f"F1={final_val_results['per_class_f1'][i]:.1f}%, "
          f"Support={final_val_results['support'][i]}")

print("\n" + "=" * 60)
print("สรุป Example 4: Validation Loop Implementation")
print("=" * 60)
print("✓ ValidationFramework ที่ครบถ้วน")
print("✓ K-fold Cross Validation")
print("✓ Holdout Validation")
print("✓ Bootstrap Validation")
print("✓ Model Selection Framework")
print("✓ Detailed metrics และ confidence analysis")
print("✓ การเปรียบเทียบ validation methods ต่างๆ")
print("=" * 60)