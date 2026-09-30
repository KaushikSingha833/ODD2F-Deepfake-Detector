import argparse
import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import json

from model import ODD2F
from dataset import get_dataloaders

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize model
    model = ODD2F(num_classes=2).to(device)
    
    # Dataloaders
    train_loader, val_loader = get_dataloaders(args.train_dir, args.val_dir, batch_size=args.batch_size)
    
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    start_epoch = 0
    start_batch = 0
    running_loss = 0.0
    correct = 0
    total = 0
    
    resume_path = os.path.join(args.save_dir, 'resume_checkpoint.pth')
    best_model_path = os.path.join(args.save_dir, 'best_model.pth')
    if os.path.exists(resume_path):
        print(f"Resuming from checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        start_batch = checkpoint['batch_idx'] + 1
        running_loss = checkpoint.get('running_loss', 0.0)
        correct = checkpoint.get('correct', 0)
        total = checkpoint.get('total', 0)
        print(f"Resumed at Epoch {start_epoch+1}, Batch {start_batch}")
    elif os.path.exists(best_model_path):
        print(f"Loading previous best model weights from {best_model_path} for continuous learning!")
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    best_val_auc = 0.0
    patience_counter = 0

    for epoch in range(start_epoch, args.epochs):
        model.train()
        if epoch > start_epoch or start_batch == 0:
            running_loss = 0.0
            correct = 0
            total = 0
            start_batch = 0
            
        # Initialize progress bar instantly before first batch loads
        progress_data = {
            "status": "training",
            "epoch": epoch + 1,
            "total_epochs": args.epochs,
            "batch": start_batch,
            "total_batches": len(train_loader),
            "loss": 0.0,
            "accuracy": 0.0
        }
        try:
            with open(os.path.join(os.path.dirname(__file__), 'ui', 'progress.json'), 'w') as f:
                json.dump(progress_data, f)
        except Exception:
            pass

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for batch_idx, (rgb, ela, labels) in enumerate(pbar):
            if batch_idx < start_batch:
                continue
                
            # --- PAUSE / CANCEL CHECK ---
            control_file = os.path.join(os.path.dirname(__file__), 'ui', 'control.json')
            if os.path.exists(control_file):
                try:
                    with open(control_file, 'r') as f:
                        control_data = json.load(f)
                    action = control_data.get("action", "resume")
                    
                    if action in ["pause", "cancel"]:
                        # Save mid-epoch state!
                        torch.save({
                            'epoch': epoch,
                            'batch_idx': batch_idx,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'running_loss': running_loss,
                            'correct': correct,
                            'total': total
                        }, resume_path)
                    
                    if action == "cancel":
                        print("\nTraining CANCELLED by user.")
                        with open(control_file, 'w') as f:
                            json.dump({"action": "resume"}, f)
                        sys.exit(0)
                        
                    while action == "pause":
                        pbar.set_description(f"Epoch {epoch+1}/{args.epochs} [PAUSED]")
                        time.sleep(1)
                        with open(control_file, 'r') as f:
                            control_data = json.load(f)
                        action = control_data.get("action", "resume")
                        if action == "cancel":
                            print("\nTraining CANCELLED by user.")
                            with open(control_file, 'w') as f:
                                json.dump({"action": "resume"}, f)
                            sys.exit(0)
                            
                    pbar.set_description(f"Epoch {epoch+1}/{args.epochs} [Train]")
                except Exception:
                    pass
            # ---------------------------
            
            rgb, ela, labels = rgb.to(device), ela.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(rgb, ela)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * rgb.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({'loss': loss.item(), 'acc': correct/total})
            
            # Save periodic resume checkpoint
            if batch_idx > 0 and batch_idx % 100 == 0:
                torch.save({
                    'epoch': epoch,
                    'batch_idx': batch_idx,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'running_loss': running_loss,
                    'correct': correct,
                    'total': total
                }, resume_path)
                
            # Write progress for UI every 5 batches
            current_batch = total // args.batch_size
            if current_batch % 5 == 0:
                progress_data = {
                    "status": "training",
                    "epoch": epoch + 1,
                    "total_epochs": args.epochs,
                    "batch": current_batch,
                    "total_batches": len(train_loader),
                    "loss": round(loss.item(), 4),
                    "accuracy": round(correct/total, 4)
                }
                progress_file = os.path.join(os.path.dirname(__file__), 'ui', 'progress.json')
                try:
                    with open(progress_file, 'w') as f:
                        json.dump(progress_data, f)
                except Exception:
                    pass
            
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_labels = []
        all_preds = []
        
        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]")
            for rgb, ela, labels in pbar_val:
                rgb, ela, labels = rgb.to(device), ela.to(device), labels.to(device)
                
                outputs = model(rgb, ela)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * rgb.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                probs = torch.softmax(outputs, dim=1)[:, 1] # Probability of positive class
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(probs.cpu().numpy())
                
        val_epoch_loss = val_loss / len(val_loader.dataset)
        val_epoch_acc = val_correct / val_total
        
        try:
            val_auc = roc_auc_score(all_labels, all_preds)
        except ValueError:
            val_auc = 0.5 # In case only one class is present in batch/val set
            
        # Update progress to show validation phase done
        progress_data = {
            "status": "validation_done",
            "epoch": epoch + 1,
            "total_epochs": args.epochs,
            "loss": round(val_epoch_loss, 4),
            "accuracy": round(val_epoch_acc, 4),
            "auc": round(val_auc, 4)
        }
        try:
            with open(os.path.join(os.path.dirname(__file__), 'ui', 'progress.json'), 'w') as f:
                json.dump(progress_data, f)
        except Exception:
            pass
            
        print(f"Epoch {epoch+1} Summary:")
        print(f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f}")
        print(f"Val Loss:   {val_epoch_loss:.4f} | Val Acc:   {val_epoch_acc:.4f} | Val AUC: {val_auc:.4f}")
        
        # Early Stopping and Checkpointing
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_model.pth'))
            print("Saved new best model!")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping triggered after {patience_counter} epochs without improvement.")
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ODD2F Deepfake Detection Model")
    parser.add_argument('--train_dir', type=str, required=True, help="Path to training data directory")
    parser.add_argument('--val_dir', type=str, required=True, help="Path to validation data directory")
    parser.add_argument('--save_dir', type=str, default="checkpoints", help="Directory to save model weights")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size")
    parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--weight_decay', type=float, default=1e-3, help="Weight decay for Adam (increased to stop overfitting)")
    parser.add_argument('--patience', type=int, default=10, help="Patience for early stopping")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        
    train(args)
