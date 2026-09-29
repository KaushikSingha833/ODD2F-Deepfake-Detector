import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix

from model import ODD2F
from dataset import get_dataloaders

def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize model
    model = ODD2F(num_classes=2).to(device)
    
    # Load weights
    print(f"Loading weights from {args.weights}")
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()
    
    # Dataloaders (We only need the validation/test loader)
    # We pass the same dir to both since we only use the val_loader
    _, test_loader = get_dataloaders(args.test_dir, args.test_dir, batch_size=args.batch_size)
    
    criterion = nn.CrossEntropyLoss()
    
    test_loss = 0.0
    all_labels = []
    all_preds_probs = []
    all_preds_classes = []
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Evaluating")
        for rgb, ela, labels in pbar:
            rgb, ela, labels = rgb.to(device), ela.to(device), labels.to(device)
            
            outputs = model(rgb, ela)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item() * rgb.size(0)
            
            probs = torch.softmax(outputs, dim=1)[:, 1] # Probability of positive class
            _, predicted = torch.max(outputs.data, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds_probs.extend(probs.cpu().numpy())
            all_preds_classes.extend(predicted.cpu().numpy())
            
    avg_loss = test_loss / len(test_loader.dataset)
    acc = accuracy_score(all_labels, all_preds_classes)
    
    try:
        auc = roc_auc_score(all_labels, all_preds_probs)
    except ValueError:
        auc = 0.5
        
    print("\n--- Evaluation Results ---")
    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"AUC:       {auc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds_classes, target_names=test_loader.dataset.classes))
    
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds_classes))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ODD2F Deepfake Detection Model")
    parser.add_argument('--test_dir', type=str, required=True, help="Path to test data directory")
    parser.add_argument('--weights', type=str, required=True, help="Path to model weights (.pth)")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size")
    
    args = parser.parse_args()
    evaluate(args)
