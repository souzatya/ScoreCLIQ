import torch
import numpy as np
import pandas as pd
from torch import nn, optim
from models.BertRegressor import BertRegressor
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader, TensorDataset
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import mean_squared_error, mean_absolute_error

def encode_texts(texts, tokenizer, max_length=512):
    encoded = tokenizer(
        texts,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    return encoded['input_ids'], encoded['attention_mask']

def trainer(train, val, checkpoint_dir, model=None, tokenizer=None, device='cuda', batch_size=8, epochs=3, learning_rate=2e-5, weight_decay=0.01, lamda=0, l_reg=0, model_repo='bert-base-uncased', text_column='ItemStem', target_column='Difficulty', scheduler_warmup_steps=0):
    if model==None and tokenizer==None:
        # Load pre-trained BERT model and tokenizer
        tokenizer = BertTokenizer.from_pretrained(model_repo)
        bert_model = BertModel.from_pretrained(model_repo)

        # Create the regressor model
        model = BertRegressor(bert_model).to(device)
    
    # Preprocess Data for input to model
    train_texts = train[text_column].astype(str).tolist()
    train_targets = train[target_column].astype(float).values

    val_texts = val[text_column].astype(str).tolist()
    val_targets = val[target_column].astype(float).values

    input_ids_train, attention_mask_train = encode_texts(train_texts, tokenizer)
    input_ids_val, attention_mask_val = encode_texts(val_texts, tokenizer)

    train_targets = torch.FloatTensor(train_targets.copy())
    val_targets = torch.FloatTensor(val_targets.copy())

    train_dataset = TensorDataset(input_ids_train, attention_mask_train, train_targets)
    val_dataset = TensorDataset(input_ids_val, attention_mask_val, val_targets)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Optional scheduler
    total_steps = len(train_loader) * 3
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=scheduler_warmup_steps, num_training_steps=total_steps)

    for epoch in range(epochs):
        # --- Training ---
        model.train()
        total_train_loss = 0
        
        for batch in train_loader:
            input_ids, attention_mask, targets = [b.to(device) for b in batch]
            
            optimizer.zero_grad()
            predictions = model(input_ids, attention_mask)
            loss = criterion(predictions, targets) + lamda * l_reg  # Add regularization term
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # prevent exploding gradients
            optimizer.step()
            scheduler.step()
            
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        # --- Validation ---
        model.eval()
        total_val_loss = 0
        all_preds, all_targets = [], []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids, attention_mask, targets = [b.to(device) for b in batch]
                predictions = model(input_ids, attention_mask)
                
                loss = criterion(predictions, targets)
                total_val_loss += loss.item()
                
                all_preds.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        avg_val_loss = total_val_loss / len(val_loader)
        val_mse = mean_squared_error(all_targets, all_preds)
        val_mae = mean_absolute_error(all_targets, all_preds)
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss:   {avg_val_loss:.4f} | MSE: {val_mse:.4f} | MAE: {val_mae:.4f}")

        # Save model checkpoint and tokenizer
        torch.save(model.state_dict(), f'{checkpoint_dir}/checkpoint-{epoch+1}'.replace('.', '_') + '.pt')
    
    print("Training complete!")
    return model, tokenizer

def evaluate(test, model=None, tokenizer=None, device='cuda', batch_size=8, checkpoint='checkpoints/bert_regressor_2_454e-05_mse_0_0601.pt', model_repo='bert-base-uncased', text_column='ItemStem', target_column='Difficulty'):
    if model==None and tokenizer==None:
        # Load pre-trained BERT model and tokenizer
        tokenizer = BertTokenizer.from_pretrained(model_repo)
        bert_model = BertModel.from_pretrained(model_repo)

        # Load the model 
        model = BertRegressor(bert_model).to(device)

        # Load the checkpoint
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        model.to(device)
        print(f"Checkpoint loaded from {checkpoint}")
    elif model is not None and tokenizer==None:
        # Load pre-trained BERT model and tokenizer
        tokenizer = BertTokenizer.from_pretrained(model_repo)
        print(f"Model provided, loaded tokenizer from {model_repo}")

    # Evaluate the model on test set
    model.eval()
    test_texts = test[text_column].astype(str).tolist()
    test_targets = test[target_column].astype(float).values

    input_ids_test, attention_mask_test = encode_texts(test_texts, tokenizer)
    test_targets_tensor = torch.FloatTensor(test_targets.copy())

    test_dataset = TensorDataset(input_ids_test, attention_mask_test, test_targets_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    test_preds = []
    test_targets_list = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, targets = [b.to(device) for b in batch]
            predictions = model(input_ids, attention_mask)
            test_preds.extend(predictions.cpu().numpy())
            test_targets_list.extend(targets.cpu().numpy())

    test_mse = mean_squared_error(test_targets_list, test_preds)
    test_mae = mean_absolute_error(test_targets_list, test_preds)

    print(f"MSE: {test_mse:.4f}")
    print(f"MAE: {test_mae:.4f}")

    return test_texts, test_preds, test_targets_list, test_mse, test_mae    