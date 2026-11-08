import torch

class Config:
    # --- Hardware & Device Setup ---
    DEVICE = torch.device('cpu') 
    
    # --- Model & Architecture ---
    D_MODEL = 128          
    NHEAD = 4              
    NUM_ENCODER_LAYERS = 2 
    NUM_DECODER_LAYERS = 2 
    DIM_FEEDFORWARD = 256  
    DROPOUT = 0.1
    
    # --- Vocab & Lengths ---
    VOCAB_SIZE = 256       
    MAX_LEN = 50
    
    # --- Training Hyperparameters ---
    BATCH_SIZE = 32        
    LEARNING_RATE = 3e-4   
    EPOCHS = 40            
    
    # --- Special Tokens ---
    PAD_IDX = 0
    SOS_IDX = 1
    EOS_IDX = 2
    UNK_IDX = 3