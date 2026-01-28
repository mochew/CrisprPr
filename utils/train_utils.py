import numpy as np
import torch

def remove_on_target(train_data, inter_Val_data, cross_Val_data):
    matches = [1, 7, 13, 19, 25, 31]
    print("remove on target")
    train_On_Target_mask = np.array([any(x not in matches for x in arr[:-3]) for arr in train_data[0]])

    train_Pos_list = train_data[0][train_On_Target_mask]
    train_labels = train_data[1][train_On_Target_mask]

    inter_Val_On_Target_mask = np.array([any(x not in matches for x in arr[:-3]) for arr in inter_Val_data[0]])
    inter_Val_Pos_list = inter_Val_data[0][inter_Val_On_Target_mask]
    inter_Val_labels = inter_Val_data[1][inter_Val_On_Target_mask]
    inter_Sg_list = inter_Val_data[2][inter_Val_On_Target_mask]

    cross_Val_On_Target_mask = np.array([any(x not in matches for x in arr[:-3]) for arr in cross_Val_data[0]])
    cross_Val_Pos_list = cross_Val_data[0][cross_Val_On_Target_mask]
    cross_Val_labels = cross_Val_data[1][cross_Val_On_Target_mask]
    cross_Sg_list = cross_Val_data[2][cross_Val_On_Target_mask]

    return [train_Pos_list, train_labels], [inter_Val_Pos_list, inter_Val_labels, inter_Sg_list],\
           [cross_Val_Pos_list, cross_Val_labels, cross_Sg_list]



class EarlyStopping_prc:
    def __init__(self, patience=5, verbose=False, delta=0, path=""):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path 
        self.counter = 0
        self.best_prc = 0
        self.early_stop = False

    def __call__(self, val_prc, model, epoch):
        if epoch > 40:
            if val_prc > self.best_prc - self.delta:
                if self.verbose:
                    print(f'Validation loss decreased ({self.best_prc:.4f} --> {val_prc:.4f}). Saving model ...')
                self.best_prc = val_prc
                self.counter = 0
                save_model = torch.jit.script(model)
                save_model.save(self.path)
            else:
                self.counter += 1
                if self.verbose:
                    print(f'Validation loss did not improve. Counter: {self.counter}/{self.patience}')
                
                if self.counter >= self.patience:
                    self.early_stop = True
                    if self.verbose:
                        print("Early stopping triggered")

    def load_best_model(self, model):
        if self.best_model_wts is not None:
            model.load_state_dict(self.best_model_wts)
