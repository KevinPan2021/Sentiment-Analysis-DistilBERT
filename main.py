from transformers import DistilBertTokenizerFast
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
from multiprocessing import Pool
import pickle
from bs4 import BeautifulSoup
import logging

from distilBERT import DistilBERT
from fine_tuning import model_finetuning, feedforward


# Set the logging level to suppress warnings
logging.basicConfig(level=logging.ERROR)


# supports both Mac mps and CUDA
def compute_device():
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'
    

# bidirectional dictionary
class BidirectionalMap:
    def __init__(self):
        self.key_to_value = {}
        self.value_to_key = {}
    
    def __len__(self):
        return len(self.key_to_value)
    
    def add_mapping(self, key, value):
        self.key_to_value[key] = value
        self.value_to_key[value] = key

    def get_value(self, key):
        return self.key_to_value.get(key, 0)

    def get_key(self, value):
        return self.value_to_key.get(value)
    
    

def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text_data = f.read()
        # Parse HTML and extract text, remove html tags
        soup = BeautifulSoup(text_data, "html.parser")
        clean_text = soup.get_text()
        return clean_text
    
    
    
def read_data(path):
    texts = []
    labels = []
    pool = Pool(processes=6)
    
    for label_class in ['pos', 'neg']:
        label_dir = os.path.join(path, label_class)
        file_paths = [
            os.path.join(label_dir, text_file) for text_file in os.listdir(label_dir)
        ]
        
        # Use multiprocessing to read text files in parallel
        with tqdm(total=len(file_paths), desc=f"Reading {label_class} files") as pbar:
            for content in pool.imap_unordered(read_text_file, file_paths):
                texts.append(content)
                labels.append(0 if label_class.endswith('neg') else 1)
                pbar.update(1)
    
    pool.close()
    pool.join()
    
    return texts, labels



class IMDB_Dataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    
    def __len__(self):
        return len(self.labels)
    
    
    
@torch.no_grad() 
def inference(model, data):
    model.eval()
    device = next(model.parameters()).device
    
    input_ids = data.clone().unsqueeze(0).to(device)  # Add batch dimension
    attention_mask = torch.ones_like(input_ids).to(device) # placeholder
    outputs = model(input_ids, attention_mask=attention_mask) 
    
    labels = torch.argmax(outputs.logits, dim=-1).squeeze()
    labels = model.config.id2label[labels.item()]
    return labels



def main():
    # create a ind to label map
    label_ind_map = BidirectionalMap()
    label_ind_map.add_mapping("Negative", 0)
    label_ind_map.add_mapping("Positive", 1)
    label2id = label_ind_map.key_to_value
    id2label  = label_ind_map.value_to_key
    # Save the instance to a pickle file
    with open("label_ind_map.pkl", "wb") as f:
        pickle.dump(label_ind_map, f)
    
    # load model and tokenizer
    model = DistilBERT(id2label, label2id).model
    model = model.to(compute_device())
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    
    # load and process text data
    X, Y = read_data('../Datasets/Stanford IMDB Movie Review/train')
    testX, testY = read_data('../Datasets/Stanford IMDB Movie Review/test')
    
    # train-test split
    trainX, valX, trainY, valY = train_test_split(X, Y, test_size=0.2)
    del X, Y
    
    # tokenize
    trainX = tokenizer(trainX, truncation=True, padding=True)
    valX = tokenizer(valX, truncation=True, padding=True)
    testX = tokenizer(testX, truncation=True, padding=True)
    
    # dataset
    train_dataset = IMDB_Dataset(trainX, trainY)
    val_dataset = IMDB_Dataset(valX, valY)
    test_dataset = IMDB_Dataset(testX, testY)
    del trainX, trainY, valX, valY, testX, testY
    
    # dataloader
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    # visualize some examples
    for i in range(0, len(train_dataset), len(train_dataset)//5):
        item = train_dataset[i]
        x = item['input_ids']
        x_decoded = tokenizer.decode(x, skip_special_tokens=True)
        y = item['labels']
        label = label_ind_map.get_key(y.tolist())
        print('>', x_decoded)
        print('=', label)
        print()
    

    # fine tuning
    model_finetuning(model, train_loader, val_loader)
    
    # load the best model
    model.load_state_dict(torch.load(f'{type(model).__name__}_finetuned.pth'))
    
    # get the test dataset metrics
    feedforward(model, test_loader)
    for i in range(0, len(test_dataset), len(test_dataset)//5):
        item = test_dataset[i]
        x = item['input_ids']
        x_decoded = tokenizer.decode(x, skip_special_tokens=True)
        y = item['labels']
        label = label_ind_map.get_key(y.tolist())
        pred = inference(model, x)
        print('>', x_decoded)
        print('=', label)
        print('<', pred)
        print()
        

if __name__ == "__main__":
    main()