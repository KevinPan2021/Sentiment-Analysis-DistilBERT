from transformers import DistilBertTokenizerFast, AutoModelForSequenceClassification
import torch.nn.functional as F
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
from multiprocessing import Pool

from fine_tuning import model_finetuning, feedforward

# supports both Mac mps and CUDA
def compute_device():
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'
    

def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()
    
    
def read_data(path):
    
    texts = []
    labels = []
    pool = Pool(processes=6)
    
    for label_class in ['pos', 'neg']:
        label_dir = os.path.join(path, label_class)
        file_paths = [os.path.join(label_dir, text_file) for text_file in os.listdir(label_dir)]
        
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
    
    
    
def inference(data, model, tokenizer, device):
    model.eval()
    batch = tokenizer(data, padding=True, truncation=True, max_length=512, return_tensors='pt')
    batch = batch.to(device)
    with torch.no_grad():
        outputs = model(**batch) 
        pred = F.softmax(outputs.logits, dim=1)
        labels = torch.argmax(pred, dim=1)
        labels = [model.config.id2label[label_id] for label_id in labels.tolist()]
    return labels
    



def main():
    # define a model and a tokenizer
    #model_name = 'distilbert-base-uncased-finetuned-sst-2-english'
    model_name = 'distilbert-base-uncased'
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model = model.to(compute_device())
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
    
    # load and process text data
    trainX, trainY = read_data('../Datasets/Stanford IMDB Movie Review/train')
    testX, testY = read_data('../Datasets/Stanford IMDB Movie Review/test')
    
    # visualize some examples
    for i in range(0, len(trainX), len(trainX)//5):
        print('>', trainX[i])
        print('=', trainY[i])
        print()
    
    # train-test split
    trainX, valX, trainY, valY = train_test_split(trainX, trainY, test_size=0.2)
    
    # tokenize
    train_encodings = tokenizer(trainX, truncation=True, padding=True)
    val_encodings = tokenizer(valX, truncation=True, padding=True)
    test_encodings = tokenizer(testX, truncation=True, padding=True)
    
    # dataset
    train_dataset = IMDB_Dataset(train_encodings, trainY)
    val_dataset = IMDB_Dataset(val_encodings, valY)
    test_dataset = IMDB_Dataset(test_encodings, testY)
    
    # dataloader
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    if not 'finetuned' in model_name:
        # fine tuning
        model_finetuning(model, train_loader, val_loader, compute_device())
    
    # load the best model
    model.load_state_dict(torch.load(f'{type(model).__name__}_finetuned.pth'))
    
    # get the test dataset metrics
    loss, acc = feedforward(model, test_loader, compute_device())
    print('test loss', loss, 'acc', acc)
    
    # visualize some examples
    for i in range(0, len(testX), len(testX)//5):
        output = inference(testX[i], model, tokenizer, compute_device())
        print('>', testX[i])
        print('=', testY[i])
        print('<', output)
        print()
        

if __name__ == "__main__":
    main()
    
    
    