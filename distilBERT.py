from transformers import AutoModelForSequenceClassification

from summary import Summary
    
    
class DistilBERT():
    def __init__(self, id2label, label2id):
        super(DistilBERT, self).__init__()

        # define a model and a tokenizer
        model_name = 'distilbert-base-uncased'
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=len(id2label), id2label=id2label, label2id=label2id
        )
        
        # freeze first 5 layer weights        
        for name, param in self.model.named_parameters():
            if 'layer.5' not in name and 'classifier' not in name:
                param.requires_grad = False
        
    def forward(self, x):
        return self.model(x)
    

def main():    
    device = 'cuda'
    
    id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    label2id = {"NEGATIVE": 0, "POSITIVE": 1}

    # Creating model and testing output shapes 
    model = DistilBERT(id2label, label2id).model
    model.to(device)  # Move the underlying model to the desired device
    
    Summary(model, input_size=(64,))
    

if __name__ == "__main__": 
    main()
    
    