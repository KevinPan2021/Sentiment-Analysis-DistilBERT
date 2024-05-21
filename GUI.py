application_name = 'Sentiment Analysis'
# pyqt packages
from PyQt5.QtWidgets import QMainWindow, QApplication

import pickle
import sys
import torch
from transformers import DistilBertTokenizerFast

from qt_main import Ui_Application
from main import BidirectionalMap, compute_device, inference
from distilBERT import DistilBERT



class QT_Action(Ui_Application, QMainWindow):
    def __init__(self):
        # system variable
        super(QT_Action, self).__init__()
        self.setupUi(self)
        self.retranslateUi(self)
        self.setWindowTitle(application_name) # set the title
        
        # runtime variable
        self.model = None
        self.tokenizer = None
        with open('label_ind_map.pkl', 'rb') as f:
            self.label_ind_map = pickle.load(f)
            
        # load the model
        self.load_model_action()
        

            
    # linking all button/textbox with actions    
    def link_commands(self,):
        self.comboBox_model.activated.connect(self.load_model_action)
        self.toolButton_process.clicked.connect(self.process_action)
        
                
            
    # choosing between models
    def load_model_action(self,):
        self.model_name = self.comboBox_model.currentText()
        
        # load the model
        if self.model_name == 'DistilBERT':
            label2id = self.label_ind_map.key_to_value
            id2label = self.label_ind_map.value_to_key
            self.model = DistilBERT(id2label, label2id).model
            self.tokenizer = DistilBertTokenizerFast.from_pretrained('dslim/distilbert-NER')
        
        self.model.load_state_dict(torch.load(f'{type(self.model).__name__}_finetuned.pth'))
            
        # move model to GPU
        self.model = self.model.to(compute_device())
        
    
        
    def process_action(self):
        # get the input sentence
        data = self.textEdit_text.toPlainText()

        tok = self.tokenizer(data)
        
        # model inference
        output = inference(self.model, torch.tensor(tok['input_ids']))
        
        # print out the output sentence
        self.lineEdit_response.setText(output)
        
        
def main():
    app = QApplication(sys.argv)
    action = QT_Action()
    action.link_commands()
    action.show()
    sys.exit(app.exec_())
    
    
if __name__ == '__main__':
    main()