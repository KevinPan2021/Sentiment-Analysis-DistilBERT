application_name = 'Sentiment Analysis'
# pyqt packages
from PyQt5.QtWidgets import QMainWindow, QApplication

import sys
import torch
from transformers import DistilBertTokenizerFast, AutoModelForSequenceClassification

from qt_main import Ui_Application
from main import compute_device, inference



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
            model_name = 'distilbert-base-uncased'
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
        
        self.model.load_state_dict(torch.load(f'{type(self.model).__name__}_finetuned.pth'))
            
        # move model to GPU
        self.model = self.model.to(compute_device())
    
        
    
        
    def process_action(self):
        # get the input sentence
        question = self.textEdit_input.toPlainText()
        
        # model inference
        output = inference(question, self.model, self.tokenizer, compute_device())[0]
        
        if output.endswith('0'):
            output = 'negative'
        else:
            output = 'positive'
            
        # print out the output sentence
        self.textEdit_output.setPlainText(output)
        
        
def main():
    app = QApplication(sys.argv)
    action = QT_Action()
    action.link_commands()
    action.show()
    sys.exit(app.exec_())
    
    
if __name__ == '__main__':
    main()