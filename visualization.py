import matplotlib.pyplot as plt


# plot the loss and acc curves
def plot_training_curves(train_acc, train_loss, valid_acc, valid_loss):
    plt.figure()
    plt.plot(train_loss, label='train')
    plt.plot(valid_loss, label='valid')
    plt.title('loss curves')
    plt.xlabel('epochs')
    plt.ylabel('cross entropy loss')
    plt.legend()
    
    plt.figure()
    plt.plot(train_acc, label='train')
    plt.plot(valid_acc, label='valid')
    plt.title('Accuracy curves')
    plt.xlabel('epochs')
    plt.ylabel('Accuracy score')
    plt.legend()
    
    
# print the sentences inline
def print_sentences(input_sentence, output_sentence, pred_sentence=None):
    print('>', input_sentence.strip())
    print('=', output_sentence.strip())
    if not pred_sentence is None:
        print('<', pred_sentence.strip())
    print()