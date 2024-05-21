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
    plt.show()
    
    plt.figure()
    plt.plot(train_acc, label='train')
    plt.plot(valid_acc, label='valid')
    plt.ylim([0, 100])
    plt.title('Accuracy curves')
    plt.xlabel('epochs')
    plt.ylabel('Accuracy score')
    plt.legend()
    plt.show()
    