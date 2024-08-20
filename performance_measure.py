import matplotlib.pyplot as plt

def loss_plot(train, val):
    epochs = range(1, len(train)+1)

    plt.plot(epochs, train, label='Train Loss')
    plt.plot(epochs, val,   label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.title('Train & Validation Loss')
    plt.xticks(epochs)
    plt.savefig('./loss_plot.png')



def accuracy_plot(train, val):
    epochs = range(1, len(train)+1)

    plt.plot(epochs, train, label='Train Acc')
    plt.plot(epochs, val,   label='Val Acc')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Train & Validation Accuracy')
    plt.legend()    
    plt.xticks(epochs)
    plt.savefig('./accuracy_plot.png')


def cfm():
    pass

if __name__=='__main__':
    train = [0.8, 0.5, 0.6, 0.6, 0.4]
    val   = [0.5, 0.4, 0.3, 0.2, 0.4]

    loss_plot(train, val)
    accuracy_plot(train, val)