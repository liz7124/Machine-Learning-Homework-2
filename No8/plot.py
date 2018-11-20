import matplotlib.pyplot as plt

class Plot:
    def plot_accuracy_and_loss(self, t_title_list, t_result_list):
            f, ax = plt.subplots(2, 2, figsize = (14, 6))
            f.canvas.set_window_title('Result')
            colors = ['b', 'y', 'r', 'c', 'm', 'g']

            ax[0, 0].set_title('Training accuracy')
            ax[0, 1].set_title('Validation accuracy')
            ax[1, 0].set_title('Training loss')
            ax[1, 1].set_title('Validation loss')

            for i in range(len(t_title_list)):
                hist = t_result_list[i].history
                acc = hist['acc']
                val_acc = hist['val_acc']
                loss = hist['loss']
                val_loss = hist['val_loss']
                epochs = range(len(acc))

                ax[0, 0].plot(epochs, acc, colors[i], label = t_title_list[i])
                ax[0, 1].plot(epochs, val_acc, colors[i], label = t_title_list[i])
                ax[1, 0].plot(epochs, loss, colors[i], label = t_title_list[i])
                ax[1, 1].plot(epochs, val_loss, colors[i], label = t_title_list[i])

            ax[0, 0].legend()
            ax[0, 1].legend()
            ax[1, 0].legend()
            ax[1, 1].legend()

            plt.show()