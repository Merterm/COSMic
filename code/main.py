"""
#########################################################
main.py

Description: Creates the NN for the metric and runs the training.

Dependencies:
    - model
    - dataloader
    - pytorch
    - matplotlib

Example run:
python main.py

Example run with pre-loaded model:
python main.py metric_nn_epoch_100

Author: Mert Inan
Date: 21 Jan 2021
#########################################################
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from dataloader import *
from model  import *
import sys
import time

'''
################################################################
############                PARAMETERS              ############
################################################################
'''
batch_size = 1
hidden_sizes = [512, 256, 128, 64, 32, 16, 8]
output_size = 1
numEpochs = 100
lr = 1e-2
print('GPU', torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''
################################################################
############                MAIN METHOD             ############
################################################################
'''
def main():
    # 1. Load the dataset from dataloader.py
    (train_dataloader, val_dataloader, test_dataloader, \
    input_cptn_size, input_label_size, input_visual_size) = load_data(batch_size)

    # 2. Create the model from model.py
    model, criterion, optimizer, lr_scheduler = create_net(input_cptn_size,
                                                            input_label_size,
                                                            input_visual_size,
                                                            hidden_sizes,
                                                            output_size, lr)

    # Load the pretrained model
    if len(sys.argv) > 1:
        dir = sys.argv[1]
        print('Loading the pretrained model...')
        model.load_state_dict(torch.load(dir))

    print(model)

    # 3. Run the epochs
    print('Starting the training...')
    model.to(device)
    model.train()

    for epoch in range(numEpochs):
        avg_loss = 0.0
        for batch_num, (c1, l1, c2, l2, v1, y) in enumerate(train_dataloader):
            # print((c1, l1, c2, l2, v1, y))
            c1, l1 = c1.to(device), l1.to(device)
            c2, l2 = c2.to(device), l2.to(device)
            v1 = v1.to(device)
            y = y.to(device)

            start = time.time()

            # Optimization steps
            optimizer.zero_grad()
            outputs = model(c1, l1, c2, l2, v1)
            # print(outputs.shape, y.shape)
            loss = criterion(outputs, y.reshape([-1,1]))
            loss.backward()

            optimizer.step()

            end = time.time()

            avg_loss += loss.item()

            #print('Learning rate', optimizer.lr)
            if batch_num % 10 == 9:
                print('Epoch: {}\tBatch: {}\tAvg-Loss: {:.4f}\tTime Elapsed: {}'.format(epoch+1,
                        batch_num+1, avg_loss, (end-start)))
                avg_loss = 0.0

            torch.cuda.empty_cache()
            del c1, l1, c2, l2, v1
            del y
            del loss

        #Save the current model
        print('Saving the model...')
        torch.save(model.state_dict(),('output/model/metric_nn_epoch_' + str(epoch)))

        # Step the LR Scheduler
        lr_scheduler.step()

        # # 4. Get your predictions for validation
        # print('Running on validation dataset...')
        # model.eval()
        # val_loss = []
        # accuracy = 0
        # total = 0
        # val_prediction = np.array([])
        # with torch.no_grad():
        #     for batch_num, (c1, l1, c2, l2, v1, y) in enumerate(val_dataloader):
        #         c1, l1 = c1.to(device), l1.to(device)
        #         c2, l2 = c2.to(device), l2.to(device)
        #         v1 = v1.to(device)
        #         y = y.to(device)
        #
        #         scores = model(c1, l1, c2, l2, v1)
        #
        #         loss = criterion(scores, y)
        #
        #         val_loss.extend([loss.item()]*feats.size()[0])
        #
        #         torch.cuda.empty_cache()
        #         del c1, l1, c2, l2, v1
        #         del y

        # 7. Get your predictions for test data
        print('Running on test dataset...')
        model.eval()
        scores = []
        with torch.no_grad():
            for batch_num, (c1, l1, c2, l2, v1, y) in enumerate(test_dataloader):
                c1, l1 = c1.to(device), l1.to(device)
                c2, l2 = c2.to(device), l2.to(device)
                v1 = v1.to(device)
                y = y.to(device)

                score = model(c1, l1, c2, l2, v1)
                scores.append(score.cpu().numpy())

                torch.cuda.empty_cache()
                del c1, l1, c2, l2, v1
                del y

        model.train()

        # 9. Output the file for your predictions
        with open('output/scores.txt', 'w') as f:
            for i in range(len(scores)):
                f.write(str(scores[i][0])+'\n')


        # # Plot validation statistics
        # plt.plot(range(len(val_accuracies)), val_accuracies)
        # plt.xlabel('Number of Epochs')
        # plt.title('Validation Accuracy')
        # plt.ylabel('Accuracy')
        # plt.savefig('val_accuracies.png')
        # plt.clf()

if __name__ == '__main__':
    main()
