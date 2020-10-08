import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import torchvision
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import time
import os
import copy

import pickle
import json

from utils import AE_MNIST, AMoC

seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)

def train_model(model, dataloaders, criterion, optimizer, scheduler, input_size, device, num_epochs=25):
    since = time.time()

    train_loss_history = []
    val_loss_history = []
    dotp_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 999

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_dotp = 0.0

            # Iterate over data.
            for inputs, _ in dataloaders[phase]:
                inputs = inputs.view(-1,input_size).to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    outputs = model(inputs)
                    loss = criterion(outputs, inputs)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()                        

                # statistics
                running_loss += loss.item() * inputs.size(0)
                if phase == 'train' and optimizer.__class__.__name__ == 'AMoC':
                    running_dotp += optimizer.dotp * inputs.size(0)


            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            if phase == 'train':
                epoch_dotp = running_dotp / len(dataloaders[phase].dataset)
                dotp_history.append(epoch_dotp)
                print('{} Loss: {:.4f} DotP: {:.4f}'.format(phase, epoch_loss, epoch_dotp))
            else:
                print('{} Loss: {:.4f}'.format(phase, epoch_loss))

            # deep copy the model
            if phase == 'train':
                train_loss_history.append(epoch_loss)
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_loss_history.append(epoch_loss)

        scheduler.step()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, train_loss_history, val_loss_history, dotp_history, best_loss, time_elapsed


def test_model(model, dataloaders, criterion, input_size):
    running_loss = 0

    with torch.no_grad():
        for inputs, _ in dataloaders:
            input_test = inputs.view(-1, input_size).to(device)
            outputs = model_ft(input_test).to(device)
            
            loss = criterion(outputs, input_test)
            running_loss += loss.item() * inputs.size(0) 

    test_loss = running_loss / len(dataloaders.dataset)
    print('Test Loss: {:.4f}'.format(test_loss))

    return test_loss

if __name__ == "__main__":

    cuda_device = 0
    experiment_num = 1
    model_name = 'autoenc_mnist'
    dataset = 'mnist'
    input_size = 784

    num_epochs = 500
    batch_size = 200
    test_batch_size = 1000
    beta = 0.1
    lr = 5e-3
    momentum = 0.99
    dampening = 0
    weight_decay = 0
    algorithm = 'AMoC'
    gamma = 0.1
    milestones = [200, 400, 800]
    betas = [0.99, 0.999]
    epsilon = 1e-8

    data_transforms = {
        'train': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]),
        'test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    }

    device = torch.device("cuda:"+str(cuda_device) if torch.cuda.is_available() else "cpu")

    if dataset == 'mnist':
        mnist = torchvision.datasets.MNIST('./', train=True, transform=data_transforms['train'], target_transform=None, download=True)
        mnist_train, mnist_val = torch.utils.data.random_split(mnist, [54000, 6000])
        image_datasets = {'train': mnist_train, 'val': mnist_val}
        image_datasets_test = torchvision.datasets.MNIST('./', train=False, transform=data_transforms['test'], target_transform=None, download=True)

    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True) for x in ['train', 'val']}

    if model_name == 'autoenc_mnist':
        model = AE_MNIST(input_shape=input_size).to(device)

    if algorithm == 'AMoC':
        optimizer = AMoC(model.parameters(), lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay, nesterov=False, beta=beta)
    elif algorithm == 'AMoC-N':
        optimizer = AMoC(model.parameters(), lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay, nesterov=True, beta=beta)
    elif algorithm == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.0, dampening=dampening, weight_decay=weight_decay, nesterov=False)
    elif algorithm == 'Heavy Ball':    
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay, nesterov=False)
    elif algorithm == 'Nesterov':    
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay, nesterov=True)
    elif algorithm == 'Adam':    
        optimizer = optim.Adam(model.parameters(), lr=lr, betas=betas, eps=epsilon, weight_decay=weight_decay, amsgrad=False)
    elif algorithm == 'AMSGrad':
        optimizer = optim.Adam(model.parameters(), lr=lr, betas=betas, eps=epsilon, weight_decay=weight_decay, amsgrad=True)

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    criterion = nn.MSELoss()
    model_ft, train_hist, val_hist, dotp_hist, best_loss, time_elapsed = train_model(model, dataloaders_dict, criterion, optimizer, scheduler, input_size, device, num_epochs=num_epochs)

    #test
    dataloaders_test = torch.utils.data.DataLoader(image_datasets_test, batch_size=test_batch_size, shuffle=True, num_workers=0, pin_memory=True)
    test_loss = test_model(model_ft, dataloaders_test, criterion, input_size)

    #save
    path = model_name+'/' 
    fname = path+model_name+'_'+str(experiment_num)
    os.mkdir(fname)
    fname = fname+'/'+model_name+'_'+str(experiment_num)
    torch.save(model_ft.state_dict(), fname+".pt")
    with open(fname+".txt", "wb") as fp:
        pickle.dump(train_hist, fp)
        pickle.dump(val_hist, fp)
        pickle.dump(dotp_hist, fp)
    
    dictinoary = {}
    dictinoary['hyper-parameters'] = []
    dictinoary['hyper-parameters'].append({'num_epochs': num_epochs,
                                    'batch_size' : batch_size,
                                    'test_batch_size' : test_batch_size,
                                    'input_size' : input_size,
                                    'cuda_device' : cuda_device,
                                    'beta' : beta,
                                    'lr' : lr,
                                    'momentum' : momentum,
                                    'dampening' : dampening,
                                    'weight_decay' : weight_decay,
                                    'algorithm' : algorithm,
                                    'gamma' : gamma,
                                    'milestones' : milestones,
                                    'betas' : betas,
                                    'epsilon' : epsilon,
                                    'seed': seed,
                                    'stat_decay' : stat_decay,
                                    'damping' : damping,
                                    'kl_clip' : kl_clip,
                                    'TCov' : TCov,
                                    'TInv' : TInv,
                                    })
    dictinoary['values'] = []
    dictinoary['values'].append({'best_loss': best_loss,
                                 'time_elapsed': time_elapsed,
                                 'test_loss': test_loss
                                })
    with open(fname+"_dict"+".txt", "w") as fp:
        json.dump(dictinoary, fp)






