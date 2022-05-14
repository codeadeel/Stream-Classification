#!/usr/bin/env python3

"""
STREAM CLASSIFICATION TRAINER
=============================

The following program is used to train model on updated or new data
"""

# %%
# Importing Libraries
from model import *
import argparse
import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, jaccard_score
import seaborn as sns

# %%
# Main Data Loading Class
class Data(torch.utils.data.Dataset):
    def __init__(self, addr, label, transforms):
        """
        This method is used to initialize data loading class

        Method Input
        =============
        addr : List containing absolute address of files to include as data
        label : List containing respective Ohe Hot Encoded labels for images list
        transforms : Subject transforms to apply on the data
        
        Method Output
        ==============
        None
        """
        self.label = label
        self.__files__ = addr
        self.__transforms__ = transforms
    
    def __len__(self):
        """
        This method is used to find the number of files in a images directory

        Method Input
        =============
        None

        Method Output
        ==============
        None
        """
        return len(self.__files__)
    
    def __getitem__(self, idx):
        """
        This method is used to load and process the image based on file number

        Method Input
        =============
        idx : File number ( 0 - self.__len__())

        Method Output
        ==============
        Processed Image Data, Respective Image One Hot Encoded Label
        """
        self.data = Image.open(self.__files__[idx])
        self.data = self.__transforms__(self.data)
        return self.data, torch.Tensor([self.label[idx]]).type(torch.int32)

# %%
# Main Trainer Class
class Trainer:
    def __init__(self, addr, OHE, mod_addr, percentage=[70, 15, 15], epochs = 5, learn_rate = 0.001, batch_size=32, train_shuffle=True, seed=42):
        """
        This method is used to initialize model trainer

        Method Input
        =============
        addr : Absolute address of the parent directory of images sub-directories
        OHE : Absolute address to save one hot encoded labels file
        mod_addr : Absolute address to save model file
        percentage : List of splitting percentage of the data into Training, Testing & Validation
                            FORMAT : [ Training Data, Validation Data, Testing Data]
        epochs : Number of epochs to which model should be trained upto
        learn_rate : Learning rate for model training ( default : 0.001 )
        batch_size : Batch size of input data ( default : 32 )
        train_shuffle : Boolean valiable to shuffle training data ( default : True )
        seed : Seed value for the random split ( default : 42 )

        Method Output
        ==============
        None
        """
        torch.cuda.empty_cache()
        self.dataset_address = addr
        self.ohe_address = OHE
        self.model_address = mod_addr
        self.percentage = percentage
        self.epochs = epochs
        self.learning_rate = learn_rate
        self.batch_size = batch_size
        self.training_shuffle = train_shuffle
        self.seed = seed
        self.classes = os.listdir(self.dataset_address)
        self.current_ohes = dict()
        self.distribution = dict()
        self.__addr_labels__ = {'addrs': list(), 'labels': list()}
        self.__device__ = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.grad_scaler = torch.cuda.amp.GradScaler()
        for cla in range(len(self.classes)):
            self.current_ohes[self.classes[cla]] = cla
            temp_addrs = os.listdir(f'{self.dataset_address}/{self.classes[cla]}')
            self.distribution[self.classes[cla]] = len(temp_addrs)
            self.__addr_labels__['addrs'].extend([f'{self.dataset_address}/{self.classes[cla]}/{i}' for i in temp_addrs])
            self.__addr_labels__['labels'].extend([self.current_ohes[self.classes[cla]]] * len(temp_addrs))
        self.__history__ = {'training' : {'epoch' : list(),'batch' : list(),'loss' : list(), 'accuracy' : list()}, 'validation' : {'epoch' : list(),'batch' : list(),'loss' : list(), 'accuracy' : list()}, 'testing' : {'batch' : list(),'loss' : list(), 'accuracy' : list()}}
        self.__epoch_history__ = {'training' : {'epoch' : list(),'loss' : list(), 'accuracy' : list()}, 'validation' : {'epoch' : list(),'loss' : list(), 'accuracy' : list()}, 'testing' : {'loss' : list(), 'accuracy' : list()}}
    
    def __str__(self):
        """
        This method is __str__ implementation of subject class

        Method Input
        =============
        None

        Method Output
        ==============
        New Line
        """
        print("""
        ========================================
        | Stream Classification Model Training |
        ========================================
        """)
        print(f'Acceleration Device: {self.__device__}')
        print(f'Training Epochs: {self.epochs}')
        print(f'Learning Rate: {self.learning_rate}')
        print(f'Batch Size: {self.batch_size}')
        print(f'Training Data Shuffling: {self.training_shuffle}')
        print(f'Seed Value: {self.seed}')
        print(f'Data Splitting Percentage: {self.percentage}')
        print(f'Available Classes: {self.classes}')
        print(f'One Hot Encoded Labels: {self.current_ohes}')
        print(f'Data Distribution: {self.distribution}')
        print(f'Dataset Address: {self.dataset_address}')
        print(f'One Hot Encoded Labels Address: {self.ohe_address}')
        print(f'Model Address: {self.model_address}')
        print('\n---------------------------------------------')
        return '\n'

    def __get_percentage_values__(self, data_len, percentage):
        """
        This method is used to divide percentage into respective ranges

        Method Input
        =============
        data_len : Total dataset length to divide into respective percentages
        percentage : List of splitting percentage of the data into Training, Testing & Validation
                            FORMAT : [ Training Data, Validation Data, Testing Data]

        Method Output
        ==============
        New Line
        """
        per_values = [(data_len * i)//100 for i in percentage]
        if ((percentage[1] != 0) and (percentage[2] == 0)):
            per_values[1] = data_len - per_values[0]
        elif ((percentage[1] == 0) and (percentage[2] != 0)):
            per_values[2] = data_len - per_values[0]
        else:
            per_values[2] = data_len - per_values[0] - per_values[1]
        return per_values
    
    def __training_logs__(self):
        """
        This method is used to record training logs

        Method Input
        =============
        None

        Method Output
        ==============
        New Line
        """
        for ep, bs, tl, ta in zip(self.__history__['training']['epoch'], self.__history__['training']['batch'], self.__history__['training']['loss'], self.__history__['training']['accuracy']):
            self.hist_dat += f'{ep},{bs},{tl},{ta}\n'
        self.hist_dat += '\n\nEpoch Based Training History\nEpoch,Training Loss,Training Accuracy\n'
        for ep, tl, ta in zip(self.__epoch_history__['training']['epoch'], self.__epoch_history__['training']['loss'], self.__epoch_history__['training']['accuracy']):
            self.hist_dat += f'{ep},{tl},{ta}\n'
        self.hist_dat += '\n\n'
        if len(self.valid_data) != 0:
            self.hist_dat += 'Validation History\nEpoch,Batch,Validation Loss,Validation Accuracy\n'
            for ep, bs, vl, va in zip(self.__history__['validation']['epoch'], self.__history__['validation']['batch'], self.__history__['validation']['loss'], self.__history__['validation']['accuracy']):
                self.hist_dat += f'{ep},{bs},{vl},{va}\n'
            self.hist_dat += '\n\nEpoch Based Validation History\nEpoch,Validation Loss,Validation Accuracy\n'
            for ep, vl, va in zip(self.__epoch_history__['validation']['epoch'], self.__epoch_history__['validation']['loss'], self.__epoch_history__['validation']['accuracy']):
                self.hist_dat += f'{ep},{vl},{va}\n'
            self.hist_dat += '\n\n'
        if len(self.test_data) != 0:
            self.___test_epoch__()
            self.hist_dat += 'Testing History\nBatch,Testing Loss,Testing Accuracy\n'
            for bs, tl, ta in zip(self.__history__['testing']['batch'], self.__history__['testing']['loss'], self.__history__['testing']['accuracy']):
                self.hist_dat += f'{bs},{tl},{ta}\n'.format(bs, tl, ta)
            self.hist_dat += '\n\nOverall Testing Results\nLoss,{}\nAccuracy,{}'.format(self.__history__['testing']['loss'][-1], self.__epoch_history__['testing']['accuracy'][-1])
    
    def __accuracy__(self, out, target):
        """
        This method is used to find accuracy of the ongoing iterations

        Method Input
        =============
        out : Output data from model
        target : Target for the training

        Method Output
        ==============
        Accuracy of a batch
        """
        accu = torch.argmax(out, axis=1) == target
        return (sum(accu) / len(accu)).item()
    
    def __data_process__(self):
        """
        This method is used process the data for model training

        Method Input
        =============
        None

        Method Output
        ==============
        None
        """
        generator = np.random.default_rng(self.seed)
        dat_addr, dat_labs = np.array(self.__addr_labels__['addrs']), np.array(self.__addr_labels__['labels'])
        ranger = np.arange(len(dat_addr))
        generator.shuffle(ranger)
        dat_addr, dat_labs = dat_addr[ranger].tolist(), dat_labs[ranger].tolist()
        per_values = self.__get_percentage_values__(len(ranger), self.percentage)
        dd1, dl1 = dat_addr[:per_values[0]], dat_labs[:per_values[0]]
        dd2, dl2 = dat_addr[per_values[0] : per_values[0] + per_values[1]], dat_labs[per_values[0] : per_values[0] + per_values[1]]
        dd3, dl3 = dat_addr[per_values[0] + per_values[1] :], dat_labs[per_values[0] + per_values[1] :]
        self.train_data, self.valid_data, self.test_data = Data(dd1, dl1, training_transforms), Data(dd2, dl2, inference_transforms), Data(dd3, dl3, inference_transforms)
        self.train_batches, self.valid_batches, self.test_batches = len(self.train_data) // self.batch_size, len(self.valid_data) // self.batch_size, len(self.test_data) // self.batch_size
        self.training_data_loader = torch.utils.data.DataLoader(self.train_data, batch_size = self.batch_size, shuffle = self.training_shuffle, drop_last = True)
        if len(self.valid_data) != 0:
            self.validation_data_loader = torch.utils.data.DataLoader(self.valid_data, batch_size = self.batch_size, shuffle = False, drop_last = True)
        if len(self.test_data) != 0:
            self.testing_data_loader = torch.utils.data.DataLoader(self.test_data, batch_size = self.batch_size, shuffle = False, drop_last = True)
        self.hist_dat = 'Training History' + (',' * 50000) + '\nEpoch,Batch,Training Loss,Training Accuracy\n'
    
    def __confusion__(self):
        """
        This method is used to find confusion matrix for training

        Method Input
        =============
        None

        Method Output
        ==============
        None
        """
        torch.cuda.empty_cache()
        combined_data = Data(self.__addr_labels__['addrs'], self.__addr_labels__['labels'], inference_transforms)
        actual, predicted = list(), list()
        reverse_class_ohe = {values:keys for keys, values in self.current_ohes.items()}
        ranger = len(combined_data) // self.batch_size
        cdl = iter(torch.utils.data.DataLoader(combined_data, batch_size = self.batch_size, shuffle = False, drop_last = True))
        with tqdm.tqdm(total = ranger, bar_format = '{l_bar}{bar:10}{r_bar}{bar:-10b}', position = 0, leave = True) as bar:
            for i in range(ranger):
                try:
                    dat, labs = next(cdl)
                    labs = labs.squeeze().type(torch.LongTensor).tolist()
                    out = self.mod(dat.to(self.__device__))
                    out = torch.argmax(out, axis=1).tolist()
                    for l, o in zip(labs, out):
                        actual.append(reverse_class_ohe[l])
                        predicted.append(reverse_class_ohe[o])
                except StopIteration:
                    None
                bar.set_description('Calculating Confusion Matrix | Batch Size: {:<5} | Batch'.format(self.batch_size))
                bar.update(1)
        confus = confusion_matrix(actual, predicted)
        plt.figure('Confusion Matrix', figsize = ([13.1, 7.1]))
        dfr = pd.DataFrame(confus, list(reverse_class_ohe.values()), list(reverse_class_ohe.values()))
        plotter = sns.heatmap(dfr, cmap='viridis', annot=True, fmt='d', cbar=False).get_figure()
        plt.savefig(f'{self.model_address}_Confusion_Matrix.png')
        pres, reca = precision_score(actual, predicted, average = None), recall_score(actual, predicted, average = None)
        f1, jacc = f1_score(actual, predicted, average = None), jaccard_score(actual, predicted, average = None)
        self.hist_dat += '\n\nMetrices Scores,'+','.join(list(reverse_class_ohe.values())) + ',,Average Scores'
        self.hist_dat += '\nPrecision,{},,{}'.format(','.join([str(i) for i in pres]), sum(pres)/len(reverse_class_ohe))
        self.hist_dat += '\nRecall,{},,{}'.format(','.join([str(i) for i in reca]), sum(reca)/len(reverse_class_ohe))
        self.hist_dat += '\nF1 Score,{},,{}'.format(','.join([str(i) for i in f1]), sum(f1)/len(reverse_class_ohe))
        self.hist_dat += '\nJaccard Score,{},,{}\n'.format(','.join([str(i) for i in jacc]), sum(jacc)/len(reverse_class_ohe))
    
    def ___train_epoch__(self, current_epoch, tr_bar):
        """
        This method is used to perform one training epoch

        Method Input
        =============
        current_epoch : Number of current epoch
        tr_bar : Training bar for visualization

        Method Output
        ==============
        Description to be printed on progress bar
        """
        tr_bar.reset()
        self.mod.train()
        self.training_data_loader_iter = iter(self.training_data_loader)
        for tb in range(self.train_batches):
            try:
                self.optimizer.zero_grad()
                dat, labs = next(self.training_data_loader_iter)
                labs = labs.squeeze().type(torch.LongTensor).to(self.__device__)
                with torch.cuda.amp.autocast():
                    out = self.mod(dat.to(self.__device__))
                    loss1 = self.loss(out, labs)
                    self.__history__['training']['epoch'].append(current_epoch + 1)
                    self.__history__['training']['batch'].append(tb + 1)
                    self.__history__['training']['accuracy'].append(self.__accuracy__(out, labs))
                    self.__history__['training']['loss'].append(loss1.item())
                self.grad_scaler.scale(loss1).backward()
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
                tr_bar.set_description('Current Training Batch: [ L: {:.5f} | A: {:.5f} % ] | Batch'.format(self.__history__['training']['loss'][-1], self.__history__['training']['accuracy'][-1]))
                tr_bar.update(1)
            except (StopIteration, AttributeError):
                break
        self.__epoch_history__['training']['epoch'].append(current_epoch + 1)
        self.__epoch_history__['training']['loss'].append(sum(self.__history__['training']['loss'][-self.train_batches:]) / self.train_batches)
        self.__epoch_history__['training']['accuracy'].append(sum(self.__history__['training']['accuracy'][-self.train_batches:]) / self.train_batches)
        return 'Training: [ L: {:.5f} | A: {:.5f} % ]  '.format(self.__epoch_history__['training']['loss'][-1], self.__epoch_history__['training']['accuracy'][-1])
    
    def ___valid_epoch__(self, current_epoch, va_bar):
        """
        This method is used to perform one validation epoch

        Method Input
        =============
        current_epoch : Number of current epoch
        tr_bar : Training bar for visualization

        Method Output
        ==============
        Description to be printed on progress bar
        """
        va_bar.reset()
        self.mod.eval()
        self.validation_data_loader_iter = iter(self.validation_data_loader)
        for vb in range(self.valid_batches):
            try:
                dat, labs = next(self.validation_data_loader_iter)
                labs = labs.squeeze().type(torch.LongTensor).to(self.__device__)
                out = self.mod(dat.to(self.__device__))
                loss1 = self.loss(out, labs)
                self.__history__['validation']['epoch'].append(current_epoch + 1)
                self.__history__['validation']['batch'].append(vb + 1)
                self.__history__['validation']['accuracy'].append(self.__accuracy__(out, labs))
                self.__history__['validation']['loss'].append(loss1.item())
                va_bar.set_description('Current Validation Batch: [ L: {:.5f} | A: {:.5f} % ] | Batch'.format(self.__history__['validation']['loss'][-1], self.__history__['validation']['accuracy'][-1]))
                va_bar.update(1)
            except (StopIteration, AttributeError):
                break
        self.__epoch_history__['validation']['epoch'].append(current_epoch + 1)
        self.__epoch_history__['validation']['loss'].append(sum(self.__history__['validation']['loss'][-self.valid_batches:]) / self.valid_batches)
        self.__epoch_history__['validation']['accuracy'].append(sum(self.__history__['validation']['accuracy'][-self.valid_batches:]) / self.valid_batches)
        return 'Validation: [ L: {:.5f} | A: {:.5f} % ] '.format(self.__history__['validation']['loss'][-1], self.__epoch_history__['validation']['accuracy'][-1])
    
    def ___test_epoch__(self):
        """
        This method is used to calculate testing loss & accuracy

        Method Input
        =============
        None

        Method Output
        ==============
        None
        """
        self.testing_data_loader_iter = iter(self.testing_data_loader)
        with tqdm.tqdm(total = self.test_batches, bar_format = '{l_bar}{bar:10}{r_bar}{bar:-10b}', position = 0, leave = True) as te_bar:
            for teb in range(self.test_batches):
                try:
                    dat, labs = next(self.testing_data_loader_iter)
                    labs = labs.squeeze().type(torch.LongTensor).to(self.__device__)
                    out = self.mod(dat.to(self.__device__))
                    loss1 = self.loss(out, labs)
                    self.__history__['testing']['batch'].append(teb + 1)
                    self.__history__['testing']['accuracy'].append(self.__accuracy__(out, labs))
                    self.__history__['testing']['loss'].append(loss1.item())
                    te_bar.set_description('Current Testing Batch: [ L: {:.5f} | A: {:.5f} % ] | Batch'.format(self.__history__['testing']['loss'][-1], self.__history__['testing']['accuracy'][-1]))
                    te_bar.update(1)
                except (StopIteration, AttributeError):
                    break
            self.__epoch_history__['testing']['loss'].append(sum(self.__history__['testing']['loss'][-self.test_batches:]) / self.test_batches)
            self.__epoch_history__['testing']['accuracy'].append(sum(self.__history__['testing']['accuracy'][-self.test_batches:]) / self.test_batches)
            te_bar.set_description('Testing: [ L: {:.5f} | A: {:.5f} % ] '.format(self.__history__['testing']['loss'][-1], self.__epoch_history__['testing']['accuracy'][-1]))
    
    def __call__(self):
        """
        This method is used to train & save the model

        Method Input
        =============
        None

        Method Output
        ==============
        None
        """
        self.__data_process__()
        self.mod = Model(len(self.classes))
        self.mod.to(self.__device__)
        self.loss = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.mod.parameters(), lr =self.learning_rate, weight_decay=1e-4)
        with tqdm.tqdm(total = self.epochs, bar_format = '{l_bar}{bar:10}{r_bar}{bar:-10b}', position = 1) as bar:
            with tqdm.tqdm(total = self.train_batches, bar_format = '{l_bar}{bar:10}{r_bar}{bar:-10b}', position = 0, leave = True) as tr_bar:
                with tqdm.tqdm(total = self.valid_batches, bar_format = '{l_bar}{bar:10}{r_bar}{bar:-10b}', position = 0, leave = True) as va_bar:
                    for e in range(self.epochs):
                        desc = self.___train_epoch__(e, tr_bar)
                        torch.save(self.mod.state_dict(), self.model_address)
                        if len(self.valid_data) != 0:
                            desc += self.___valid_epoch__(e, va_bar)
                        bar.set_description(f'{desc}| Epoch')
                        bar.update(1)
        self.__training_logs__()
        self.__confusion__()
        with open(self.ohe_address, 'wb') as file1:
            pickle.dump(self.current_ohes, file1)
        with open(f'{self.model_address}_Training_Logs.csv', 'w') as file1:
            file1.write(self.hist_dat)
        print('\n---------------------------------------------\n')
        print(f'>>>>> One Hot Encoded Labels Saved at {self.ohe_address}')
        print(f'>>>>> Model Saved at {self.model_address}')
        print(f'>>>>> Training Logs Saved at {self.model_address}_Training_Logs.csv')
        print(f'>>>>> Confusion Matrix Saved at {self.model_address}_Confusion_Matrix.png')
        print('\n---------------------------------------------\n')

# %%
# Training Execution
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Stream Classification Model Trainer.')
    parser.add_argument('-e', '--epochs', type = int, help = 'Number of Epochs to Which Model Should be Trained Upto', required = True)
    parser.add_argument('-l', '--lr', type = float, help = 'Learning Rate for Model Training', default = 0.0001)
    parser.add_argument('-bs', '--batch_size', type = int, help = 'Batch Size of Input Data', default = 32)
    parser.add_argument('-trs', '--training_split', type = int, help = 'Training Split Percentage', default = 70)
    parser.add_argument('-vas', '--validation_split', type = int, help = 'Validation Split Percentage', default = 15)
    parser.add_argument('-tes', '--testing_split', type = int, help = 'Testing Split Percentage', default = 15)
    parser.add_argument('-sd', '--seed', type = int, help = 'Seed Value to Randomize Dataset', default = 42)
    parser.add_argument('-ts', '--train_shuffle', type = bool, help = 'Boolean Valiable to Shuffle Training Data ( True / False )', default = True)
    parser.add_argument('-d', '--data', type = str, help = 'Absolute Aaddress of the Parent Directory of Images Sub-Directories', default = '/data')
    parser.add_argument('-ohe', '--OHE', type = str, help = 'Absolute Address to Save One Hot Encoded Labels file', default = '/resources/OHE.labels')
    parser.add_argument('-ms', '--msaddr', type = str, help = 'Absolute Address to Save Model File', default = '/resources/convnext.model')
    args = vars(parser.parse_args())
    tra = Trainer(addr = args['data'], OHE = args['OHE'], mod_addr = args['msaddr'], percentage = [args['training_split'], args['validation_split'], args['testing_split']], epochs = args['epochs'], learn_rate = args['lr'], batch_size = args['batch_size'], train_shuffle = args['train_shuffle'], seed = args['seed'])
    print(tra)
    tra()
    