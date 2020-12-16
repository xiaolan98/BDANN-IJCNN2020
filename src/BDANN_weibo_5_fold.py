import numpy as np
import argparse
import time, os
# import random
import process_data_weibo_5_fold as process_data
import copy
import pickle as pickle
from random import sample
import torchvision
import torch
import torch.nn as nn
from torch.autograd import Variable, Function
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn import metrics
from transformers import *


import warnings

warnings.filterwarnings("ignore")

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


class Rumor_Data(Dataset):
    def __init__(self, dataset):
        self.text = torch.from_numpy(np.array(dataset['post_text']))
        self.image = list(dataset['image'])
        # self.social_context = torch.from_numpy(np.array(dataset['social_feature']))
        self.mask = torch.from_numpy(np.array(dataset['mask']))
        self.label = torch.from_numpy(np.array(dataset['label']))
        self.event_label = torch.from_numpy(np.array(dataset['event_label']))
        print('TEXT: %d, Image: %d, label: %d, Event: %d'
              % (len(self.text), len(self.image), len(self.label), len(self.event_label)))

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return (self.text[idx], self.image[idx], self.mask[idx]), self.label[idx], self.event_label[idx]


class ReverseLayerF(Function):

    # @staticmethod
    def forward(self, x):
        self.lambd = args.lambd
        return x.view_as(x)

    # @staticmethod
    def backward(self, grad_output):
        return (grad_output * -self.lambd)


def grad_reverse(x):
    return ReverseLayerF()(x)


class CNN_Fusion_text(nn.Module):
    # def __init__(self, args, W):
    def __init__(self, args):
        super(CNN_Fusion_text, self).__init__()
        self.args = args

        self.event_num = args.event_num

        vocab_size = args.vocab_size
        emb_dim = args.embed_dim

        C = args.class_num
        self.hidden_size = args.hidden_dim
        self.lstm_size = args.embed_dim
        self.social_size = 19

        # # TEXT RNN
        #
        # self.embed = nn.Embedding(vocab_size, emb_dim)
        # # self.embed.weight = nn.Parameter(torch.from_numpy(W))
        # self.lstm = nn.LSTM(self.lstm_size, self.lstm_size)
        # self.text_fc = nn.Linear(self.lstm_size, self.hidden_size)
        # self.text_encoder = nn.Linear(emb_dim, self.hidden_size)
        #
        # ### TEXT CNN
        # channel_in = 1
        # filter_num = 20
        # window_size = [1, 2, 3, 4]
        # self.convs = nn.ModuleList([nn.Conv2d(channel_in, filter_num, (K, emb_dim)) for K in window_size])
        # self.fc1 = nn.Linear(len(window_size) * filter_num, self.hidden_size)

        # Bert
        bertModel = BertModel.from_pretrained('bert-base-chinese')
        self.bert_hidden_size = args.bert_hidden_dim
        self.fc2 = nn.Linear(self.bert_hidden_size, self.hidden_size)

        for param in bertModel.parameters():
            param.requires_grad = False
        self.bertModel = bertModel
        self.dropout = nn.Dropout(args.dropout)

        # IMAGE
        # hidden_size = args.hidden_dim
        vgg_19 = torchvision.models.vgg19(pretrained=True)
        for param in vgg_19.parameters():
            param.requires_grad = False
        # visual model
        num_ftrs = vgg_19.classifier._modules['6'].out_features
        self.vgg = vgg_19
        self.image_fc1 = nn.Linear(num_ftrs, self.hidden_size)
        # self.image_fc2 = nn.Linear(512, self.hidden_size)
        self.image_adv = nn.Linear(self.hidden_size, int(self.hidden_size))
        self.image_encoder = nn.Linear(self.hidden_size, self.hidden_size)

        ###social context
        self.social = nn.Linear(self.social_size, self.hidden_size)

        ##ATTENTION
        self.attention_layer = nn.Linear(self.hidden_size, emb_dim)

        ## Class  Classifier
        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(self.hidden_size, 2))
        # self.class_classifier.add_module('c_bn1', nn.BatchNorm2d(100))
        # self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        # self.class_classifier.add_module('c_drop1', nn.Dropout2d())
        # self.class_classifier.add_module('c_fc2', nn.Linear(self.hidden_size, 2))
        # self.class_classifier.add_module('c_bn2', nn.BatchNorm2d(self.hidden_size))
        # self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        # self.class_classifier.add_module('c_fc3', nn.Linear(100, 10))
        self.class_classifier.add_module('c_softmax', nn.Softmax(dim=1))

        ###Event Classifier
        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(self.hidden_size, self.hidden_size))
        # self.domain_classifier.add_module('d_bn1', nn.BatchNorm2d(self.hidden_size))
        self.domain_classifier.add_module('d_relu1', nn.LeakyReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(self.hidden_size, self.event_num))
        self.domain_classifier.add_module('d_softmax', nn.Softmax(dim=1))

        ####Image and Text Classifier
        self.modal_classifier = nn.Sequential()
        self.modal_classifier.add_module('m_fc1', nn.Linear(self.hidden_size, self.hidden_size))
        # self.domain_classifier.add_module('d_bn1', nn.BatchNorm2d(self.hidden_size))
        self.modal_classifier.add_module('m_relu1', nn.LeakyReLU(True))
        self.modal_classifier.add_module('m_fc2', nn.Linear(self.hidden_size, 2))
        self.modal_classifier.add_module('m_softmax', nn.Softmax(dim=1))

    def init_hidden(self, batch_size):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (to_var(torch.zeros(1, batch_size, self.lstm_size)),
                to_var(torch.zeros(1, batch_size, self.lstm_size)))

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (sample number,hidden_dim, length)
        # x = F.avg_pool1d(x, x.size(2)).squeeze(2)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)

        return x

    def forward(self, text, mask):

        #########CNN##################
        # text = self.embed(text)
        # text = text * mask.unsqueeze(2).expand_as(text)
        # text = text.unsqueeze(1)
        # text = [F.relu(conv(text)).squeeze(3) for conv in self.convs]  # [(N,hidden_dim,W), ...]*len(window_size)
        # #text = [F.avg_pool1d(i, i.size(2)).squeeze(2) for i in text]  # [(N,hidden_dim), ...]*len(window_size)
        # text = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in text]
        # text = torch.cat(text, 1)
        # text = F.relu(self.fc1(text))
        # #text = self.dropout(text)
        # bert
        last_hidden_state = torch.mean(self.bertModel(text)[0], dim=1, keepdim=False)
        text = self.fc2(last_hidden_state)

        ### Class
        # class_output = self.class_classifier(text_image)
        class_output = self.class_classifier(text)
        ## Domain
        reverse_feature = grad_reverse(text)
        domain_output = self.domain_classifier(reverse_feature)

        return class_output, domain_output


class CNN_Fusion_image(nn.Module):
    def __init__(self, args):
        super(CNN_Fusion_image, self).__init__()
        self.args = args

        self.event_num = args.event_num

        vocab_size = args.vocab_size
        emb_dim = args.embed_dim

        C = args.class_num
        self.hidden_size = args.hidden_dim
        self.lstm_size = args.embed_dim
        self.social_size = 19

        # # TEXT RNN
        # self.embed = nn.Embedding(vocab_size, emb_dim)
        # self.embed.weight = nn.Parameter(torch.from_numpy(W))
        # self.lstm = nn.LSTM(self.lstm_size, self.lstm_size)
        # self.text_fc = nn.Linear(self.lstm_size, self.hidden_size)
        # self.text_encoder = nn.Linear(emb_dim, self.hidden_size)
        #
        # ### TEXT CNN
        # channel_in = 1
        # filter_num = 20
        # window_size = [1, 2, 3, 4]
        # self.convs = nn.ModuleList([nn.Conv2d(channel_in, filter_num, (K, emb_dim)) for K in window_size])
        # self.fc1 = nn.Linear(len(window_size) * filter_num, self.hidden_size)

        # bert
        bert_model = BertModel.from_pretrained('bert-base-chinese')
        self.bert_hidden_size = args.bert_hidden_dim
        self.fc2 = nn.Linear(self.bert_hidden_size, self.hidden_size)

        for param in bert_model.parameters():
            param.requires_grad = False
        self.bertModel = bert_model

        self.dropout = nn.Dropout(args.dropout)

        #IMAGE
        #hidden_size = args.hidden_dim
        vgg_19 = torchvision.models.vgg19(pretrained=True)
        for param in vgg_19.parameters():
            param.requires_grad = False
        # visual model
        num_ftrs = vgg_19.classifier._modules['6'].out_features
        self.vgg = vgg_19
        # self.image_fc1 = nn.Linear(num_ftrs,  self.hidden_size)
        self.image_fc1 = nn.Linear(num_ftrs,  512)
        self.image_fc2 = nn.Linear(512, self.hidden_size)
        self.image_adv = nn.Linear(self.hidden_size,  int(self.hidden_size))
        self.image_encoder = nn.Linear(self.hidden_size, self.hidden_size)

        ###social context
        self.social = nn.Linear(self.social_size, self.hidden_size)

        ##ATTENTION
        self.attention_layer = nn.Linear(self.hidden_size, emb_dim)

        ## Class  Classifier
        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(2 * self.hidden_size, 2))
        #self.class_classifier.add_module('c_bn1', nn.BatchNorm2d(100))
        #self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        #self.class_classifier.add_module('c_drop1', nn.Dropout2d())
        #self.class_classifier.add_module('c_fc2', nn.Linear(self.hidden_size, 2))
        #self.class_classifier.add_module('c_bn2', nn.BatchNorm2d(self.hidden_size))
        #self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        #self.class_classifier.add_module('c_fc3', nn.Linear(100, 10))
        self.class_classifier.add_module('c_softmax', nn.Softmax(dim=1))

        ###Event Classifier
        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(2 * self.hidden_size, self.hidden_size))
        # self.domain_classifier.add_module('d_fc1', nn.Linear(self.hidden_size, self.hidden_size))
        #self.domain_classifier.add_module('d_bn1', nn.BatchNorm2d(self.hidden_size))
        self.domain_classifier.add_module('d_relu1', nn.LeakyReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(self.hidden_size, self.event_num))
        self.domain_classifier.add_module('d_softmax', nn.Softmax(dim=1))

    def init_hidden(self, batch_size):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (to_var(torch.zeros(1, batch_size, self.lstm_size)),
                to_var(torch.zeros(1, batch_size, self.lstm_size)))

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (sample number,hidden_dim, length)
        #x = F.avg_pool1d(x, x.size(2)).squeeze(2)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)

        return x

    def forward(self, text, image,  mask):
        ### IMAGE #####
        image = self.vgg(image) #[N, 512]
        # image = self.image_fc1(image)
        image = F.relu(self.image_fc1(image))
        image = F.relu(self.image_fc2(image))

        ##########CNN##################
        # text = self.embed(text)
        # text = text * mask.unsqueeze(2).expand_as(text)
        # text = text.unsqueeze(1)
        # text = [F.leaky_relu(conv(text)).squeeze(3) for conv in self.convs]  # [(N,hidden_dim,W), ...]*len(window_size)
        # #text = [F.avg_pool1d(i, i.size(2)).squeeze(2) for i in text]  # [(N,hidden_dim), ...]*len(window_size)
        # text = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in text]
        last_hidden_state = torch.mean(self.bertModel(text)[0], dim=1, keepdim=False)
        # text = self.fc2(last_hidden_state)
        text = F.relu(self.fc2(last_hidden_state))
        text_image = torch.cat((text, image), 1)

        ### Fake or real
        class_output = self.class_classifier(text_image)
        ## Domain (which Event )
        reverse_feature = grad_reverse(text_image)
        domain_output = self.domain_classifier(reverse_feature)

        # ### Multimodal
        # text_reverse_feature = grad_reverse(text)
        # image_reverse_feature = grad_reverse(image)
        # text_output = self.modal_classifier(text_reverse_feature)
        # image_output = self.modal_classifier(image_reverse_feature
        return class_output, domain_output


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def to_np(x):
    return x.data.cpu().numpy()


def clean_data(test_post_id, text_output, image_output, true_label, fold_idx):
    assert len(test_post_id) == len(text_output) == len(image_output) == len(true_label)
    deleted_id = []
    for idx in range(len(text_output)):
        if text_output[idx] == true_label[idx]:
            if image_output[idx] != true_label[idx]:
                deleted_id.append(test_post_id[idx])
    pickle.dump(deleted_id, open('../Data/weibo/deleted_post_fold_' + str(fold_idx) + '.pkl', 'wb'))


def main(args, fold_id):
    print('loading image data')
    train, validation, test = load_data(args, fold_id)

    train_dataset_image = Rumor_Data(train)

    validate_dataset_image = Rumor_Data(validation)

    test_dataset_image = Rumor_Data(test)

    # Data Loader (Input Pipeline)
    train_image_loader = DataLoader(dataset=train_dataset_image,
                              batch_size=args.batch_size,
                              shuffle=True)

    validate_image_loader = DataLoader(dataset=validate_dataset_image,
                                 batch_size=args.batch_size,
                                 shuffle=False)

    test_image_loader = DataLoader(dataset=test_dataset_image,
                             batch_size=args.batch_size,
                             shuffle=False)

    print('building model')
    model_image = CNN_Fusion_image(args)
    model_text = CNN_Fusion_text(args)

    if torch.cuda.is_available():
        print("CUDA")
        model_image.cuda()
        model_text.cuda()
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer_image = torch.optim.Adam(filter(lambda p: p.requires_grad, list(model_image.parameters())),
                                       lr=args.learning_rate)
    optimizer_text = torch.optim.Adam(filter(lambda p: p.requires_grad, list(model_text.parameters())),
                                      lr=args.learning_rate)
    print("loader size " + str(len(train_image_loader)))
    best_validate_acc_image = 0.000
    best_validate_acc_text = 0.000
    best_validate_dir_image = ''
    best_validate_dir_text = ''
    # best_validate_dir = '../Data/weibo/RESULT/53.pkl'

    print('training model')
    # Train the Model
    for epoch in range(args.num_epochs):

        p = float(epoch) / 100
        lr = 0.001 / (1. + 10 * p) ** 0.75

        optimizer_image.lr = lr
        optimizer_text.lr = lr
        cost_vector_image = []
        cost_vector_text = []
        class_cost_vector_image = []
        class_cost_vector_text = []
        domain_cost_vector_image = []
        domain_cost_vector_text = []
        acc_vector_image = []
        acc_vector_text = []
        valid_acc_vector_image = []
        valid_acc_vector_text = []
        vali_cost_vector_image = []
        vali_cost_vector_text = []

        for i, (train_data, train_labels, event_labels) in enumerate(train_image_loader):
            train_text, train_image, train_mask, train_labels, event_labels = \
                to_var(train_data[0]), to_var(train_data[1]), to_var(train_data[2]), \
                to_var(train_labels), to_var(event_labels)

            # Forward + Backward + Optimize
            optimizer_image.zero_grad()
            optimizer_text.zero_grad()

            class_outputs_image, domain_outputs_image = model_image(train_text, train_image, train_mask)
            class_outputs_text, domain_outputs_text = model_text(train_text, train_mask)

            # Fake or Real loss
            class_loss_image = criterion(class_outputs_image, train_labels)
            class_loss_text = criterion(class_outputs_text, train_labels)
            # Event Loss
            domain_loss_image = criterion(domain_outputs_image, event_labels)
            domain_loss_text = criterion(domain_outputs_text, event_labels)
            # loss = class_loss + domain_loss
            loss_image = class_loss_image - domain_loss_image
            loss_text = class_loss_text - domain_loss_text
            loss_image.backward()
            loss_text.backward()
            optimizer_image.step()
            optimizer_text.step()
            _, argmax_image = torch.max(class_outputs_image, 1)
            _, argmax_text = torch.max(class_outputs_text, 1)

            accuracy_image = (train_labels == argmax_image.squeeze()).float().mean()
            accuracy_text = (train_labels == argmax_text.squeeze()).float().mean()

            class_cost_vector_image.append(class_loss_image.item())
            class_cost_vector_text.append(class_loss_text.item())
            domain_cost_vector_image.append(domain_loss_image.item())
            domain_cost_vector_text.append(domain_loss_text.item())
            cost_vector_image.append(loss_image.item())
            cost_vector_text.append(loss_text.item())
            acc_vector_image.append(accuracy_image.item())
            acc_vector_text.append(accuracy_text.item())
            # if i == 0:
            #     train_score = to_np(class_outputs.squeeze())
            #     train_pred = to_np(argmax.squeeze())
            #     train_true = to_np(train_labels.squeeze())
            # else:
            #     class_score = np.concatenate((train_score, to_np(class_outputs.squeeze())), axis=0)
            #     train_pred = np.concatenate((train_pred, to_np(argmax.squeeze())), axis=0)
            #     train_true = np.concatenate((train_true, to_np(train_labels.squeeze())), axis=0)

        model_image.eval()
        model_text.eval()
        validate_acc_vector_temp_image = []
        validate_acc_vector_temp_text = []
        for i, (validate_data, validate_labels, event_labels) in enumerate(validate_image_loader):
            validate_text, validate_image, validate_mask, validate_labels, event_labels = \
                to_var(validate_data[0]), to_var(validate_data[1]), to_var(validate_data[2]), \
                to_var(validate_labels), to_var(event_labels)
            validate_outputs_image, domain_outputs_image = model_image(validate_text, validate_image, validate_mask)
            validate_outputs_text, domain_outputs_text = model_text(validate_text, validate_mask)
            _, validate_argmax_image = torch.max(validate_outputs_image, 1)
            _, validate_argmax_text = torch.max(validate_outputs_text, 1)
            vali_loss_image = criterion(validate_outputs_image, validate_labels)
            vali_loss_text = criterion(validate_outputs_text, validate_labels)
            # domain_loss = criterion(domain_outputs, event_labels)
            # _, labels = torch.max(validate_labels, 1)
            validate_accuracy_image = (validate_labels == validate_argmax_image.squeeze()).float().mean()
            validate_accuracy_text = (validate_labels == validate_argmax_text.squeeze()).float().mean()
            vali_cost_vector_image.append(vali_loss_image.item())
            vali_cost_vector_text.append(vali_loss_image.item())
            # validate_accuracy = (validate_labels == validate_argmax.squeeze()).float().mean()
            validate_acc_vector_temp_image.append(validate_accuracy_image.item())
            validate_acc_vector_temp_text.append(validate_accuracy_text.item())
        validate_acc_image = np.mean(validate_acc_vector_temp_image)
        validate_acc_text= np.mean(validate_acc_vector_temp_text)
        valid_acc_vector_image.append(validate_acc_image)
        valid_acc_vector_text.append(validate_acc_text)
        model_image.train()
        model_text.train()
        print('Image Epoch [%d/%d],  Loss: %.4f, Class Loss: %.4f, '
              'domain loss: %.4f, Train_Acc: %.4f,  Validate_Acc: %.4f.'
              % (
                  epoch + 1, args.num_epochs, np.mean(cost_vector_image), np.mean(class_cost_vector_image),
                  np.mean(domain_cost_vector_image),
                  np.mean(acc_vector_image), validate_acc_image))
        print('Text Epoch [%d/%d],  Loss: %.4f, Class Loss: %.4f, '
              'domain loss: %.4f, Train_Acc: %.4f,  Validate_Acc: %.4f.'
              % (
                  epoch + 1, args.num_epochs, np.mean(cost_vector_text), np.mean(class_cost_vector_text),
                  np.mean(domain_cost_vector_text),
                  np.mean(acc_vector_text), validate_acc_text))

        if validate_acc_image > best_validate_acc_image:
            best_validate_acc_image = validate_acc_image
            if not os.path.exists(args.output_file):
                os.mkdir(args.output_file)
            best_validate_dir_image = args.output_file + '_image_' + str(epoch + 1) + '.pkl'
            torch.save(model_image.state_dict(), best_validate_dir_image)
        if validate_acc_text > best_validate_acc_text:
            best_validate_acc_text = validate_acc_text
            if not os.path.exists(args.output_file):
                os.mkdir(args.output_file)
            best_validate_dir_text = args.output_file + '_text_' + str(epoch + 1) + '.pkl'
            torch.save(model_text.state_dict(), best_validate_dir_text)

    # Test the Model
    print('testing model')
    model_image = CNN_Fusion_image(args)
    model_text = CNN_Fusion_text(args)
    model_image.load_state_dict(torch.load(best_validate_dir_image))
    model_text.load_state_dict(torch.load(best_validate_dir_text))
    if torch.cuda.is_available():
        model_image.cuda()
        model_text.cuda()
    model_image.eval()
    model_text.eval()
    test_score_image = []
    test_pred_image = []
    test_true_image = []
    test_score_text = []
    test_pred_text = []
    test_true_text = []
    for i, (test_data, test_labels, event_labels) in enumerate(test_image_loader):
        test_text, test_image, test_mask, test_labels = to_var(
            test_data[0]), to_var(test_data[1]), to_var(test_data[2]), to_var(test_labels)
        test_outputs_image, domain_outputs_image = model_image(test_text, test_image, test_mask)
        test_outputs_text, domain_outputs_text = model_text(test_text, test_mask)
        _, test_argmax_image = torch.max(test_outputs_image, 1)
        _, test_argmax_text = torch.max(test_outputs_text, 1)
        if i == 0:
            test_score_image = to_np(test_outputs_image.squeeze())
            test_pred_image = to_np(test_argmax_image.squeeze())
            test_true_image = to_np(test_labels.squeeze())
            test_score_text = to_np(test_outputs_text.squeeze())
            test_pred_text = to_np(test_argmax_text.squeeze())
            test_true_text = to_np(test_labels.squeeze())
        else:
            test_score_image = np.concatenate((test_score_image, to_np(test_outputs_image.squeeze())), axis=0)
            test_pred_image = np.concatenate((test_pred_image, to_np(test_argmax_image.squeeze())), axis=0)
            test_true_image = np.concatenate((test_true_image, to_np(test_labels.squeeze())), axis=0)
            test_score_text = np.concatenate((test_score_text, to_np(test_outputs_text.squeeze())), axis=0)
            test_pred_text = np.concatenate((test_pred_text, to_np(test_argmax_text.squeeze())), axis=0)
            test_true_text = np.concatenate((test_true_text, to_np(test_labels.squeeze())), axis=0)

    test_accuracy_image = metrics.accuracy_score(test_true_image, test_pred_image)
    test_f1_image = metrics.f1_score(test_true_image, test_pred_image, average='macro')
    test_precision_image = metrics.precision_score(test_true_image, test_pred_image, average='macro')
    test_recall_image = metrics.recall_score(test_true_image, test_pred_image, average='macro')
    test_score_convert_image = [x[1] for x in test_score_image]
    test_aucroc_image = metrics.roc_auc_score(test_true_image, test_score_convert_image, average='macro')

    test_confusion_matrix_image = metrics.confusion_matrix(test_true_image, test_pred_image)
    clean_data(test['post_id'], test_pred_text, test_pred_image, test_true_image, fold_id)

    print("Image Classification Acc: %.4f, AUC-ROC: %.4f"
          % (test_accuracy_image, test_aucroc_image))
    print("Image Classification report:\n%s\n"
          % (metrics.classification_report(test_true_image, test_pred_image)))
    print("Image Classification confusion matrix:\n%s\n"
          % test_confusion_matrix_image)

    test_accuracy_text = metrics.accuracy_score(test_true_text, test_pred_text)
    test_f1_text = metrics.f1_score(test_true_text, test_pred_text, average='macro')
    test_precision_text = metrics.precision_score(test_true_text, test_pred_text, average='macro')
    test_recall_text = metrics.recall_score(test_true_text, test_pred_text, average='macro')
    test_score_convert_text = [x[1] for x in test_score_text]
    test_aucroc_text = metrics.roc_auc_score(test_true_text, test_score_convert_text, average='macro')

    test_confusion_matrix_text = metrics.confusion_matrix(test_true_text, test_pred_text)

    print("Text Classification Acc: %.4f, AUC-ROC: %.4f"
          % (test_accuracy_text, test_aucroc_text))
    print("Text Classification report:\n%s\n"
          % (metrics.classification_report(test_true_text, test_pred_text)))
    print("Text Classification confusion matrix:\n%s\n"
          % test_confusion_matrix_text)


def parse_arguments(parser):
    parser.add_argument('training_file', type=str, metavar='<training_file>', help='')
    # parser.add_argument('validation_file', type=str, metavar='<validation_file>', help='')
    parser.add_argument('testing_file', type=str, metavar='<testing_file>', help='')
    parser.add_argument('output_file', type=str, metavar='<output_file>', help='')

    parse.add_argument('--static', type=bool, default=True, help='')
    parser.add_argument('--sequence_length', type=int, default=28, help='')
    parser.add_argument('--class_num', type=int, default=2, help='')
    parser.add_argument('--hidden_dim', type=int, default=32, help='')
    parser.add_argument('--embed_dim', type=int, default=32, help='')
    parser.add_argument('--vocab_size', type=int, default=300, help='')
    parser.add_argument('--dropout', type=int, default=0.5, help='')
    parser.add_argument('--filter_num', type=int, default=5, help='')
    parser.add_argument('--lambd', type=int, default=1, help='')
    parser.add_argument('--text_only', type=bool, default=False, help='')

    #    parser.add_argument('--sequence_length', type = int, default = 28, help = '')
    #    parser.add_argument('--input_size', type = int, default = 28, help = '')
    #    parser.add_argument('--hidden_size', type = int, default = 128, help = '')
    #    parser.add_argument('--num_layers', type = int, default = 2, help = '')
    #    parser.add_argument('--num_classes', type = int, default = 10, help = '')
    parser.add_argument('--d_iter', type=int, default=3, help='')
    parser.add_argument('--batch_size', type=int, default=32, help='')
    parser.add_argument('--num_epochs', type=int, default=50, help='')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='')
    parser.add_argument('--event_num', type=int, default=10, help='')

    parser.add_argument('--bert_hidden_dim', type=int, default=768, help='')

    #    args = parser.parse_args()
    return parser


def get_top_post(output, label, test_id, top_n=500):
    filter_output = []
    filter_id = []
    # print(test_id)
    # print(output)
    for i, l in enumerate(label):
        # print(np.argmax(output[i]))
        if np.argmax(output[i]) == l and int(l) == 1:
            filter_output.append(output[i][1])
            filter_id.append(test_id[i])

    filter_output = np.array(filter_output)

    top_n_indice = filter_output.argsort()[-top_n:][::-1]

    top_n_id = np.array(filter_id)[top_n_indice]
    top_n_id_dict = {}
    for i in top_n_id:
        top_n_id_dict[i] = True

    pickle.dump(top_n_id_dict, open("../Data/weibo/top_n_id.pickle", "wb"))

    return top_n_id


def word2vec(post, word_id_map, W):
    word_embedding = []
    mask = []
    # length = []

    for sentence in post:
        sen_embedding = []
        seq_len = len(sentence) - 1
        mask_seq = np.zeros(args.sequence_len, dtype=np.float32)
        mask_seq[:len(sentence)] = 1.0
        for i, word in enumerate(sentence):
            sen_embedding.append(word_id_map[word])

        while len(sen_embedding) < args.sequence_len:
            sen_embedding.append(0)

        word_embedding.append(copy.deepcopy(sen_embedding))
        mask.append(copy.deepcopy(mask_seq))
        # length.append(seq_len)
    return word_embedding, mask


def re_tokenize_sentence(flag):
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    tokenized_texts = []
    original_texts = flag['original_post']
    for sentence in original_texts:
        tokenized_text = tokenizer.encode(sentence)
        tokenized_texts.append(tokenized_text)
    flag['post_text'] = tokenized_texts


def get_all_text(train, validate, test):
    all_text = list(train['post_text']) + list(validate['post_text']) + list(test['post_text'])
    return all_text


def align_data(flag, args):
    text = []
    mask = []
    for sentence in flag['post_text']:
        sen_embedding = []
        mask_seq = np.zeros(args.sequence_len, dtype=np.float32)
        mask_seq[:len(sentence)] = 1.0
        for i, word in enumerate(sentence):
            sen_embedding.append(word)

        while len(sen_embedding) < args.sequence_len:
            sen_embedding.append(0)

        text.append(copy.deepcopy(sen_embedding))
        mask.append(copy.deepcopy(mask_seq))
    flag['post_text'] = text
    flag['mask'] = mask


def get_event_num(flag):
    events = [int(event) for event in flag['event_label']]
    event_num = max(events)
    return event_num


def load_data(args, fold_id):
    train, validate, test, args.event_num = process_data.get_data(args.text_only, fold_id)
    re_tokenize_sentence(train)
    re_tokenize_sentence(validate)
    re_tokenize_sentence(test)
    all_text = get_all_text(train, validate, test)
    max_len = len(max(all_text, key=len))
    args.sequence_len = max_len
    align_data(train, args)
    align_data(validate, args)
    align_data(test, args)
    return train, validate, test


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parser = parse_arguments(parse)
    train = ''
    test = ''

    for fold_id in range(1, 5):
        output = '../Data/weibo/RESULT_' + str(fold_id+1) + '_fold/'
        args = parser.parse_args([train, test, output])
        main(args, fold_id+1)


