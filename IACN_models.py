'''
This is a supporting library with the code of the model.
'''

from __future__ import division
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch import optim
import numpy as np
import math, random
import sys
from collections import defaultdict
import os
import sys
from tqdm import tqdm, trange, tqdm_notebook, tnrange
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

PATH = "./"

try:
    get_ipython
    trange = tnrange
    tqdm = tqdm_notebook
except NameError:
    pass

total_reinitialization_count = 0

# A NORMALIZATION LAYER
class NormalLinear(nn.Linear):
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.normal_(0, stdv)
        if self.bias is not None:
            self.bias.data.normal_(0, stdv)


# THE IACN MODULE
class IACN(nn.Module):
    def __init__(self, args, num_features, num_users, num_items):
        super(IACN,self).__init__()

        print("*** Initializing the IACN model ***")
        self.embedding_dim = args.embedding_dim
        self.num_users = num_users
        self.num_items = num_items
        self.user_static_embedding_size = num_users
        self.item_static_embedding_size = num_items

        print("Initializing user and item embeddings")
        self.initial_user_embedding = nn.Parameter(torch.Tensor(2*args.embedding_dim))
        self.initial_item_embedding = nn.Parameter(torch.Tensor(args.embedding_dim))

        rnn_input_size_items = rnn_input_size_users = self.embedding_dim + 1 + num_features

        print( "Initializing user and item RNNs")
        self.item_rnn = nn.RNNCell(rnn_input_size_users, self.embedding_dim)
        self.user_rnn = nn.RNNCell(rnn_input_size_items, self.embedding_dim*2)
        self.decay_rate = nn.RNNCell(2*self.embedding_dim, 1)

        print ("Initializing linear layers")
        self.linear_layer1 = nn.Linear(self.embedding_dim, 50)
        self.linear_layer2 = nn.Linear(50, 2)
        self.linear_layer3 = nn.Linear(self.embedding_dim, 1)
        self.linear_layer4 = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.linear_layer5 = nn.Linear(1,1)
        self.prediction_layer = nn.Linear(self.user_static_embedding_size + self.item_static_embedding_size + self.embedding_dim * 2, self.item_static_embedding_size + self.embedding_dim)
        self.prediction_layer2 = nn.Linear(self.user_static_embedding_size+ self.item_static_embedding_size+ 2*self.embedding_dim, self.user_static_embedding_size+self.embedding_dim)
        self.embedding_layer = NormalLinear(1, self.embedding_dim)
        #self.embedding_layer2 = NormalLinear(1, self.item_static_embedding_size+self.embedding_dim)
        print( "*** IACN initialization complete ***\n\n")
        
    def forward(self,args,  user_embeddings, item_embeddings, timediffs=None, features=None, select=None):
        if select == 'item_update':
            user_embeddings_input = user_embeddings[:,:args.embedding_dim]
            input1 = torch.cat([user_embeddings_input, timediffs, features], dim=1)
            item_embedding_output = self.item_rnn(input1, item_embeddings)
            return F.normalize(item_embedding_output)

        elif select == 'user_update':
            input2 = torch.cat([item_embeddings, timediffs, features], dim=1)
            user_embedding_output = self.user_rnn(input2, user_embeddings)
            return F.normalize(user_embedding_output)

    def compute_NN(self, itemids, userids, local_embeddings, item_user):

        item_userids = torch.nonzero(item_user[itemids]).split(1,dim=1)

        if len(item_userids)>0:
            local_embeddings[userids, item_userids[1]]+=1

        return local_embeddings


    def project_user(self, args, embeddings,  userids, local_embeddings, timediffs, timestamps, user_timestamp, user_embeddings,user_current_transaction_ids, alpha, delta):
        user_embedding = embeddings[:,args.embedding_dim:]
        delta_u = self.linear_layer3(user_embedding)

        user_embeddings_key = self.linear_layer4(user_embedding)

        user_timestamp_tensor = torch.t(user_timestamp.repeat(1,timestamps.shape[0]))


        timestamps_tensor = timestamps.unsqueeze(1).expand_as(user_timestamp_tensor)

        user_timestamp_tensor = timestamps_tensor-user_timestamp_tensor


        user_timestamp= scaler.fit_transform(np.transpose(user_timestamp_tensor.data.cpu().numpy()))

        user_timestamp_tensor= torch.t(torch.from_numpy(user_timestamp).cuda())

        delta_key = delta[userids,:]
        alpha_key = alpha[userids,:]

        exp_timediffs = torch.exp(-delta_key* user_timestamp_tensor)
        local_embeddings = local_embeddings[userids,:]
        temp1 = local_embeddings.clone()
        local_embeddings[local_embeddings==0]= -1000000
                  #print(temp1)
        local_embeddings = torch.where(torch.sum(temp1, dim=1).unsqueeze(1) ==0,
                                                                                                  torch.zeros(local_embeddings.shape).cuda(),torch.softmax(local_embeddings,dim=1))
        static_user_embeddings_coeff =  user_current_transaction_ids[userids,:]*local_embeddings*alpha_key*exp_timediffs #size = batchsize*num_users

        user_embeddings_query = user_embeddings[:, :args.embedding_dim]
        static_user_embeddings =torch.matmul(static_user_embeddings_coeff.clone(), user_embeddings_query.clone() ) #batch_size*embeddingdim

        new_embeddings = user_embeddings_key+static_user_embeddings

        return new_embeddings

    def predict_label(self, user_embeddings):
        X_out = nn.ReLU()(self.linear_layer1(user_embeddings))
        X_out = self.linear_layer2(X_out)
        return X_out

    def predict_item_embedding(self, user_embeddings):
        X_out = self.prediction_layer(user_embeddings)
        return X_out

    def predict_user_embedding(self, item_embeddings):
        X_out = self.prediction_layer2(item_embeddings)
        return X_out

# INITIALIZE T-BATCH VARIABLES
def reinitialize_tbatches():
    global current_tbatches_interactionids, current_tbatches_user, current_tbatches_item, current_tbatches_timestamp, current_tbatches_feature, current_tbatches_label, current_tbatches_previous_item
    global tbatchid_user, tbatchid_item, current_tbatches_user_timediffs, current_tbatches_item_timediffs, current_tbatches_user_timediffs_next, current_tbatches_timestamps, current_tbatches_item_user

    # list of users of each tbatch up to now
    current_tbatches_interactionids = defaultdict(list)
    current_tbatches_user = defaultdict(list)
    current_tbatches_item = defaultdict(list)
    current_tbatches_timestamp = defaultdict(list)
    current_tbatches_feature = defaultdict(list)
    current_tbatches_label = defaultdict(list)
    current_tbatches_previous_item = defaultdict(list)
    current_tbatches_user_timediffs = defaultdict(list)
    current_tbatches_item_timediffs = defaultdict(list)
    current_tbatches_user_timediffs_next = defaultdict(list)
    current_tbatches_user_timediffs_next= defaultdict(list)
    current_tbatches_timestamps = defaultdict(list)
    current_tbatches_item_user = defaultdict(lambda:defaultdict(list))

    # the latest tbatch a user is in
    tbatchid_user = defaultdict(lambda: -1)

    # the latest tbatch a item is in
    tbatchid_item = defaultdict(lambda: -1)

    global total_reinitialization_count
    total_reinitialization_count +=1


# CALCULATE LOSS FOR THE PREDICTED USER STATE 
def calculate_state_prediction_loss(model, tbatch_interactionids, user_embeddings_time_series, y_true, loss_function):
    # PREDCIT THE LABEL FROM THE USER DYNAMIC EMBEDDINGS
    prob = model.predict_label(user_embeddings_time_series[tbatch_interactionids,:])
    y = Variable(torch.LongTensor(y_true).cuda()[tbatch_interactionids])
    
    loss = loss_function(prob, y)

    return loss


# SAVE TRAINED MODEL TO DISK
def save_model(model, optimizer, args, epoch, user_embeddings, item_embeddings,local_embedding, train_end_idx, alpha, delta,item_user, user_current_timestamp, user_current_transactionids, user_embeddings_time_series=None, item_embeddings_time_series=None, path=PATH):
    print ("*** Saving embeddings and model ***")
    state = {
            'user_embeddings': user_embeddings.data.cpu().numpy(),
            'item_embeddings': item_embeddings.data.cpu().numpy(),
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
            'train_end_idx': train_end_idx,
            'alpha' : alpha.cpu().detach().numpy(),
            'delta' : delta.cpu().detach().numpy(),
            'item_user' : item_user.cpu().detach().numpy(),
            'user_current_timestamp' : user_current_timestamp.cpu().numpy(),
            'local_embedding': local_embedding.cpu().numpy(),
            'user_current_transactionids' :  user_current_transactionids.cpu().numpy()
            }

    if user_embeddings_time_series is not None and item_embeddings_time_series is not None:
        state['user_embeddings_time_series'] = user_embeddings_time_series.data.cpu().numpy()
        state['item_embeddings_time_series'] = item_embeddings_time_series.data.cpu().numpy()

    directory = os.path.join(path, 'saved_models/%s' % args.network)
    if not os.path.exists(directory):
        os.makedirs(directory)

    filename = os.path.join(directory, "checkpoint.ep%d.tp%.1f.pth.tar" % (epoch, args.train_proportion))
    torch.save(state, filename)
    print( "*** Saved embeddings and model to file: %s ***\n\n" % filename)


# LOAD PREVIOUSLY TRAINED AND SAVED MODEL
def load_model(model, optimizer, args, epoch):
    #print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    filename =  PATH+"saved_models/%s/checkpoint.ep%d.tp%.1f.pth.tar" % (args.network, epoch, args.train_proportion)
    checkpoint = torch.load(filename)
    print ("Loading saved embeddings and model: %s" % filename)
    args.start_epoch = checkpoint['epoch']
    user_embeddings = Variable(torch.from_numpy(checkpoint['user_embeddings']).cuda())
    item_embeddings = Variable(torch.from_numpy(checkpoint['item_embeddings']).cuda())
    alpha = Variable(torch.from_numpy(checkpoint['alpha']).cuda())
    delta = Variable(torch.from_numpy(checkpoint['delta']).cuda())
    user_current_timestamp = Variable(torch.from_numpy(checkpoint['user_current_timestamp']).cuda())
    local_embedding = Variable(torch.from_numpy(checkpoint['local_embedding']).cuda())
    item_user = Variable(torch.from_numpy(checkpoint['item_user']).cuda())
    user_current_transactionids = Variable(torch.from_numpy(checkpoint['user_current_transactionids']).cuda())
    try:
        train_end_idx = checkpoint['train_end_idx'] 
    except KeyError:
        train_end_idx = None

    try:
        user_embeddings_time_series = Variable(torch.from_numpy(checkpoint['user_embeddings_time_series']).cuda())
        item_embeddings_time_series = Variable(torch.from_numpy(checkpoint['item_embeddings_time_series']).cuda())
    except:
        user_embeddings_time_series = None
        item_embeddings_time_series = None

    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    return [model, optimizer, user_embeddings, item_embeddings,local_embedding, user_embeddings_time_series, item_embeddings_time_series, user_current_transactionids,train_end_idx, alpha, delta,item_user, user_current_timestamp]


# SET USER AND ITEM EMBEDDINGS TO THE END OF THE TRAINING PERIOD 
def set_embeddings_training_end(user_embeddings, item_embeddings, user_embeddings_time_series, item_embeddings_time_series, user_data_id, item_data_id, train_end_idx):
    userid2lastidx = {}
    for cnt, userid in enumerate(user_data_id[:train_end_idx]):
        userid2lastidx[userid] = cnt
    itemid2lastidx = {}
    for cnt, itemid in enumerate(item_data_id[:train_end_idx]):
        itemid2lastidx[itemid] = cnt

    try:
        embedding_dim = user_embeddings_time_series.size(1)
    except:
        embedding_dim = user_embeddings_time_series.shape[1]
    for userid in userid2lastidx:
        user_embeddings[userid, :embedding_dim] = user_embeddings_time_series[userid2lastidx[userid]]
    for itemid in itemid2lastidx:
        item_embeddings[itemid, :embedding_dim] = item_embeddings_time_series[itemid2lastidx[itemid]]

    user_embeddings.detach_()
    item_embeddings.detach_()

