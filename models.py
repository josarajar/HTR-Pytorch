import torch as T
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from text_loader_prueba import TextImageFromTextTableDataset, ImageDataLoader
from torchvision import transforms
import numpy as np
import tensorflow as tf
from transforms import Resize

class CNN5_LSTM5(nn.Module):
    def __init__(self, num_classes, img_height=128):
        super().__init__()
        self.conv_keep_prob = 0.8
        self.blstm_keep_prob = 0.5
        self.blstm_units = 256
        self.blstm_layers = 5
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(1,16,3,1,padding=1)
        self.conv2 = nn.Conv2d(16,32,3,1,padding=1)
        self.conv3 = nn.Conv2d(32,64,3,1,padding=1)
        self.conv4 = nn.Conv2d(64,48,3,1,padding=1)
        self.conv5 = nn.Conv2d(48,80,3,1,padding=1)

        T.nn.init.xavier_normal_(self.conv1.weight), T.nn.init.zeros_(self.conv1.bias)
        T.nn.init.xavier_normal_(self.conv2.weight), T.nn.init.zeros_(self.conv2.bias)
        T.nn.init.xavier_normal_(self.conv3.weight), T.nn.init.zeros_(self.conv3.bias)
        T.nn.init.xavier_normal_(self.conv4.weight), T.nn.init.zeros_(self.conv4.bias)
        T.nn.init.xavier_normal_(self.conv5.weight), T.nn.init.zeros_(self.conv5.bias)

        self.conv_out_features = int(80*img_height/4)

        self.blstm_block = nn.LSTM(self.conv_out_features, self.blstm_units, self.blstm_layers, bidirectional=True, batch_first=True, dropout=(1-self.blstm_keep_prob))

        self.linear = nn.Linear(512, self.num_classes)
        T.nn.init.normal_(self.linear.weight,0,0.01), T.nn.init.zeros_(self.linear.bias)

    def forward(self, x, inputs_lengths):
        x = F.dropout2d(F.max_pool2d(F.leaky_relu(self.conv1(x)), 2, 2), 1 - self.conv_keep_prob)
        x = F.dropout2d(F.max_pool2d(F.leaky_relu(self.conv2(x)), 2, 2), 1 - self.conv_keep_prob)
        x = F.dropout2d(F.max_pool2d(F.leaky_relu(self.conv3(x)), 1, 1), 1 - self.conv_keep_prob)
        x = F.dropout2d(F.leaky_relu(self.conv4(x)), 1 - self.conv_keep_prob)
        x = F.dropout2d(F.leaky_relu(self.conv5(x)), 1 - self.conv_keep_prob)
        inputs_lengths = inputs_lengths/4

        # CNN outputs = N,C,H,W

        x = x.view(-1, x.shape[1]*x.shape[2], x.shape[3]) # (N, C*H, W)

        x = x.transpose(2, 1)  # (N, W, C*H)

        x, _ = self.blstm_block(x)  # (N, W, C)

        x = self.linear(x)

        x = x.transpose(1,0)  # (W,N,C)

        return x, inputs_lengths


T.manual_seed(0)
np.random.seed(0)
tf.set_random_seed(0)

EPOCHS = 100
BATCH_SIZE = 16

txt_table = 'tmp/Database/transcriptions.txt'
img_dirs = ['/Users/aradillas/Documents/polygon_general_data']
char_list_file = '/Users/aradillas/Desktop/cm_final.txt'

height = 64

dataset = TextImageFromTextTableDataset(txt_table, char_list_file, img_dirs, img_transform=transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    Resize(height),
    transforms.ToTensor(),
    ]), pad_img=False,  pad_txt=False)

tr_dataset, va_dataset = T.utils.data.random_split(dataset, [round(len(dataset)*0.9), round(len(dataset)*0.1)])
tr_dataset = dataset

train_loader = ImageDataLoader(dataset=tr_dataset,
                             batch_size=BATCH_SIZE,
                             image_channels=1,
                             image_height=None,
                             image_width=None,
                             shuffle=True,
                             num_workers=3)

valid_loader = ImageDataLoader(dataset=va_dataset,
                             batch_size=1,
                             image_channels=1,
                             image_height=None,
                             image_width=None,
                             shuffle=False)

num_classes = len(tr_dataset.char2num) + 1  # for CTC blanck

#img = T.rand(3,1,128,250)
net = CNN5_LSTM5(num_classes=num_classes, img_height=height)
#print(net)
#logits = net(img)
optimizer = optim.Adam(net.parameters(), lr=0.0003)
loss_function = nn.CTCLoss(blank=num_classes - 1, reduction='none', zero_infinity=True)

#x.log_softmax(2).detach(), y, x_lengths, y_lengths)


for epoch in range(EPOCHS):
    for batch in tqdm(train_loader): # from 0, to the len of x, stepping BATCH_SIZE at a time. [:50] ..for now just to dev
        #print(f"{i}:{i+BATCH_SIZE}")
        X = batch['img'][0]
        print(X.shape)
        y = batch['seq'][0]
        x_lengths = batch['img'][1][:,2]
        y_lengths = batch['seq'][1]

        net.zero_grad()

        outputs, output_lengths = net(X, x_lengths)
        loss = T.mean(loss_function(outputs, y, output_lengths, y_lengths))
        loss.backward()
        optimizer.step()    # Does the update

    with T.no_grad():
        for batch in valid_loader:
            X = batch['img'][0]
            y = batch['seq'][0]

            x_lengths = batch['img'][1][:, 2]
            y_lengths = batch['seq'][1].numpy()
            txt = batch['txt']
            outputs, output_lengths = net(X, x_lengths)
            outputs = outputs.detach().numpy()
            output_lengths = output_lengths.numpy()
            predictions, log_prob = tf.nn.ctc_beam_search_decoder(outputs, output_lengths, merge_repeated=False)
            with tf.Session() as sess:
                pred = sess.run(predictions)
                print(f"Gt: {txt[0]}")
                print('Pr: ' + ' '.join([tr_dataset.num2char[pred[0].values[ind]] for ind in range(pred[0].values.shape[0])]))

    print(f"Epoch: {epoch}. Loss: {loss}")

