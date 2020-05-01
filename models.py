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
import pandas as pd
import os

class CNN5_LSTM5(nn.Module):
    def __init__(self, num_classes, img_height=128):
        super().__init__()
        self.conv_keep_prob = 0.8
        self.blstm_keep_prob = 0.5
        self.blstm_units = 256
        self.blstm_layers = 5
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(1, 16, 3, 1, padding=1)
        self.conv1_bn = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, padding=1)
        self.conv2_bn = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3, 1, padding=1)
        self.conv3_bn = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 48, 3, 1, padding=1)
        self.conv4_bn = nn.BatchNorm2d(48)
        self.conv5 = nn.Conv2d(48, 80, 3, 1, padding=1)
        self.conv5_bn = nn.BatchNorm2d(80)


        T.nn.init.xavier_normal_(self.conv1.weight), T.nn.init.zeros_(self.conv1.bias)
        T.nn.init.xavier_normal_(self.conv2.weight), T.nn.init.zeros_(self.conv2.bias)
        T.nn.init.xavier_normal_(self.conv3.weight), T.nn.init.zeros_(self.conv3.bias)
        T.nn.init.xavier_normal_(self.conv4.weight), T.nn.init.zeros_(self.conv4.bias)
        T.nn.init.xavier_normal_(self.conv5.weight), T.nn.init.zeros_(self.conv5.bias)

        self.conv_out_features = int(80 * img_height / 8)

        self.packer = T.nn.utils.rnn.pack_padded_sequence

        self.blstm_block = nn.LSTM(self.conv_out_features, self.blstm_units, self.blstm_layers, bidirectional=True,
                                   batch_first=True, dropout=(1 - self.blstm_keep_prob))

        self.unpacker = T.nn.utils.rnn.pad_packed_sequence

        self.linear = nn.Linear(2*self.blstm_units, self.num_classes)
        T.nn.init.normal_(self.linear.weight, 0, 0.01), T.nn.init.zeros_(self.linear.bias)

    def forward(self, x, inputs_lengths):
        x = F.dropout2d(F.max_pool2d(F.leaky_relu(self.conv1_bn(self.conv1(x))), 2, 2), 1 - self.conv_keep_prob)
        x = F.dropout2d(F.max_pool2d(F.leaky_relu(self.conv2_bn(self.conv2(x))), 2, 2), 1 - self.conv_keep_prob)
        x = F.dropout2d(F.max_pool2d(F.leaky_relu(self.conv3_bn(self.conv3(x))), 2, 2), 1 - self.conv_keep_prob)
        x = F.dropout2d(F.leaky_relu(self.conv4_bn(self.conv4(x))), 1 - self.conv_keep_prob)
        x = F.dropout2d(F.leaky_relu(self.conv5_bn(self.conv5(x))), 1 - self.conv_keep_prob)
        outputs_lengths = inputs_lengths / 8

        # CNN outputs = N,C,H,W

        x = x.view(-1, x.shape[1] * x.shape[2], x.shape[3])  # (N, C*H, W)

        x = x.transpose(2, 1)  # (N, W, C*H)

        x = self.packer(x, outputs_lengths, batch_first=True, enforce_sorted=False)

        x, _ = self.blstm_block(x)  # (N, W, C)

        x, outputs_lengths = self.unpacker(x, batch_first=True)

        x = self.linear(x)

        x = x.transpose(1, 0)  # (W,N,C)

        x = x.log_softmax(2)

        return x, outputs_lengths


def levenshtein(s1, s2):
    if len(s1) < len(s2):
        return levenshtein(s2, s1)

    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[
                             j + 1] + 1  # j+1 instead of j since previous_row and current_row are one character longer
            deletions = current_row[j] + 1  # than s2
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

T.manual_seed(0)
np.random.seed(0)
tf.set_random_seed(0)

EPOCHS = 200
BATCH_SIZE = 16

# txt_table = 'tmp/transcriptions.txt'
txt_table = '/media/HDD/Databases/ICFHR2018_Competition/transcriptions.txt'
img_dirs = ['/media/HDD/Databases/ICFHR2018_Competition/polygon_general_data']
char_list_file = '/media/HDD/Databases/ICFHR2018_Competition/cm_final.txt'

height = 128

dataset = TextImageFromTextTableDataset(txt_table, char_list_file, img_dirs, img_transform=transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    Resize(height),
    transforms.ToTensor(),
]), pad_img=False, pad_txt=False)

tr_dataset, va_dataset = T.utils.data.random_split(dataset, [round(len(dataset) * 0.99), round(len(dataset) * 0.01)])

train_loader = ImageDataLoader(dataset=tr_dataset,
                               batch_size=BATCH_SIZE,
                               image_channels=1,
                               image_height=None,
                               image_width=None,
                               shuffle=True,
                               num_workers=3)

valid_loader = ImageDataLoader(dataset=va_dataset,
                               batch_size=BATCH_SIZE,
                               image_channels=1,
                               image_height=None,
                               image_width=None,
                               shuffle=False)

num_classes = len(dataset.char2num) + 1  # for CTC blanck

# img = T.rand(3,1,128,250)
if T.cuda.is_available():
    device = T.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
    print("Running on the GPU")
else:
    device = T.device("cpu")
    print("Running on the CPU")

if not os.path.exists("./models"):
    os.mkdir("./models")

model_path = os.path.join("./models", f"CNN5_LSTM5_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}")

net = CNN5_LSTM5(num_classes=num_classes, img_height=height).to(device)
# print(net)
# logits = net(img)
optimizer = optim.Adam(net.parameters(), lr=0.0003)
loss_function = nn.CTCLoss(blank=num_classes - 1, reduction='none', zero_infinity=True)

# x.log_softmax(2).detach(), y, x_lengths, y_lengths)

N_BATCHES = len(tr_dataset)/BATCH_SIZE
df = pd.DataFrame(columns=['Epoch', 'CER', 'Loss'])

for epoch in range(EPOCHS):
    net.train(True)
    total_loss = 0
    for batch in tqdm(
            train_loader):  # from 0, to the len of x, stepping BATCH_SIZE at a time. [:50] ..for now just to dev
        # print(f"{i}:{i+BATCH_SIZE}")
        X = batch['img'][0].to(device)
        y = batch['seq'][0].to(device)
        x_lengths = batch['img'][1][:, 2].to(device)
        y_lengths = batch['seq'][1].to(device)

        net.zero_grad()

        outputs, output_lengths = net(X, x_lengths)
        loss = T.mean(loss_function(outputs, y, output_lengths, y_lengths))
        loss.backward()
        optimizer.step()  # Does the update

        total_loss += loss/N_BATCHES

    print(f"Epoch: {epoch}. Loss: {total_loss}")
    
    if epoch % 25 == 0:
        T.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, model_path)

        with T.no_grad():
            net.eval()
            CER = 0
            for batch in valid_loader:
                X = batch['img'][0].to(device)
                y = batch['seq'][0].to(device)
    
                x_lengths = batch['img'][1][:, 2]
                y_lengths = batch['seq'][1].numpy()
                txt = batch['txt']
                outputs, output_lengths = net(X, x_lengths)
                outputs = outputs.transpose(1, 0).cpu().detach().numpy()
                output_lengths = output_lengths.numpy()
                for ind, (output, output_length) in enumerate(zip(outputs, output_lengths)):
                    output = output.reshape([1,output.shape[0], output.shape[1]]).transpose(1, 0, 2)
                    output_length = np.array([output_length])
                    predictions, log_prob = tf.nn.ctc_beam_search_decoder(output, output_length, merge_repeated=False)
                    with tf.Session() as sess:
                        pred = sess.run(predictions)
                        CER += float(levenshtein(y[ind,:y_lengths[ind, 0]], pred[0].values))/y_lengths[ind,0]/len(va_dataset)
                        print(f"Gt: {txt[ind]}")
                        print('Pr: ' + ''.join(
                            [dataset.num2char[pred[0].values[ind]] for ind in range(pred[0].values.shape[0])]))
                        print('\n')
            print(f"CER: {CER}\n")
        df = df.append({'Epoch': epoch, 'CER': CER, 'Loss': total_loss.detach()}, ignore_index=True)
df.to_csv('results.csv')