import torch
import torch.nn as nn
import torch.nn.functional as F

class MetricNN_V2(nn.Module):
    """docstring for MetricNN_V2."""

    def __init__(self, input_cptn_size, input_label_size, input_visual_size,
                    hidden_sizes, output_size):
        super(MetricNN_V2, self).__init__()
        self.linear_caption = nn.Linear(input_cptn_size, hidden_sizes[0],
                                        bias=True)
        self.caption_mlp = []
        for idx in range(len(hidden_sizes)-1):
            self.caption_mlp.append(nn.Linear(hidden_sizes[idx],
                                            hidden_sizes[idx+1],
                                            bias=True))
            self.caption_mlp.append(nn.ReLU(inplace=True))
        self.caption_mlp = nn.Sequential(*self.caption_mlp)

        self.linear_label = nn.Linear(input_label_size, hidden_sizes[0],
                                        bias=True)
        self.label_mlp = []
        for idx in range(len(hidden_sizes)-1):
            self.label_mlp.append(nn.Linear(hidden_sizes[idx],
                                            hidden_sizes[idx+1],
                                            bias=True))
            self.label_mlp.append(nn.ReLU(inplace=True))
        self.label_mlp = nn.Sequential(*self.label_mlp)

        self.linear_visual = nn.Linear(input_visual_size, hidden_sizes[0],
                                        bias=True)
        self.visual_mlp = []
        for idx in range(len(hidden_sizes)-1):
            self.visual_mlp.append(nn.Linear(hidden_sizes[idx],
                                            hidden_sizes[idx+1],
                                            bias=True))
            self.visual_mlp.append(nn.ReLU(inplace=True))
        self.visual_mlp = nn.Sequential(*self.visual_mlp)

        self.linear_summarize = nn.Linear(5*hidden_sizes[-1], hidden_sizes[0],
                                            bias=True)
        self.score_mlp = []
        for idx in range(len(hidden_sizes)-1):
            self.score_mlp.append(nn.Linear(hidden_sizes[idx],
                                            hidden_sizes[idx+1],
                                            bias=True))
            self.score_mlp.append(nn.ReLU(inplace=True))
        self.score_mlp = nn.Sequential(*self.score_mlp)

        self.linear_score = nn.Linear(hidden_sizes[-1], output_size,
                                        bias=True)

    def forward(self, c1, l1, c2, l2, v1):
        # Caption linear embedding
        c1_out = self.linear_caption(c1)
        c1_out = self.caption_mlp(c1_out)
        c2_out = self.linear_caption(c2)
        c2_out = self.caption_mlp(c2_out)

        # Label linear embedding
        l1_out = self.linear_label(l1)
        l1_out = self.label_mlp(l1_out)
        l2_out = self.linear_label(l2)
        l2_out = self.label_mlp(l2_out)

        # Visual linear embedding
        v1_out = self.linear_visual(v1)
        v1_out = self.visual_mlp(v1_out)

        # Score calculation
        summary = self.linear_summarize(torch.cat((c1_out, c2_out, l1_out, l2_out, v1_out),axis=1))
        mlp_out = self.score_mlp(summary)
        score = self.linear_score(mlp_out)

        return score


class MetricNN(nn.Module):
    """docstring for MetricNN."""

    def __init__(self, input_cptn_size, input_label_size, input_visual_size,
                    hidden_sizes, output_size):
        super(MetricNN, self).__init__()
        self.linear_caption = nn.Linear(input_cptn_size, hidden_sizes[0],
                                        bias=True)
        self.linear_label = nn.Linear(input_label_size, hidden_sizes[0],
                                        bias=True)
        self.linear_visual = nn.Linear(input_visual_size, hidden_sizes[0],
                                        bias=True)

        self.linear_summarize = nn.Linear(5*hidden_sizes[0], hidden_sizes[0],
                                            bias=True)
        self.score_mlp = []
        for idx in range(len(hidden_sizes)-1):
            self.score_mlp.append(nn.Linear(hidden_sizes[idx],
                                            hidden_sizes[idx+1],
                                            bias=True))
            self.score_mlp.append(nn.ReLU(inplace=True))

        self.score_mlp = nn.Sequential(*self.score_mlp)

        self.linear_score = nn.Linear(hidden_sizes[-1], output_size,
                                        bias=True)

    def forward(self, c1, l1, c2, l2, v1):
        # Caption linear embedding
        c1_out = self.linear_caption(c1)
        c2_out = self.linear_caption(c2)

        # Label linear embedding
        l1_out = self.linear_label(l1)
        l2_out = self.linear_label(l2)

        # Visual linear embedding
        v1_out = self.linear_visual(v1)

        # Score calculation
        summary = self.linear_summarize(torch.cat((c1_out, c2_out, l1_out, l2_out, v1_out),axis=1))
        mlp_out = self.score_mlp(summary)
        score = self.linear_score(mlp_out)

        return score


def create_net(input_cptn_size, input_label_size, input_visual_size,
                hidden_sizes, output_size, lr):
    # 1. Create the model
    # net = MetricNN(input_cptn_size, input_label_size, input_visual_size,
    #                 hidden_sizes, output_size)
    net = MetricNN_V2(input_cptn_size, input_label_size, input_visual_size,
                    hidden_sizes, output_size)

    # 2. Weight Initialization
    net.apply(xavier_init_weights)

    # 3. MSE Loss as criterion
    criterion = nn.MSELoss()

    # 4. Optimizer
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)

    # 5. LR Scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.01)

    return net, criterion, optimizer, lr_scheduler


def xavier_init_weights(layer):
    if type(layer) == nn.Linear:
        nn.init.xavier_uniform_(layer.weight)


# class BiLSTM(torch.nn.Module):
#     def __init__(self, in_vocab, hidden_size, out_vocab, num_layers):
#         super(BiLSTM, self).__init__()
#         self.lstm = nn.LSTM(in_vocab, hidden_size, bidirectional=True,
#                                   num_layers=num_layers, dropout=0.3)
#         self.output = nn.Linear(hidden_size * 2, hidden_size * 4)
#         self.output2 = nn.Linear(hidden_size * 4, hidden_size * 8)
#         self.output3 = nn.Linear(hidden_size * 8, out_vocab)
#
#     def forward(self, X):
#         # print("\nShape of Input X:\n\t", X.shape)
#
#         # packed_X = nn.utils.rnn.pack_sequence(X, lengths, enforce_sorted=False)
#         packed_X = X
#         # print("\nShapes in packed embedding: \n\t", [px.shape for px in packed_X])
#
#         packed_out = self.lstm(packed_X)[0]
#         # print("\nShape of LSTM Output: \n\t", packed_out.data.shape)
#
#         out, out_lens = nn.utils.rnn.pad_packed_sequence(packed_out)
#         # print("\nShape of padded LSTM Output: \n\t", out.shape)
#         # print("\nShape of padded LSTM Output lengths: \n\t", out_lens.shape)
#
#         out = self.output(out)
#         out = nn.functional.dropout(out,p=0.3)
#         out = self.output2(out)
#         out = nn.functional.dropout(out,p=0.3)
#         out = self.output3(out).log_softmax(2)
#         # print("\nShape of post-softmax of packed LSTM Output:\n\t", out.shape)
#         return out, out_lens
#
# class Network(nn.Module):
#     def __init__(self, num_feats, hidden_sizes, num_classes, feat_dim=10):
#         super(Network, self).__init__()
#
#         self.hidden_sizes = [num_feats] + hidden_sizes + [num_classes]
#
#         self.layers = []
#         for idx, channel_size in enumerate(hidden_sizes):
#             self.layers.append(nn.Conv2d(in_channels=self.hidden_sizes[idx],
#                                          out_channels=self.hidden_sizes[idx+1],
#                                          kernel_size=3, stride=2, bias=False))
#             self.layers.append(nn.ReLU(inplace=True))
#             self.layers.append(BasicBlock(channel_size = channel_size))
#
#         self.layers = nn.Sequential(*self.layers)
#         self.linear_label = nn.Linear(self.hidden_sizes[-2],
#                                       self.hidden_sizes[-1], bias=False)
#
#         # For creating the embedding to be passed into the Center Loss criterion
#         self.linear_closs = nn.Linear(self.hidden_sizes[-2], feat_dim, bias=False)
#         self.relu_closs = nn.ReLU(inplace=True)
#
#     def forward(self, x, evalMode=False):
#         output = x
#         output = self.layers(output)
#
#         output = F.avg_pool2d(output, [output.size(2), output.size(3)], stride=1)
#         output = output.reshape(output.shape[0], output.shape[1])
#
#         label_output = self.linear_label(output)
#         label_output = label_output/torch.norm(self.linear_label.weight, dim=1)
#
#         # Create the feature embedding for the Center Loss
#         closs_output = self.linear_closs(output)
#         closs_output = self.relu_closs(closs_output)
#
#         return closs_output, label_output
