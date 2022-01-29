import torch
from tqdm import tqdm


class PNCF(torch.nn.Module):
    def __init__(self,
                 writer,
                 uvec,
                 ivec,
                 layer_nums=3
                 ):
        """PNCF Recommender 

        Args:
            writer (torch.utils.tensorboard.SummaryWriter): tensorboard writer
            uvec (list): user's vectors
            ivec (list): items'vectors
            layer_nums (int, optional): MLP layer nums. Defaults to 3.
        """
        super(PNCF, self).__init__()
        freeze = False
        self.writer = writer
        self.layer_nums = layer_nums
        self.user_embedding = torch.nn.Embedding.from_pretrained(
            torch.tensor(uvec), freeze=freeze)
        self.item_embedding = torch.nn.Embedding.from_pretrained(
            torch.tensor(ivec), freeze=freeze)
        in_length = self.user_embedding.weight.shape[1] + \
            self.item_embedding.weight.shape[1]
        layers = []
        for i in range(self.layer_nums, 0, -1):
            input_size = in_length * 2 ** i if self.layer_nums != i else in_length
            next_size = in_length * 2 ** (i - 1)
            layers.append(
                torch.nn.Linear(input_size, next_size)
            )
            layers.append(torch.nn.BatchNorm1d(next_size)),
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout())
        layers.append(torch.nn.Linear(in_length, 1))
        layers.append(torch.nn.Sigmoid())
        self.mlp = torch.nn.Sequential(*layers)

        def init_weights(m):
            if type(m) == torch.nn.Linear:
                torch.nn.init.normal_(m.weight, std=0.01)

        self.mlp.apply(init_weights)

        self.early_stop = True

    def forward(self, user_idxs, tag_idxs):
        user_vec = self.user_embedding(user_idxs)
        tag_vec = self.item_embedding(tag_idxs)
        input = torch.cat([user_vec, tag_vec], dim=1)
        output_MLP = self.mlp(input)
        return output_MLP.view(-1)

    def fit(self, train_loader, lr: int = 0.01, epochs: int = 1):
        if torch.cuda.is_available():
            self.cuda()
        else:
            self.cpu()
        criterion = torch.nn.MSELoss(reduction='sum')
        optimizer = torch.optim.Adam(
            self.parameters(), lr=lr)
        last_loss = 0.
        self.writer.add_graph(
            self, (torch.tensor([0]).cuda(), torch.tensor([0]).cuda()))
        count = 0
        for epoch in range(1, epochs + 1):
            current_loss = 0.
            # set process bar display
            pbar = tqdm(train_loader)
            pbar.set_description(f'[Epoch {epoch:03d}]')

            for user, item, label in pbar:
                self.train()
                count += 1
                if torch.cuda.is_available():
                    user = user.cuda()
                    item = item.cuda()
                    label = label.cuda().float()
                else:
                    user = user.cpu()
                    item = item.cpu()
                    label = label.cpu().float()

                self.zero_grad()
                prediction = self.forward(user, item)
                loss = criterion(prediction, label)
                if torch.isnan(loss):
                    raise ValueError(
                        f'Loss=Nan or Infinity: current settings does not fit the recommender')
                loss.backward()
                optimizer.step()
                l = loss.item()
                self.eval()

                if count % 100 == 0:
                    pbar.set_postfix(loss="%.2f" % l)
                    self.writer.add_scalar('Train/Loss', l, count)
                    self.writer.flush()
                current_loss += l

            delta_loss = float(current_loss - last_loss)
            if (abs(delta_loss) < 1e-5) and self.early_stop:
                print('Satisfy early stop mechanism')
                break
            else:
                last_loss = current_loss

    def get_test_accuracy(self):
        preds = self.forward(self.uidxs, self.iidxs)
        return self.get_accuracy(torch.ones(preds.shape[0]).cuda(), preds)

    def get_accuracy(self, label, prediction):
        return torch.sum(label == torch.tensor(
            [0 if i < 0.5 else 1 for i in prediction]).cuda()) / label.shape[0]
