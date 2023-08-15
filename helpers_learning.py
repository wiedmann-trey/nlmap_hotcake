import torch
from torch import nn, utils
import torchmetrics
import matplotlib.pyplot as plt
from sklearn import preprocessing

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: https://towardsdatascience.com/a-friendly-introduction-to-siamese-networks-85ab17522942
    """

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, x0, x1, y):
        # euclidian distance
        diff = x0 - x1
        dist_sq = torch.sum(torch.pow(diff, 2), 1)
        dist = torch.sqrt(dist_sq)

        mdist = self.margin - dist
        dist = torch.clamp(mdist, min=0.0)
        loss = y * dist_sq + (1 - y) * torch.pow(dist, 2)
        loss = torch.sum(loss) / 2.0 / x0.size()[0]
        return loss

class SiameseNetwork(nn.Module):
    def __init__(self, vild_dim=512, position_dim=3, representation_dim=256):
        super(SiameseNetwork, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(vild_dim+position_dim, representation_dim),
            nn.ReLU(),
            nn.Linear(representation_dim, representation_dim),
            nn.ReLU(),
            nn.Linear(representation_dim, representation_dim),
            nn.ReLU(),
            nn.Linear(representation_dim, representation_dim)
        )
        self.representation_dim = representation_dim

    def forward(self, x):
        x = self.model(x)
        return x

class LearnRepresentation(nn.Module):
    def __init__(self, n_classes, vild_dim=512, position_dim=3, representation_dim=256):
        super(LearnRepresentation, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(vild_dim+position_dim, representation_dim),
            nn.ReLU(),
            nn.Linear(representation_dim, representation_dim),
            nn.ReLU(),
            nn.Linear(representation_dim, representation_dim),
            nn.ReLU(),
            nn.Linear(representation_dim, representation_dim)
        )
        self.representation_dim = representation_dim
        self.classify = nn.Linear(representation_dim, n_classes)

    def forward(self, x, is_train=False):
        x = self.model(x)
        if is_train:
            x = self.classify(x)
        return x

def train_siamese_network(data, batch_size=32, n_epochs=5, cache_path='/cache'):
    data_len = min(250, len(data['label']))

    le = preprocessing.LabelEncoder()
    labels = torch.Tensor(le.fit_transform(data["label"][:data_len])).long()
    class_n = len(le.classes_)
    print(le.classes_)

    paired_list = []
    embeddings_1, embeddings_2, does_match = [],[],[]

    for i in range(data_len):
        for j in range(i+1, data_len):
            is_same_object = labels[i] == labels[j]
            embeddings_1.append(data['vild'][i]+data['position'][i])
            embeddings_2.append(data['vild'][j]+data['position'][j])
            does_match.append(1 if is_same_object else 0)

    dataset = utils.data.TensorDataset(torch.Tensor(embeddings_1), torch.Tensor(embeddings_2), torch.Tensor(does_match))
    train_set, validate_set = utils.data.random_split(dataset, [.8, .2])
    train, validate = utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True), utils.data.DataLoader(validate_set, batch_size=batch_size)

    model = SiameseNetwork()
    optimizer = torch.optim.Adam(model.parameters())
    lossfn = ContrastiveLoss()

            
    losses = []
    for epoch in range(n_epochs):
        epoch_loss = 0
        n_batches = 0
        for batch_idx, sample in enumerate(train):
            batch_emb_1, batch_emb_2, label = sample
            optimizer.zero_grad()

            model_1, model_2 = model(batch_emb_1), model(batch_emb_2)
            loss = lossfn(model_1, model_2, label)

            loss.backward()
            optimizer.step()

        for batch_idx, sample in enumerate(validate):
            batch_emb_1, batch_emb_2, label = sample
            with torch.no_grad():
                model_1, model_2 = model(batch_emb_1), model(batch_emb_2)
                loss = lossfn(model_1, model_2, label)
                epoch_loss += float(loss)
                n_batches += 1

        losses.append(epoch_loss/n_batches)
        
        torch.save(model.state_dict(), cache_path+"learned_representation_model_"+str(epoch))

    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Validation Loss', color='tab:red')
    ax1.plot(losses, color='tab:red')
    ax1.tick_params(axis='y', labelcolor='tab:red')

    fig.tight_layout()
    plt.savefig(cache_path+"siamese_loss_curve.png")


def train_learned_representation(data, label_dict, n_epochs=30, batch_size=32, cache_path='cache/'):
    data_len = min(250, len(data['label']))
    vild_embeddings = torch.Tensor(data["vild"][:data_len])
    _3d_positions = torch.Tensor(data["position"][:data_len])

    embeddings = torch.cat([vild_embeddings, _3d_positions], dim=1)

    le = preprocessing.LabelEncoder()
    labels = torch.Tensor(le.fit_transform(data["label"][:data_len])).long()
    class_n = len(le.classes_)
    print(le.classes_)

    dataset = utils.data.TensorDataset(embeddings, labels)
    train_set, validate_set = utils.data.random_split(dataset, [.8, .2])
    train, validate = utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True), utils.data.DataLoader(validate_set, batch_size=batch_size)

    model = LearnRepresentation(class_n)

    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss()
    accuracy_fn = torchmetrics.classification.MulticlassAccuracy(class_n)
    accuracy_fn2 = torchmetrics.classification.MulticlassAccuracy(class_n, top_k=3)

    train_losses = []
    validation_accs = []
    validation_losses = []

    for epoch in range(n_epochs):
        
        train_acc = 0
        train_loss = 0
        n_train_samples = 0
        for batch_idx, sample in enumerate(train):
            batch_emb, batch_label = sample
            optimizer.zero_grad()

            logits = model(batch_emb, is_train=True)
            loss = loss_fn(logits, batch_label)
            accuracy = accuracy_fn(logits, batch_label)

            train_loss += float(loss)
            train_acc += float(accuracy)
            n_train_samples += 1

            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch} train accuracy: {train_acc/n_train_samples}")
        train_losses.append(train_loss/n_train_samples)

        total_acc = 0
        total_loss = 0
        n_samples = 0
        for batch_idx, sample in enumerate(validate):
            batch_emb, batch_label = sample
            n_samples += 1
            with torch.no_grad():
                logits = model(batch_emb, is_train=True)
                accuracy = accuracy_fn(logits, batch_label)
                loss = loss_fn(logits, batch_label)
                total_acc += accuracy
                total_loss += loss
        print(f"Epoch {epoch} validation accuracy: {total_acc/n_samples} validation loss: {total_loss/n_samples}")
        validation_accs.append(total_acc/n_samples)
        validation_losses.append(total_loss/n_samples)

    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Train Loss', color='tab:red')
    ax1.plot(train_losses, color='tab:red')
    ax1.plot(validation_losses, color='tab:orange')
    ax1.tick_params(axis='y', labelcolor='tab:red')

    ax2 = ax1.twinx() 

    ax2.set_ylabel('Validation Accuracy', color='tab:blue') 
    ax2.plot(validation_accs, color='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    fig.tight_layout()
    plt.savefig(cache_path+"loss_curve.png")

    torch.save(model.state_dict(), cache_path+"learned_representation_model")
    with open('cache/number_classes.txt', 'w') as f:
        f.write(str(class_n))
    return model
