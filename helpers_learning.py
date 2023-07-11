import torch
from torch import nn, utils
import torchmetrics
import matplotlib.pyplot as plt
from sklearn import preprocessing

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