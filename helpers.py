import torch
from torch import nn, utils
import torchmetrics
import matplotlib.pyplot as plt
from sklearn import preprocessing

def intersect_over_gt(bb, gt):
    """
    Calculate the fraction of groundtruth bounding box that is
    covered by the bounding box

    We adapt this code from https://stackoverflow.com/questions/25349178/calculating-percentage-of-bounding-box-overlap-for-image-detector-evaluation
    Parameters
    ----------
    bb : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    gt : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb['x1'] <= bb['x2']
    assert bb['y1'] <= bb['y2']
    assert gt['x1'] <= gt['x2']
    assert gt['y1'] <= gt['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb['x1'], gt['x1'])
    y_top = max(bb['y1'], gt['y1'])
    x_right = min(bb['x2'], gt['x2'])
    y_bottom = min(bb['y2'], gt['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box.
    # NOTE: We MUST ALWAYS add +1 to calculate area when working in
    # screen coordinates, since 0,0 is the top left pixel, and w-1,h-1
    # is the bottom right pixel. If we DON'T add +1, the result is wrong.
    intersection_area = (x_right - x_left + 1) * (y_bottom - y_top + 1)

    # compute the area of both AABBs
    bb_area = (bb['x2'] - bb['x1'] + 1) * (bb['y2'] - bb['y1'] + 1)
    gt_area = (gt['x2'] - gt['x1'] + 1) * (gt['y2'] - gt['y1'] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    io_gt = intersection_area / float(gt_area)
    #print("Bounding box")
    #print(bb)
    #print("Ground truth")
    #print(gt)
    #print(f"Percent overlap {io_gt}")
    assert io_gt >= 0.0
    assert io_gt <= 1.0
    return io_gt

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
    
        self.classify = nn.Linear(representation_dim, n_classes)

    def forward(self, x, is_train=False):
        x = self.model(x)
        if is_train:
            x = self.classify(x)
        return x
    
def train_learned_representation(data, label_dict, n_epochs=50, batch_size=32, cache_path='cache/'):
    vild_embeddings = torch.Tensor(data["vild"])
    _3d_positions = torch.Tensor(data["position"])

    embeddings = torch.cat([vild_embeddings, _3d_positions], dim=1)

    le = preprocessing.LabelEncoder()
    labels = torch.Tensor(le.fit_transform(data["label"])).long()
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
