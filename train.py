import torch
import torch.utils.data
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from engine import train_one_epoch, evaluate
import utils
from dataset import Dataset
from transforms import get_transform

DATA_DIR = '/home/master/dataset/train/'
SAVE_MODEL = ('fasterRCNN')

# load a pre-trained model for classification and return
# only the features
backbone = torchvision.models.mobilenet_v2(pretrained=True).features
# FasterRCNN needs to know the number of
# output channels in a backbone. For mobilenet_v2, it's 12img
# so we need to add it here
backbone.out_channels = 1280

# let's make the RPN generate 5 x 3 anchors per spatial
# location, with 5 different sizes and 3 different aspect
# ratios. We have a Tuple[Tuple[int]] because each featureimg
# map could potentially have different sizes and
# aspect ratios
anchor_generator = AnchorGenerator(
    sizes=((32, 64, 128, 256),), aspect_ratios=((0.5, 1.0, 2.0),)
)

# let's define what are the feature maps that we will
# use to perform the region of interest cropping, as well as
# the size of the crop after rescaling.
# if your backbone returns a Tensor, featmap_names is expected to
# be [0]. More generally, the backbone should return an
# OrderedDict[Tensor], and in featmap_names you can choose which
# feature maps to use
roi_pooler = torchvision.ops.MultiScaleRoIAlign(
    featmap_names=["0"], output_size=7, sampling_ratio=2
)


# put the pieces together inside a FasterRCNN model
# one class for fish, other for the backgroud
model = FasterRCNN(
    backbone,
    num_classes=2,
    rpn_anchor_generator=anchor_generator,
    box_roi_pool=roi_pooler,
    min_size=300, max_size=300
)

# See the model architecture
print(model)

# use our dataset and defined transformations
dataset = Dataset(DATA_DIR, transforms=get_transform(train=True))
dataset_val = Dataset(DATA_DIR, transforms=get_transform(train=False))

# split the dataset into train and validation sets
torch.manual_seed(1)
indices = torch.randperm(len(dataset)).tolist()

dataset_sub = torch.utils.data.Subset(dataset, indices[:-500])
dataset_val_sub = torch.utils.data.Subset(dataset_val, indices[-500:])

# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
    dataset_sub, batch_size=6, shuffle=True, num_workers=4, collate_fn=utils.collate_fn
)

data_loader_val = torch.utils.data.DataLoader(
    dataset_val_sub, batch_size=6, shuffle=False, num_workers=4, collate_fn=utils.collate_fn
)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)

model.to(device)

# define an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# and a learning rate scheduler which decreases the learning rate
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

num_epochs = 20

for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    epoch_loss = train_one_epoch(model, optimizer, data_loader,
                                    device, epoch, print_freq=10)
    # update the learning rate
    lr_scheduler.step()
    # evaluate on the validation dataset
    evaluator = evaluate(model, data_loader_val, dataset_val, device)

    torch.save(model, SAVE_MODEL)
