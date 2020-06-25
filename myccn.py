import pycocotools

import os

import torch
import torch.utils.data
from PIL import Image, ImageDraw
import pandas as pd
import numpy as np
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision as tv
from engine import train_one_epoch, evaluate
import utils
import transforms as T


def parse_one_annot(path_to_data_file, filename):
    data = pd.read_csv(path_to_data_file)
    boxes_array = data[data["filename"] == filename][["xmin", "ymin", "xmax", "ymax"]].values

    return boxes_array


class RaccoonDataSet(torch.utils.data.Dataset):
    def __init__(self, root, data_file, transforms=None):
        self.root = root
        self.transforms = transforms
        self.imgs = sorted(os.listdir(os.path.join(root, "images")))
        self.path_tp_data_file = data_file

    def __getitem__(self, idx):
        #   load images and bounding boxes
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
        box_list = parse_one_annot(self.path_tp_data_file, self.imgs[idx])
        boxes = torch.as_tensor(box_list, dtype=torch.float32)

        num_objs = len(box_list)

        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # supposed all instances are not crowd

        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        target = {"boxes": boxes, "labels": labels, "image_id": image_id, "area": area, "iscrowd": iscrowd}
        print(labels)
        print(image_id)
        print(area)
        print(iscrowd)
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target

    def __len__(self):
        return len(self.imgs)


def get_model(num_classes):
    # load an object detection model pre-trained on COCO
    model = tv.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # get the number of input features for the classifier

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new on
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def get_transform(train):
    transforms = []
    # converts the image .a PIL image, into a pytorch tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


# use our dataset and defined transformations


dataset = RaccoonDataSet(root="/content/pytorchCNN/connection_dataset",
                         data_file="/content/pytorchCNN/connection_dataset/data/connection_annotations.csv",
                         transforms=get_transform(train=True))

dataset_test = RaccoonDataSet(root="/content/pytorchCNN/connection_dataset",
                              data_file="/content/pytorchCNN/connection_dataset/data/connection_annotations.csv",
                              transforms=get_transform(train=False))

# split the dataset in train and test set

torch.manual_seed
indices = torch.randperm(len(dataset)).tolist()
dataset = torch.utils.data.Subset(dataset, indices[:-40])

batch_size = 64
# yolov5
nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers

# yolov5
# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True, collate_fn=utils.collate_fn
)

data_loader_test = torch.utils.data.DataLoader(
    dataset, batch_size=64, shuffle=False, num_workers=0,
    collate_fn=utils.collate_fn
)
print("We have: {} examples, {} are training and {} testing".format(len(indices), len(dataset), len(dataset_test)))

if torch.cuda.is_available():
    print('WORKING ON GPU ! \n')
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# out dataset has two classes only - raccon and not raccon

num_classes = 2

# get the model using our helper function

model = get_model(num_classes)
# move model to the right device
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)

# and a learning rate sheduler which decreases the learning rate by 10x every3 epochs

lrscheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                              step_size=3,
                                              gamma=0.1)

# lets train for 10 epochs

num_epochs = 5
for epoch in range(num_epochs):
    # train for one epoch , printing every 10 iterations
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
    # update learning rate
    lrscheduler.step()
# evaluate in the test dataset
evaluate(model, data_loader_test, device=device)

os.mkdir("/content/pytorchCNN/pytorch_detection/raccoon/")
torch.save(model.state_dict(), "/content/pytorchCNN/pytorch_detection/raccoon/model")
# print(dataset.__getitem__(0))

# create empty mode and load previously trained model
loaded_model = get_model(num_classes=2)

loaded_model.load_state_dict(torch.load("/content/pytorchCNN/pytorch_detection/raccoon/model"))

idx = 0

img, _ = dataset_test[idx]
label_boxes = np.array((dataset_test[idx][1]["boxes"]))

# put the model in evaluation mode
loaded_model.eval()

with torch.no_grad():
    prediction = loaded_model([img])
image = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())
draw = ImageDraw.Draw(image)

# draw groundthuth

for elem in range(len(label_boxes)):
    draw.rectangle([(label_boxes[elem][0], label_boxes[elem][1]),
                    (label_boxes[elem][2], label_boxes[elem][3])],
                   outline="green", width=3)
for element in range(len(prediction[0]["boxes"])):
    boxes = prediction[0]["boxes"][element].cpu().numpy()
    score = np.round(prediction[0]["scores"][element].cpu().numpy(),
                     decimals=4)
    if score > 0.5:
        draw.rectangle([(boxes[0], boxes[1]), (boxes[2], boxes[3])],
                       outline="red", width=3)
        draw.text((boxes[0], boxes[1]), text=str(score))
image.save("test.jpg")
