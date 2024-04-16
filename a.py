import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision
import torchvision.transforms as T

import torch.profiler
from torchvision import models

model = models.resnet50(pretrained=False)
model.cuda()
cudnn.benchmark = True

class D(torch.utils.data.Dataset):
    def __init__(self) -> None:
        super().__init__()

    def __len__(self):
        return 1000
    
    def __getitem__(self, index):
        input = torch.randn(size=(3, 224, 224))
        label = 0
        return input, label

# transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
# trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
#                                         download=True, transform=transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
#                                           shuffle=True, num_workers=4)
trainloader = torch.utils.data.DataLoader(D(), batch_size=32,
                                          shuffle=True, num_workers=4)
criterion = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
device = torch.device("cuda:0")
model.train()

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(
        wait=1,
        warmup=1,
        active=2),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./result', worker_name='worker0'),
    record_shapes=True,
    profile_memory=True,  # This will take 1 to 2 minutes. Setting it to False could greatly speedup.
    with_stack=True
) as p:
    for step, data in enumerate(trainloader, 0):
        print("step:{}".format(step))
        inputs, labels = data[0].to(device=device), data[1].to(device=device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step + 1 >= 4:
            break
        p.step()