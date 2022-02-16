from pytorch_diffusion import Diffusion
import torchvision
import torchattacks
from Cifar10_models import ResNet18, model_with_diffusion
import torch
from torchvision import transforms
from torch import nn, einsum
import copy

train = 0
if train:
    model = ResNet18()
    model = model.cuda()

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.Resize(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    trainset = torchvision.datasets.CIFAR10(
        root='~/data', train=True, download=True, transform=transform_train)

    transform_test = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor()
    ])
    testset = torchvision.datasets.CIFAR10(
        root='~/data', train=False, download=True, transform=transform_test)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=True, pin_memory=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(100):
        running_loss = 0
        train_accuracy = 0
        test_accuracy = 0

        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()

            outputs = model(inputs)

            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            max_vals, max_indices = torch.max(outputs, 1)
            correct = (max_indices == labels).sum().data.cpu().numpy() / max_indices.size()[0]
            train_accuracy += 100 * correct

        running_loss /= len(train_loader)
        train_accuracy /= len(train_loader)
        model.eval()

        for i, data in enumerate(test_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)
            max_vals, max_indices = torch.max(outputs, 1)
            correct = (max_indices == labels).sum().data.cpu().numpy() / max_indices.size()[0]
            test_accuracy += 100 * correct

        test_accuracy /= len(test_loader)

        print(epoch)
        print(running_loss, train_accuracy, test_accuracy)

        data = {
            'model': model.state_dict()
        }
        torch.save(data, f'model-prediction.pt')


model = ResNet18()
model = model.cuda()

data = torch.load(f'model-prediction.pt')
model.load_state_dict(data['model'])
model.eval()

transform_test = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor()
])
testset = torchvision.datasets.CIFAR10(
    root='~/data', train=False, download=True, transform=transform_test)


steps = 2
test_loader = torch.utils.data.DataLoader(testset, batch_size=36, shuffle=True, pin_memory=True)
diffusion = Diffusion.from_pretrained("ema_cifar10")
defense = model_with_diffusion(model=model, diffusion_model=diffusion, steps=steps)
defense_2 = model_with_diffusion(model=model, diffusion_model=diffusion, steps=10, with_grad=False)
defense = defense.eval()
defense_2 = defense_2.eval()
#defense = copy.deepcopy(model)
atk = torchattacks.CW(defense, c=5)


orig_acc = 0
adv_acc = 0
def2_adv_acc = 0
cnt = 0

for i, data in enumerate(test_loader, 0):
    inputs, labels = data
    inputs, labels = inputs.cuda(), labels.cuda()

    outputs = defense(inputs)

    max_vals, max_indices = torch.max(outputs, 1)
    correct = (max_indices == labels).sum().data.cpu().numpy() / max_indices.size()[0]
    orig_acc += 100 * correct
    if i == 0:
        print("Original Prediction")
        print(correct)
        torchvision.utils.save_image(inputs, f'original.png', nrow=6)

    adv_images = atk(inputs, labels)
    outputs = defense(adv_images)
    max_vals, max_indices = torch.max(outputs, 1)
    correct = (max_indices == labels).sum().data.cpu().numpy() / max_indices.size()[0]
    adv_acc += 100 * correct
    if i == 0:
        print("Adversarial Prediction")
        print(correct)
        torchvision.utils.save_image(inputs, f'adversarial.png', nrow=6)

    outputs = defense_2(adv_images)
    max_vals, max_indices = torch.max(outputs, 1)
    correct = (max_indices == labels).sum().data.cpu().numpy() / max_indices.size()[0]
    def2_adv_acc += 100 * correct

    cnt += 1
    print("Count")
    print(cnt)

    print("Original Accuracy")
    print(orig_acc / cnt)

    print("Adversarial Accuracy")
    print(adv_acc / cnt)

    print("Defend Adversarial Accuracy")
    print(def2_adv_acc / cnt)