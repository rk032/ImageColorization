import PIL
import torch
from matplotlib import pyplot as plt
from torchvision import transforms
from utils.models import MainModel
from utils.utils import lab_to_rgb
from fastai.vision.learner import create_body
from torchvision.models.resnet import resnet18
from fastai.vision.models.unet import DynamicUnet

def build_res_unet(n_input=1, n_output=2, size=256):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    body = create_body(resnet18(), pretrained=True, n_in=n_input, cut=-2)
    net_G = DynamicUnet(body, n_output, (size, size)).to(device)
    return net_G

if __name__ == '__main__':
    #for testing the baseline model
    '''model = MainModel()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load and inspect the state dict without loading it into the model
    state_dict = torch.load("final_epoch/model_epoch_5.pt", map_location=device)
    model.load_state_dict(state_dict)'''
    
    #for resting using resnet model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net_G = build_res_unet(n_input=1, n_output=2, size=256)
    net_G.load_state_dict(torch.load("resnet-epoch/res18-unet.pt", map_location=device))
    model = MainModel(net_G=net_G)
    model.load_state_dict(torch.load("resnet-epoch/model_epoch_5.pt", map_location=device))

    
    path = "test_data/image.jpg"
    img = PIL.Image.open(path)
    plt.imshow(img)
    
    img = img.resize((256, 256))
    img = transforms.ToTensor()(img)[:1] * 2. - 1.
    
    model.eval()
    with torch.no_grad():
        preds = model.net_G(img.unsqueeze(0).to(device))
    
    colorized = lab_to_rgb(img.unsqueeze(0), preds.cpu())[0]
    plt.imshow(colorized)
    plt.savefig('Output/output.jpeg')    
    plt.show()
