# app.py
import PIL
import torch
from matplotlib import pyplot as plt
from torchvision import transforms
from utils.models import MainModel
from utils.utils import lab_to_rgb
from fastai.vision.learner import create_body
from torchvision.models.resnet import resnet18
from fastai.vision.models.unet import DynamicUnet
from flask import Flask, render_template, request, redirect, url_for
import os
from PIL import Image
import numpy as np
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static'


def build_res_unet(n_input=1, n_output=2, size=256):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    body = create_body(resnet18(), pretrained=True, n_in=n_input, cut=-2)
    net_G = DynamicUnet(body, n_output, (size, size)).to(device)
    return net_G

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net_G = build_res_unet(n_input=1, n_output=2, size=256)
net_G.load_state_dict(torch.load("resnet-epoch/res18-unet.pt", map_location=device))
model = MainModel(net_G=net_G)
model.load_state_dict(torch.load("resnet-epoch/model_epoch_5.pt", map_location=device))


def colorize_image(image_path):
    # Load black and white image
    bw_image = Image.open(image_path).convert('L')
    bw_image = bw_image.resize((256, 256))  # Resize if needed

    img = transforms.ToTensor()(bw_image)[:1] * 2. - 1.
    
    model.eval()
    with torch.no_grad():
        preds = model.net_G(img.unsqueeze(0).to(device))
    
    colorized = lab_to_rgb(img.unsqueeze(0), preds.cpu())[0]

    return colorized

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/colorize', methods=['GET', 'POST'])
def colorize():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            input_filename = secure_filename(file.filename)
            input_file_path = os.path.join(app.config['UPLOAD_FOLDER'], input_filename)
            file.save(input_file_path)
            colorized_image = colorize_image(input_file_path)
            colorized_image_pil = Image.fromarray((colorized_image * 255).astype(np.uint8))
            colorized_filename = 'colorized_' + input_filename
            colorized_file_path = os.path.join(app.config['UPLOAD_FOLDER'], colorized_filename)
            colorized_image_pil.save(colorized_file_path)
            
            return render_template('displaycolorized_image.html', 
                                   original_image=input_filename,
                                   colorized_image=colorized_filename)
    return render_template('colorize.html')



if __name__ == '__main__':
    app.run(debug=True)
