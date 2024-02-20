#https://github.com/princetonvisualai/OverlookedFactors/blob/master/get_features.py



import pickle, time
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
from clip import clip

import numpy as np
import os
import tqdm

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
dtype = torch.float32

global places_model, places_model_base, clip_model, clip_preprocess

center_crop = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

def load_models(fldr):
    global places_model, places_model_base, clip_model, clip_preprocess
    
    arch = 'resnet18'
    model_file = f'{fldr}/{arch}_places365.pth.tar' 
    places_model = torchvision.models.__dict__[arch](num_classes=365)
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
    places_model.load_state_dict(state_dict)
    places_model.eval()

    places_model_base = torch.nn.Sequential( *list(places_model.children())[:-1])
    places_model_base.eval()

    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

def dump_features(dataset_name, data_fldr):
    dataset_fldr = f"{data_fldr}/{dataset_name}"
    if not os.path.exists(dataset_fldr):
        os.mkdir(dataset_fldr)

    A = pickle.load(open(f'{data_fldr}/{dataset_name}_imagelabels.pkl', 'rb'))
    for name, m in [('features', places_model_base), ('logits', places_model), ('clip', clip_model)]:
        with torch.no_grad():
            m = m.to(device)

            for split in ['train', 'val', 'test']:
                count = 0; start_time = time.time()
                img_to_scene, img_to_feature = {}, {}

                img_names = A[split]
                for img_name in tqdm.tqdm(img_names, desc=f"{dataset_name}-{name}"): 
                    img_path = img_name
                    img = Image.open(img_path).convert('RGB')

                    if name == 'clip':
                        clip_batch = clip_preprocess(img).to(device).unsqueeze(dim=0)
                        clip_features = clip_model.encode_image(clip_batch)
                        img_to_feature[img_path] = clip_features.detach().cpu().numpy().squeeze()

                    else:
                        img = center_crop(img)
                        img = img.to(device=device, dtype = dtype)
                        img = img.unsqueeze(0)

                        sc = m(img)
                        img_to_feature[img_path] = sc.detach().cpu().numpy().squeeze()

                        if name == 'logits':
                            h_x = torch.nn.functional.softmax(sc, 1).data.squeeze()
                            probs, idx = h_x.sort(0, True)

                            scene = int(idx[0].data.cpu().numpy())
                            img_to_scene[img_path] = scene               

                    count += 1
                    if count % 1000 == 0: 
                        elapsed_time = (time.time() - start_time)/60
                        print('Processed {} images in {:.2f} minutes'.format(count, elapsed_time), flush=True)

                with open(f'{dataset_fldr}/' + '{}_{}.pkl'.format(split, name), 'wb+') as f:
                    pickle.dump(img_to_feature, f)
                if name == 'logits':
                    with open('{}/{}_scene.pkl'.format(dataset_fldr, split), 'wb+') as f:
                        pickle.dump(img_to_scene, f)
                        
    os.system(f"cp {data_fldr}/{dataset_name}_imagelabels.pkl {dataset_fldr}/")
                

if __name__ == '__main__':
    fldr = '/scratch/vp421/data'
    if not os.path.exists(f"{fldr}/resnet18_places365.pth.tar"):
        os.system(f"curl -o {fldr}/resnet18_places365.pth.tar http://places2.csail.mit.edu/models_places365/resnet18_places365.pth.tar")

    load_models(fldr)
    data_fldr = f"{fldr}"
    
    dump_features('ade20k', data_fldr)
    dump_features('pascal', data_fldr)