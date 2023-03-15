import sys
sys.path.append('../')
import os
from clip import model as clipmodel
from clip import clip
from tqdm.auto import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
from sklearn.linear_model import LogisticRegression
from torch.utils.data.dataloader import DataLoader
from PIL import Image
import random

device = "cuda:0" if torch.cuda.is_available() else "cpu" 

def build_model():
    model, preprocess = clip.load("ViT-B/16",device=device,jit=False)
    state_dict = model.state_dict()
    vit = "visual.proj" in state_dict
    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))
    model = clipmodel.CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
    ) 
    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    clipmodel.convert_weights(model)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, preprocess

class image_title_dataset(Dataset):
    def __init__(self, files, name_mapper, preprocess):
        self.files = files
        self.name_mapper = name_mapper
        self.preprocess = preprocess

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image = self.preprocess(Image.open(self.files[idx][0])) # Image from PIL module
        title = self.name_mapper[self.files[idx][1]]
        return image,title

def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]

def zeroshot(dataset, text_inputs, label_mapper, model):
    test_dl = DataLoader(dataset,batch_size=64,num_workers=2,pin_memory=True,shuffle=True)
    with torch.no_grad():
        top1, n = 0., 0.
        for batch in tqdm(test_dl):  
            images,label = batch       

            images= images.to(device)
            texts = text_inputs
            target = torch.Tensor([label_mapper[l] for l in label]).to(device)

            image_features = model.encode_image(images)
            text_features = model.encode_text(texts)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            #values, indices = similarity[0].topk(5)
            acc1, acc5 = accuracy(similarity, target, topk=(1, 5))
            top1 += acc1
            n += images.size(0)

    top1 = (top1 / n) * 100

    print(f"Top-1 accuracy: {top1:.2f}")
    return top1

def get_features(model, test_dl, label_mapper, full=False):
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_dl): 
            images,label = batch
            features = model.encode_image(images.to(device))

            all_features.append(features)
            all_labels.append(torch.Tensor([label_mapper[l] for l in label]).to(device))
    return torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy()
    
def fewshot(shot_count, dataset, few_shot_dataset, text_inputs, label_mapper, model):
    # few shot results
    print(shot_count, '- shot training')
    train_dl = DataLoader(few_shot_dataset,batch_size=64,num_workers=2,pin_memory=True,shuffle=True)
    train_features, train_labels = get_features(train_dl, label_mapper)
    # all data
    print(shot_count, '- shot evaluation')
    all_dl = DataLoader(dataset,batch_size=64,num_workers=2,pin_memory=True,shuffle=True)
    test_features, test_labels = get_features(all_dl, label_mapper)
    
    # Perform logistic regression
    classifier = LogisticRegression(random_state=0, C=0.316, max_iter=1000)
    classifier.fit(train_features, train_labels)

    # Evaluate using the logistic regression classifier
    predictions = classifier.predict(test_features)
    top1 = np.mean((test_labels == predictions).astype(float)) * 100.
    return top1
    
    
def eval_acc(shot_count, dataset, text_inputs, label_mapper, model, few_shot_dataset=None):
    if shot_count == 0: # zero shot
        top1 = zeroshot(dataset, text_inputs, label_mapper, model)
    else: # few shot
        top1 = fewshot(shot_count, dataset, few_shot_dataset, text_inputs, label_mapper, model)
    return top1

def eurosat(model, preprocess, root):
    image_path_list = []
    image_class = []
    for folder in os.listdir(root):
        image_class.append(folder)
        path = os.path.join(root, folder)
        for file in os.listdir(path):
            img_path = os.path.join(path,file)
            image_path_list.append((img_path, folder))
    name_mapper = {
        'Forest': 'forest',
        'PermanentCrop': 'permanent crop land',
        'Residential': 'residential buildings or homes or apartments',
        'River': 'river',
        'Pasture': 'pasture land',
        'SeaLake': 'lake or sea',
        'HerbaceousVegetation': 'brushland or shrubland',
        'AnnualCrop': 'annual crop land',
        'Industrial': 'industrial buildings or commercial buildings',
        'Highway': 'highway or road'
    }
    image_class = [name_mapper[i] for i in image_class]
    label_mapper = dict(zip(image_class, range(len(image_class))))
    dataset = image_title_dataset(image_path_list, name_mapper, preprocess)
    ESAT_text = torch.cat([clip.tokenize(f"a satellite photo of {c}, a type of surface feature or land use") for c in image_class]).to(device)
    zero_shot_acc = eval_acc(0, dataset, ESAT_text, label_mapper, model)
    shots = [1,2,4,8,16]
    acc_list = []
    for shot_count in shots:
        # each class
        image_path_list = []
        for folder in os.listdir(root):
            path = os.path.join(root, folder)
            # select few shot per class
            few_shot_data = random.sample(os.listdir(path), shot_count)
            for file in few_shot_data:
                img_path = os.path.join(path,file)
                image_path_list.append((img_path, folder))
        few_shot_dataset = image_title_dataset(image_path_list, name_mapper)
        top1_acc = eval_acc(shot_count, dataset, ESAT_text, label_mapper, model, few_shot_dataset)
        print(f"Top-1 accuracy: {top1_acc:.2f}")
        acc_list.append(top1_acc)
    return zero_shot_acc, acc_list

def fgvcaircraft(model, preprocess, root):
    image_path_list = []
    with open(os.path.join(root, 'images_variant_train.txt')) as f:
        for line in f:
            splt = line.strip().split(' ', 1)
            splt[0] = os.path.join(root, 'images', splt[0] + '.jpg')
            image_path_list.append(tuple(splt))
    with open(os.path.join(root, 'variants.txt')) as f:
        image_class = [line.strip() for line in f]
    label_mapper = dict(zip(image_class, range(len(image_class))))
    name_mapper = {k:k for k in label_mapper}
    dataset = image_title_dataset(image_path_list, name_mapper, preprocess)
    FGVC_text = torch.cat([clip.tokenize(f"a photo of the {c}, a type of aircraft.") for c in image_class]).to(device)
    zero_shot_acc = eval_acc(0, dataset, FGVC_text, label_mapper, model)
    shots = [1,2,4,8,16]
    acc_list = []
    all_classes = np.array([x[1] for x in image_path_list])
    for shot_count in shots:
        image_path_list_2 = []
        for c in image_class:
            ix = np.random.choice(np.where(all_classes == c)[0], shot_count, replace=False)
            for img in [image_path_list[i] for i in ix]:
                image_path_list_2.append(img)
        few_shot_dataset = image_title_dataset(image_path_list_2, name_mapper)
        top1_acc = eval_acc(shot_count, dataset, FGVC_text, label_mapper, model, few_shot_dataset)
        print(f"Top-1 accuracy: {top1_acc:.2f}")
    acc_list.append(top1_acc)
    return zero_shot_acc, acc_list

def dtd(model, preprocess, root):
    image_path_list = []
    image_class = []
    for folder in os.listdir(root):
        image_class.append(folder)
        path = os.path.join(root, folder)
        for file in os.listdir(path):
            img_path = os.path.join(path,file)
            image_path_list.append((img_path, folder))
    label_mapper = dict(zip(image_class, range(len(image_class))))
    name_mapper = {k:k for k in label_mapper}
    dataset = image_title_dataset(image_path_list, name_mapper, preprocess)
    DTD_text = torch.cat([clip.tokenize(f"a photo of a {c} texture.") for c in image_class]).to(device)
    zero_shot_acc = eval_acc(0, dataset, DTD_text, label_mapper, model)
    shots = [1,2,4,8,16]
    acc_list = []
    for shot_count in shots:
        # each class
        image_path_list = []
        for folder in os.listdir(root):
            path = os.path.join(root, folder)
            # select few shot per class
            few_shot_data = random.sample(os.listdir(path), shot_count)
            for file in few_shot_data:
                img_path = os.path.join(path,file)
                image_path_list.append((img_path, folder))
        few_shot_dataset = image_title_dataset(image_path_list, name_mapper)
        top1_acc = eval_acc(shot_count, dataset, DTD_text, label_mapper, model, few_shot_dataset)
        print(f"Top-1 accuracy: {top1_acc:.2f}")
        acc_list.append(top1_acc)
    return zero_shot_acc, acc_list

def caltech101(model, preprocess, root):
    name_mapper = {
    'BACKGROUND_Google': 'background',
    'Faces': 'off-center face',
    'Faces_easy': 'centered face',
    'Leopards': 'leopard',
    'Motorbikes': 'motorbike',
    'accordion': 'accordion',
    'airplanes': 'airplane',
    'anchor': 'anchor',
    'ant': 'ant',
    'barrel': 'barrel',
    'bass': 'bass',
    'beaver': 'beaver',
    'binocular': 'binocular',
    'bonsai': 'bonsai',
    'brain': 'brain',
    'brontosaurus': 'brontosaurus',
    'buddha': 'buddha',
    'butterfly': 'butterfly',
    'camera': 'camera',
    'cannon': 'cannon',
    'car_side': 'side of a car',
    'ceiling_fan': 'ceiling fan',
    'cellphone': 'cellphone',
    'chair': 'chair',
    'chandelier': 'chandelier',
    'cougar_body': 'body of a cougar cat',
    'cougar_face': 'face of a cougar cat',
    'crab': 'crab',
    'crayfish': 'crayfish',
    'crocodile': 'crocodile',
    'crocodile_head': 'head of a  crocodile',
    'cup': 'cup',
    'dalmatian': 'dalmatian',
    'dollar_bill': 'dollar bill',
    'dolphin': 'dolphin',
    'dragonfly': 'dragonfly',
    'electric_guitar': 'electric guitar',
    'elephant': 'elephant',
    'emu': 'emu',
    'euphonium': 'euphonium',
    'ewer': 'ewer',
    'ferry': 'ferry',
    'flamingo': 'flamingo',
    'flamingo_head': 'head of a flamingo',
    'garfield': 'garfield',
    'gerenuk': 'gerenuk',
    'gramophone': 'gramophone',
    'grand_piano': 'grand piano',
    'hawksbill': 'hawksbill',
    'headphone': 'headphone',
    'hedgehog': 'hedgehog',
    'helicopter': 'helicopter',
    'ibis': 'ibis',
    'inline_skate': 'inline skate',
    'joshua_tree': 'joshua tree',
    'kangaroo': 'kangaroo',
    'ketch': 'ketch',
    'lamp': 'lamp',
    'laptop': 'laptop',
    'llama': 'llama',
    'lobster': 'lobster',
    'lotus': 'lotus',
    'mandolin': 'mandolin',
    'mayfly': 'mayfly',
    'menorah': 'menorah',
    'metronome': 'metronome',
    'minaret': 'minaret',
    'nautilus': 'nautilus',
    'octopus':'octopus',
    'okapi':'okapi',
    'pagoda':'pagoda',
    'panda':'panda',
    'pigeon':'pigeon',
    'pizza':'pizza',
    'platypus':'platypus',
    'pyramid':'pyramid',
    'revolver':'revolver',
    'rhino':'rhino',
    'rooster':'rooster',
    'saxophone':'saxophone',
    'schooner':'schooner',
    'scissors':'scissors',
    'scorpion':'scorpion',
    'sea_horse':'sea horse',
    'snoopy': 'snoopy (cartoon beagle)',
    'soccer_ball': 'soccer ball',
    'stapler':'stapler',
    'starfish':'starfish',
    'stegosaurus':'stegosaurus',
    'stop_sign': 'stop sign',
    'strawberry':'strawberry',
    'sunflower':'sunflower',
    'tick':'tick',
    'trilobite':'trilobite',
    'umbrella':'umbrella',
    'watch':'watch',
    'water_lilly':'water lilly',
    'wheelchair':'wheelchair',
    'wild_cat':'wild cat',
    'windsor_chair':'windsor chair',
    'wrench':'wrench',
    'yin_yang':'yin and yang symbol',
}
    image_path_list = []
    image_class = []
    for folder in os.listdir(root):
        image_class.append(folder)
        path = os.path.join(root, folder)
        for file in os.listdir(path):
            img_path = os.path.join(path,file)
            image_path_list.append((img_path, folder))
    image_class = [name_mapper[i] for i in image_class]
    label_mapper = dict(zip(image_class, range(len(image_class))))
    dataset = image_title_dataset(image_path_list, name_mapper, preprocess)
    CAL_text = torch.cat([clip.tokenize(f'a photo of a {c}.') for c in image_class]).to(device)
    zero_shot_acc = eval_acc(0, dataset, CAL_text, label_mapper, model)
    shots = [1,2,4,8,16]
    acc_list = []
    for shot_count in shots:
        # each class
        image_path_list = []
        for folder in os.listdir(root):
            path = os.path.join(root, folder)
            # select few shot per class
            few_shot_data = random.sample(os.listdir(path), shot_count)
            for file in few_shot_data:
                img_path = os.path.join(path,file)
                image_path_list.append((img_path, folder))
        few_shot_dataset = image_title_dataset(image_path_list, name_mapper)
        top1_acc = eval_acc(shot_count, dataset, CAL_text, label_mapper, model, few_shot_dataset)
        print(f"Top-1 accuracy: {top1_acc:.2f}")
        acc_list.append(top1_acc)
    return zero_shot_acc, acc_list

def oxfordpets(model, preprocess, root):
    image_path_list = []
    image_class = []
    for file in os.listdir(root):
        clss = file.rsplit('_',1)[0].replace('_', ' ')
        image_class.append(clss)
        img_path = os.path.join(root, file)
        image_path_list.append((img_path, clss))
    image_class = list(set(image_class))
    label_mapper = dict(zip(image_class, range(len(image_class))))
    name_mapper = {k:k for k in label_mapper}
    dataset = image_title_dataset(image_path_list, name_mapper, preprocess)
    PET_text = torch.cat([clip.tokenize(f'a photo of a {c}, a type of pet.') for c in image_class]).to(device)
    zero_shot_acc = eval_acc(0, dataset, PET_text, label_mapper, model)
    shots = [1,2,4,8,16]
    acc_list = []
    all_classes = np.array([x[1] for x in image_path_list])
    for shot_count in shots:
        image_path_list_2 = []
        for c in image_class:
            ix = np.random.choice(np.where(all_classes == c)[0], shot_count, replace=False)
            for img in [image_path_list[i] for i in ix]:
                image_path_list_2.append(img)
        few_shot_dataset = image_title_dataset(image_path_list_2, name_mapper)
        top1_acc = eval_acc(shot_count, dataset, PET_text, label_mapper, model, few_shot_dataset)
        print(f"Top-1 accuracy: {top1_acc:.2f}")
        acc_list.append(top1_acc)
    return zero_shot_acc, acc_list