from __future__ import absolute_import
import os.path as osp
from torch.utils.data import Dataset
import numpy as np
import random
from PIL import Image
import PIL
import cv2


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img



'''def patch(img, width, height, up=1, ran=1):
    #img.show()
    if up==1:
        high_ratio = random.uniform(0.20, 0.25)
        wide_ratio = random.uniform(0.35, 0.40)
    else:
        high_ratio = random.uniform(0.65, 0.75)
        wide_ratio = random.uniform(0.25, 0.30)
    
    split_high = int(high_ratio * height)
    split_wide = int(wide_ratio * width)
    
    high_len = int(random.uniform(0.15, 0.20) * height)
    wide_len = int(random.uniform(0.20, 0.25) * width)
    
    if up==1:
        loc = (split_wide, split_high, width-split_wide, split_high+high_len)
    else:
        loc = (split_wide, split_high, split_wide+wide_len, split_high+high_len)

    middle_part = img.crop(loc)
    
    middle_part = Color_aug(middle_part, ran=ran)
    middle_part = Texture_aug(middle_part, 0.25, ran=ran)
    
    return middle_part, loc'''










'''def Color_aug(img, ran=1):
    
    img = np.array(img, dtype=np.uint8).transpose(2, 0, 1)
    
    if ran==1:
        img[0] = random.random()*128
        img[1] = random.random()*128
        img[2] = random.random()*128

    else:
        img[:,:,0] = 64
        img[:,:,1] = 64
        img[:,:,2] = 64
        
    return Image.fromarray(np.array(img).transpose(1, 2, 0))'''





def position(w,h,v,ran=1):
    
    if ran==1:
        x0 = np.random.uniform(0, w)
        y0 = np.random.uniform(0, h)
    else:
        x0 = w/2
        y0 = h/2
    
    x0 = int(max(w/4., x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = int(min(3*w/4, x0 + v))
    y1 = int(min(h, y0 + v))
    
    xy = (x0, y0, x1, y1)
    
    if ran==1:
        color = (random.randint(0,64), random.randint(0,64), random.randint(0,64))
    else:
        color = (32, 32, 32)
    
    return xy, color


def Texture_aug(img, v, ran=1):
    
    #print(img.size)

    v = int(v * min(img.size))
    w, h = img.size
    img = img.copy()
    
    a = random.randint(1,3)
    for i in range(a):
        PIL.ImageDraw.Draw(img).rectangle(position(w,h,v,ran=ran)[0], position(w,h,v,ran=ran)[1])
    
    return img



def patch_back(img, width, height, back=1, ran=1):
    #img.show()
    if back==1:
        high_ratio = random.uniform(0.00, 0.05)
        wide_ratio = random.uniform(0.00, 0.10)
    elif back==2:
        high_ratio = random.uniform(0.00, 0.05)
        wide_ratio = random.uniform(0.75, 0.85)
    elif back==3:
        high_ratio = random.uniform(0.70, 0.75)
        wide_ratio = random.uniform(0.00, 0.05)
    elif back==4:
        high_ratio = random.uniform(0.70, 0.75)
        wide_ratio = random.uniform(0.85, 0.90)
    
    split_high = int(high_ratio * height)
    split_wide = int(wide_ratio * width)
    
    high_len = int(random.uniform(0.15, 0.25) * height)
    wide_len = int(random.uniform(0.15, 0.25) * width)
    

    loc = (split_wide, split_high, split_wide+wide_len, split_high+high_len)

    middle_part = img.crop(loc)
    
    #middle_part = Color_aug(middle_part, ran=ran)
    middle_part = Texture_aug(middle_part, 0.5, ran=ran)
    
    return middle_part, loc



def Text_Col(img, ran=1):

    width, height = img.size
    
    #up, up_loc = patch(img, width, height, up=1, ran=1)

    back1, back1_loc = patch_back(img, width, height, back=1, ran=1)
    back2, back2_loc = patch_back(img, width, height, back=2, ran=1)
    back3, back3_loc = patch_back(img, width, height, back=3, ran=1)
    back4, back4_loc = patch_back(img, width, height, back=4, ran=1)
    
    #img.paste(up, up_loc)
    img.paste(back1, back1_loc)
    img.paste(back2, back2_loc)
    img.paste(back3, back3_loc)
    img.paste(back4, back4_loc)
    
    return img



def weaken(image):
    
    aug = random.randint(6, 12)
    image = cv2.bilateralFilter(image, aug, 75, 75)
    #image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    
    return Image.fromarray(image)


def enhance(image):
    
    #kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    #image = cv2.filter2D(image, -1, kernel)
    blurred = cv2.GaussianBlur(image, (9, 9), 10.0)
    aug = random.random()
    image = cv2.addWeighted(image, 1+aug, blurred, -aug, 0)
    image = Image.fromarray(image)
    #image = Text_Col(image)
    
    return image


class Preprocessor_train(Dataset):
    def __init__(self, dataset, train = True, root=None, transform1=None,transform2=None):
        super(Preprocessor_train, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform1 = transform1
        self.transform2 = transform2
        self.train = train
        
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        fname, pid, camid = self.dataset[index]        
        #img_rand = read_image(fpath)
        
        #image = cv2.imread(fpath)
        #img_rand = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        #img = Image.open(fpath).convert('RGB')
        
        #img_rand = img.copy()
        if self.train:
            aug = random.randint(0, 2)
            if aug==0:
                fpath = fname
                img_rand = read_image(fpath)
            elif aug==1:
                fpath = osp.join(self.root, 'weaken', fname[fname.rfind("/")+1:])
                img_rand = read_image(fpath)
                #img_rand = weaken(img_rand)
                #img_rand = enhance(img_rand)
            elif aug==2:
                fpath = osp.join(self.root, 'enhance', fname[fname.rfind("/")+1:])
                img_rand = read_image(fpath)
                #img_rand = weaken(img_rand)
                #img_rand = enhance(img_rand)
                

        #img1 = self.transform1(img)
        img_rand = self.transform1(img_rand)
            
        
        return img_rand, fname, pid, camid, index



class Preprocessor_test(Dataset):
    def __init__(self, dataset, train = True, root=None, transform1=None,transform2=None):
        super(Preprocessor_test, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform1 = transform1
        self.transform2 = transform2
        self.train = train
        
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        fname, pid, camid = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)
        
        img = read_image(fpath)

        #img = Image.open(fpath).convert('RGB')
        
        #img_rand = img.copy()
                

        img1 = self.transform1(img)
        #img_rand = self.transform1(img_rand)
            
        
        return img1, fname, pid, camid, index