import torch
import matplotlib.pyplot as plt
import io
from torchvision import transforms as trans

def l2_norm(input,axis=1):
    norm = torch.norm(input,2,axis,True)
    output = torch.div(input, norm)
    return output

def gen_plot(fpr, tpr, save_path):
    """Create a pyplot plot and save to buffer."""
    plt.figure()
    plt.xlabel("FPR", fontsize=14)
    plt.ylabel("TPR", fontsize=14)
    plt.title("ROC Curve", fontsize=14)
    plot = plt.plot(fpr, tpr, linewidth=2)
    plt.savefig(save_path)
    plt.close()

def hflip_batch(imgs_tensor):
    def de_preprocess(tensor):
        return tensor*0.5 + 0.5
    hflip = trans.Compose([
                de_preprocess,
                trans.ToPILImage(),
                trans.functional.hflip,
                trans.ToTensor(),
                trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
    hfliped_imgs = torch.empty_like(imgs_tensor)
    for i, img_ten in enumerate(imgs_tensor):
        hfliped_imgs[i] = hflip(img_ten)
    return hfliped_imgs

