import cxr_dataset as CXR
import eval_model as E
import model as M
import torch
from torchsummary import summary

if __name__ == '__main__':
    # you will need to customize PATH_TO_IMAGES to where you have uncompressed
    # NIH images
    PATH_TO_IMAGES = r"E:/NIH"
    WEIGHT_DECAY = 1e-4
    LEARNING_RATE = 0.01
    # preds, aucs = M.train_cnn(PATH_TO_IMAGES, LEARNING_RATE, WEIGHT_DECAY)
    checkpoint_best = torch.load('checkpoint9.pt',map_location='cpu')
    model = checkpoint_best['model']
    LR = checkpoint_best['LR']
    # print(list(model.parameters())) model.state_dict()
    print("LR", LR) 
    print("model.classifier.in_features", model.classifier[1].in_features) 
    # summary(model, (3, 224, 224))

