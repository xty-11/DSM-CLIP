import torch
import torch.nn as nn
import numpy as np
from .clip import clip
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F



def load_clip_to_cpu(backbone_name, h_resolution, w_resolution, vision_stride_size):
    # 
    url = clip._MODELS[backbone_name]
    try:
        model_path = clip._download(url)
    except:
        model_path = '' # ViT-B-16.pt
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict(), h_resolution, w_resolution, vision_stride_size)

    return model


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding #(77, 512)
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection   #(512, 512)
        self.dtype = clip_model.dtype
        
    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)   #(128, 77, 512)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection  #tokenized_prompts.argmax(dim=-1)是最大下标，最后一个字符的下标11
        return x    #（128， 512） [128, 11, 512]@[512, 512]->[128,512]


class DomainClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.bn1 = nn.BatchNorm1d(hidden_size)

    def forward(self, x):

        out = self.fc1(x)
        if x.shape[0] == 1:
            out = out.repeat(2, 1)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.fc2(out)
            return F.log_softmax(out, dim=1)[0]
        else:
            out = self.bn1(out)
            out = self.relu(out)
            out = self.fc2(out)
            return F.log_softmax(out, dim=1)


class GradReverse(torch.autograd.Function):
    def __init__(self):
        super(GradReverse, self).__init__()

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.save_for_backward(lambda_)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        lambda_, = ctx.saved_tensors
        grad_input = grad_output.clone()
        return - lambda_ * grad_input, None


class GRL(torch.nn.Module):
    def __init__(self, lambd=.1):
        super(GRL, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        lam = torch.tensor(self.lambd)
        return GradReverse.apply(x, lam)


class PromptLearner(nn.Module):
    def __init__(self, num_class, dataset_num, dtype, token_embedding):
        super().__init__()
        ctx_init = "A photo of a X X X X person."
        ctx_dim = 512   
        n_ctx = 4    

        tokenized_prompts = clip.tokenize(ctx_init).cuda()

        with torch.no_grad():
            embedding = token_embedding(tokenized_prompts).type(dtype)  #(1, 77, 512)   

        self.tokenized_prompts = tokenized_prompts

        n_cls_ctx = 4
        cls_vectors = torch.empty(num_class, n_cls_ctx, ctx_dim, dtype=dtype) 
        nn.init.normal_(cls_vectors, std=0.02)
        


        self.clsctx = nn.Parameter(cls_vectors, requires_grad=True)   

        self.register_buffer("token_prefix", embedding[:, :n_ctx + 1, :])   # （1，0~4, 512）
        self.register_buffer("token_suffix", embedding[:, n_ctx + 1 + n_cls_ctx:, :])  # (1,9~77,512)


        self.num_class = num_class
        self.n_cls_ctx = n_cls_ctx
    def forward(self, label):
        cls_ctx = self.clsctx[label]    #(128)->(128, 4, 512)
        b = label.shape[0]

        prefix = self.token_prefix.expand(b, -1, -1)
        suffix = self.token_suffix.expand(b, -1, -1)

        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)   (128, 5, 512)
                cls_ctx,  # (n_cls, n_ctx, dim)  #(128, 4, 512)
                suffix,  # (n_cls, *, dim)      #(128, 68, 512)
            ],
            dim=1,
        )   #(128, 77, 512)
        B = len(cls_ctx)
        return prompts, cls_ctx.reshape(B,-1)




class Model(nn.Module):
    def __init__(self, num_classes, args, epsilon=.1, domain_num=4):
        super(Model, self).__init__()
        self.h_resolution = int((args.size_train[0] - 16) // 16 + 1)
        self.w_resolution = int((args.size_train[1] - 16) // 16 + 1)
        self.vision_stride_size = 16
        self.model_name = args.backbone
        self.neck_feat = 'before'
        if self.model_name == 'ViT-B-16':
            self.in_planes = 768
            self.in_planes_proj = 512
        if self.model_name == 'ViT-B-32':
            self.in_planes = 768
            self.in_planes_proj = 512
            self.h_resolution = int((args.size_train[0] - 32) // 32 + 1)
            self.w_resolution = int((args.size_train[1] - 32) // 32 + 1)
            self.vision_stride_size = 32
        if self.model_name == 'ViT-L-14':
            self.in_planes = 768
            self.in_planes_proj = 512
            self.h_resolution = int((args.size_train[0] - 14) // 14 + 1)
            self.w_resolution = int((args.size_train[1] - 14) // 14 + 1)
            self.vision_stride_size = 14
        elif self.model_name == 'RN50':
            self.in_planes = 2048
            self.in_planes_proj = 1024
        elif self.model_name == 'RN101':
            self.in_planes = 2048
            self.in_planes_proj = 512
        self.num_classes = num_classes
        self.grl = GRL(epsilon)
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        self.classifier_proj = nn.Linear(self.in_planes_proj, self.num_classes, bias=False)
        self.classifier_proj.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.bottleneck_proj = nn.BatchNorm1d(self.in_planes_proj)
        self.bottleneck_proj.bias.requires_grad_(False)
        self.bottleneck_proj.apply(weights_init_kaiming)

        clip_model = load_clip_to_cpu(self.model_name, self.h_resolution, self.w_resolution, self.vision_stride_size)
        clip_model.to("cuda")
        self.image_encoder = clip_model.visual
        #text_encoder
        self.promptlearner = PromptLearner(num_classes, domain_num, clip_model.dtype, clip_model.token_embedding)

        self.text_encoder = TextEncoder(clip_model)

    def forward(self, x=None, label=None, get_image=False, get_text=False, cam_label=None, view_label=None,
                domain=None, prior=False, getcls=False):
        if get_text: # get text features
            prompts, cls = self.promptlearner(label)   #(128, 77, 512) , (128, 2048(4*512))
            if getcls:
                return cls
            text_features = self.text_encoder(prompts, self.promptlearner.tokenized_prompts)#(128, 77, 512)
            return text_features

        if get_image == True: # get image features
            image_features, image_features_proj = self.image_encoder(x)
            if "RN" in self.model_name:
                return image_features_proj[0]
            elif "ViT" in self.model_name:
                return image_features_proj[:, 0]   #(128, 1, 768)

        if "RN" in self.model_name:
            image_features, image_features_proj = self.image_encoder(x)
            img_feature = nn.functional.avg_pool2d(image_features, image_features.shape[2:4]).view(x.shape[0], -1)
            img_feature_proj = image_features_proj[0]


        elif "ViT" in self.model_name:
            cv_embed = None
            image_features, image_features_proj = self.image_encoder(x, cv_embed)  # (11, 12, 12proj)
            img_feature = image_features[:, 0]          # cls_token
            img_feature_proj = image_features_proj[:, 0]   # （128, 197, 512）->(128, 512)

        feat = self.bottleneck(img_feature)     # bn1d
        feat_proj = self.bottleneck_proj(img_feature_proj) #bn1d

        if self.training: 
            cls_score = self.classifier(feat)   #（128， num_class）
            cls_score_proj = self.classifier_proj(feat_proj)  #（128， num_class）
            return [cls_score, cls_score_proj], [img_feature, img_feature_proj], img_feature_proj

        else:
            if self.neck_feat == 'after':
                return torch.cat([feat, feat_proj], dim=1)
            else:
                return torch.cat([img_feature, img_feature_proj], dim=1)   #[(128,768)(128, 512)]->(128, 1280)

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))
