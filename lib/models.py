import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision.models import mobilenet_v2
from torch.nn import functional as F
from torchvision.models import resnet18
from torchvision.models import mobilenet_v3_small
from torchvision.models import efficientnet_b0
from torchvision.models import mobilenet_v3_large
from torchvision.models import efficientnet_b3
from torchvision.models import densenet121, efficientnet_b4
from timm import create_model  # For Xception

class ClassToken(nn.Module):
    def __init__(self, num_features):
        super(ClassToken, self).__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, num_features))

    def forward(self, x):
        batch_size = x.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        return torch.cat((cls_tokens, x), dim=1)

class TransformerEncoder(nn.Module):
    def __init__(self, cf):
        super(TransformerEncoder, self).__init__()
        self.layer_norm1 = nn.LayerNorm(cf['hidden_dim'])
        self.attention = nn.MultiheadAttention(embed_dim=cf['hidden_dim'], num_heads=cf['num_heads'])
        self.layer_norm2 = nn.LayerNorm(cf['hidden_dim'])
        self.mlp = nn.Sequential(
            nn.Linear(cf['hidden_dim'], cf['mlp_dim']),
            nn.GELU(),
            nn.Dropout(cf['dropout_rate']),
            nn.Linear(cf['mlp_dim'], cf['hidden_dim']),
            nn.Dropout(cf['dropout_rate'])
        )

    def forward(self, x):
        # Self-attention part
        x = x.permute(1, 0, 2)  # Convert (batch, seq_len, feature) to (seq_len, batch, feature) for attention
        x1 = self.layer_norm1(x)
        x1, _ = self.attention(x1, x1, x1)
        x = x + x1
        x = x.permute(1, 0, 2)  # Revert back

        # MLP part
        x2 = self.layer_norm2(x)
        x2 = self.mlp(x2)
        x = x + x2

        return x







class ResNet18ViT(nn.Module):
    def __init__(self, cf):
        super(ResNet18ViT, self).__init__()
        self.num_patches = cf['num_patches']
        self.hidden_dim = cf['hidden_dim']

        # Initialize ResNet18 without the fully connected layer and without adaptive pooling
        base_resnet18 = resnet18(pretrained=True)
        base_resnet18.fc = nn.Identity()  # Remove fully connected layer
        layers = list(base_resnet18.children())[:-2]  # Remove the last pooling layer
        self.resnet18 = nn.Sequential(*layers)

        # Convolution layer for patch embeddings
        kernel_size = cf['patch_size']
        stride = cf['patch_size']
        padding = (kernel_size // 2)  # 'same' padding approximation if the kernel is odd
        self.patch_embed = nn.Conv2d(512, cf['hidden_dim'], kernel_size=kernel_size, stride=stride, padding=padding)

        # Position embeddings
        self.pos_embed = nn.Parameter(torch.randn(cf['num_patches'], cf['hidden_dim']))

        # Class Token
        self.cls_token = nn.Parameter(torch.randn(1, 1, cf['hidden_dim']))

        # Transformer Encoders
        self.transformer_encoders = nn.ModuleList([
            TransformerEncoder(cf) for _ in range(cf['num_layers'])
        ])

        # Layer normalization and classifier head
        self.layer_norm = nn.LayerNorm(cf['hidden_dim'])
        self.head = nn.Linear(cf['hidden_dim'], cf['num_classes'])

    def forward(self, x):
        x = self.resnet18(x)  # Should output in the format [batch_size, channels, height, width]
        # print("Shape after ResNet50:", x.shape)

        x = self.patch_embed(x)
        # print("Shape after patch embedding:", x.shape)

        # Flatten and reshape to match the transformer's expected input
        x = x.flatten(2).permute(0, 2, 1)  # [batch, num_patches, hidden_dim]

        # Add position embeddings
        x = x + self.pos_embed.unsqueeze(0)

        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)

        x = torch.cat((cls_tokens, x), dim=1)

        for encoder in self.transformer_encoders:
            x = encoder(x)

        x = self.layer_norm(x)
        x = x[:, 0, :]
        logits = self.head(x)

        return logits



class DenseNet121ViT(nn.Module):
    def __init__(self, cf):
        super(DenseNet121ViT, self).__init__()
        self.num_patches = cf['num_patches']
        self.hidden_dim = cf['hidden_dim']

        # Initialize DenseNet-121 without the fully connected layer
        base_densenet = densenet121(pretrained=True)
        self.densenet = nn.Sequential(*list(base_densenet.features.children()))

        # Adjust input channels from 1024 (output of DenseNet-121) to match hidden_dim
        self.patch_embed = nn.Conv2d(1024, cf['hidden_dim'], kernel_size=cf['patch_size'], stride=cf['patch_size'], padding=(cf['patch_size'] // 2))

        # Position embeddings
        self.pos_embed = nn.Parameter(torch.randn(cf['num_patches'], cf['hidden_dim']))

        # Class Token
        self.cls_token = nn.Parameter(torch.randn(1, 1, cf['hidden_dim']))

        # Transformer Encoders
        self.transformer_encoders = nn.ModuleList([
            TransformerEncoder(cf) for _ in range(cf['num_layers'])
        ])

        # Layer normalization and classifier head
        self.layer_norm = nn.LayerNorm(cf['hidden_dim'])
        self.head = nn.Linear(cf['hidden_dim'], cf['num_classes'])

    def forward(self, x):
        x = self.densenet(x)  # Should output in the format [batch_size, channels, height, width]

        x = self.patch_embed(x)

        # Flatten and reshape to match the transformer's expected input
        x = x.flatten(2).permute(0, 2, 1)  # [batch, num_patches, hidden_dim]

        # Add position embeddings
        x = x + self.pos_embed.unsqueeze(0)

        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)

        x = torch.cat((cls_tokens, x), dim=1)

        for encoder in self.transformer_encoders:
            x = encoder(x)

        x = self.layer_norm(x)
        x = x[:, 0, :]
        logits = self.head(x)

        return logits


def get_model(args):
    # Convert args to dictionary
    config = {
        "num_layers": args.num_layers,
        "hidden_dim": args.hidden_dim,
        "mlp_dim": args.mlp_dim,
        "num_heads": args.num_heads,
        "dropout_rate": args.dropout_rate,
        "image_size": args.image_size,
        "patch_size": args.patch_size,
        "num_classes": args.num_classes,
        "num_patches": args.num_patches
    }

    device = torch.device(args.device)
    if args.network == "resnet18":
        model = ResNet18ViT(config).to(device) 
        print("the model is :", "ResNet18ViT")
 # Make sure to adjust ResNet50ViT to your actual model
    # elif args.network == "MobileNetV3Small".lower :
    #     model = MobileNetV3SmallViT(config).to(device)  # Make sure to adjust ResNet50ViT to your actual model
    # elif args.network == "EfficientNetB0".lower :
    #     model = EfficientNetB0ViT(config).to(device)  # Make sure to adjust ResNet50ViT to your actual model

    # elif args.network == "MobileNetV3Large".lower :
    #     model = MobileNetV3LargeViT(config).to(device)
    elif args.network == "densnet121":
        model = DenseNet121ViT(config).to(device)  # Use EfficientNet-B3 as the backbone
        print("the model is :", "densnet121")
    else:
        model = ResNet18ViT(config).to(device)  # Make sure to adjust ResNet50ViT to your actual model


    # model = MobileNetV2ViT(config).to(device)
    return model
