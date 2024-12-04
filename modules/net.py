import torch as ch
import torch.nn as nn
import modules.vision_transformer as vits
import modules.utils as utils

################################
##### SSL Model Generic CLass ##
################################

class SSLNetwork(nn.Module):
    def __init__(
        self, cfg
    ):
        self.cfg = cfg
        arch = cfg.pretrain.model.arch
        remove_head = cfg.pretrain.model.remove_head
        mlp = cfg.pretrain.model.mlp
        patch_keep = cfg.pretrain.model.patch_keep
        fc = cfg.pretrain.model.fc
        loss = cfg.pretrain.training.loss


        super().__init__()
        if "resnet" in arch:
            import torchvision.models.resnet as resnet
            self.net = resnet.__dict__[arch]()
            if fc:
                self.net.fc = nn.Linear(2048, 256)
            else:
                self.net.fc = nn.Identity()
        elif "vgg" in arch:
            import torchvision.models.vgg as vgg
            self.net = vgg.__dict__[arch]()
            self.net.classifier = nn.Identity()
            
        elif arch in vits.__dict__.keys():
            self.net = vits.__dict__[arch](
                patch_size=16,
                drop_path_rate=0.1,  # stochastic depth
                )
            if fc:
                self.net.fc = nn.Linear(2048, 256)
            else:
                self.net.fc = nn.Identity()

        else:
            print("Arch not found")
            exit(0)

        # Compute the size of the representation
        self.representation_size = self.net(ch.zeros((1,3,224,224))).size(1)
        print("REPR SIZE:", self.representation_size)
        # Add a projector head
        self.mlp = mlp
        if remove_head:
            self.num_features = self.representation_size
            self.projector = nn.Identity()
        else:
            self.num_features = int(self.mlp.split("-")[-1])
            self.projector = self.MLP(self.representation_size)
        self.loss = loss
        if loss == "barlow":
            self.bn = nn.BatchNorm1d(self.num_features, affine=False)
        elif loss == "byol":
            self.predictor = self.MLP(self.num_features)


    def MLP(self, size):
        proj_relu = self.cfg.pretrain.model.proj_relu
        mlp_coeff = self.cfg.pretrain.model.mlp_coeff
        mlp_spec = f"{size}-{self.mlp}"
        layers = []
        f = list(map(int, mlp_spec.split("-")))
        f[-2] = int(f[-2] * mlp_coeff)
        print("MLP:", f)
        for i in range(len(f) - 2):
            layers.append(nn.Sequential(nn.Linear(f[i], f[i + 1]), nn.BatchNorm1d(f[i + 1]), nn.ReLU(True)))
        if proj_relu:
            layers.append(nn.Sequential(nn.Linear(f[-2], f[-1], bias=False), nn.ReLU(True)))
        else:
            layers.append(nn.Linear(f[-2], f[-1], bias=False))
        return nn.Sequential(*layers)

    def forward(self, inputs, embedding=False, predictor=False):
        if embedding:
            embedding = self.net(inputs)
            return embedding
        else:
            representation = self.net(inputs)
            embeddings = self.projector(representation)
            list_outputs = [representation.detach()]
            outputs_train = representation.detach()
            for l in range(len(self.projector)):
                outputs_train = self.projector[l](outputs_train).detach()
                list_outputs.append(outputs_train)
            if self.loss == "byol" and predictor:         
                embeddings = self.predictor(embeddings)
            return embeddings, list_outputs


class LinearsProbes(nn.Module):
    def __init__(self, cfg, model, num_classes):
        super().__init__()
        mlp_coeff = cfg.pretrain.model.mlp_coeff
        print("NUM CLASSES", num_classes)
        mlp_spec = f"{model.representation_size}-{model.mlp}"
        f = list(map(int, mlp_spec.split("-")))
        f[-2] = int(f[-2] * mlp_coeff)
        self.probes = []
        for num_features in f:
            self.probes.append(nn.Linear(num_features, num_classes))
        self.probes = nn.Sequential(*self.probes)

    def forward(self, list_outputs, binary=False):
        return [self.probes[i](list_outputs[i]) for i in range(len(list_outputs))]
    


class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x


class MultiCropWrapper(nn.Module):
    """
    Perform forward pass separately on each resolution input.
    The inputs corresponding to a single resolution are clubbed and single
    forward is run on the same resolution inputs. Hence we do several
    forward passes = number of different resolutions used. We then
    concatenate all the output features and run the head forward on these
    concatenated features.
    """
    def __init__(self, backbone, head):
        super(MultiCropWrapper, self).__init__()
        # disable layers dedicated to ImageNet labels classification
        backbone.fc, backbone.head = nn.Identity(), nn.Identity()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        # convert to list
        if not isinstance(x, list):
            x = [x]
        idx_crops = ch.cumsum(ch.unique_consecutive(
            ch.tensor([inp.shape[-1] for inp in x]),
            return_counts=True,
        )[1], 0)
        start_idx, output = 0, ch.empty(0).to(x[0].device)
        for end_idx in idx_crops:
            _out = self.backbone(ch.cat(x[start_idx: end_idx]))
            # The output is a tuple with XCiT model. See:
            # https://github.com/facebookresearch/xcit/blob/master/xcit.py#L404-L405
            if isinstance(_out, tuple):
                _out = _out[0]
            # accumulate outputs
            output = ch.cat((output, _out))
            start_idx = end_idx
        # Run the head forward on the concatenated features.
        return self.head(output)


def get_DINONetwork(cfg):
    USE_BN_IN_HEAD = False
    NORM_LAST_LAYER = True
    arch = cfg.pretrain.model.arch
    patch_size = cfg.pretrain.model.patch_size
    drop_path_rate = cfg.pretrain.model.drop_path_rate
    out_dim = cfg.pretrain.dino.out_dim
    
    # if arch is vit
    if arch in vits.__dict__.keys():
        student = vits.__dict__[arch](
            patch_size=patch_size,
            drop_path_rate=drop_path_rate,  # stochastic depth
    )
        teacher = vits.__dict__[arch](patch_size=patch_size)
        embed_dim = student.embed_dim
    else:
        print(f"Architecture Unknown: {arch}")
        
    # multi-crop wrapper handles forward with inputs of different resolutions
    student = MultiCropWrapper(student, DINOHead(
        embed_dim,
        out_dim,
        use_bn=USE_BN_IN_HEAD,
        norm_last_layer=NORM_LAST_LAYER,
    ))
    teacher = MultiCropWrapper(
        teacher,
        DINOHead(embed_dim, out_dim, USE_BN_IN_HEAD),
    )
    
    # student, teacher = student.cuda(), teacher.cuda()
    return student, teacher