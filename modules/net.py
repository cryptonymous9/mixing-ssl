import torch as ch
import torch.nn as nn
import modules.vision_transformer as vits

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
