import torch
from torch import nn

## PARAMETERS

device = 'cuda' if torch.cuda.is_available() else 'cpu'
patch_size = 16

####### MSA Block #######

class MutliHeadSelfAttentionBlock(nn.Module):

    def __init__(self,
                 embedding_dimension: int=768,
                 num_heads: int=12,
                 attn_dropout: float=0):
        super().__init__()

        self.norm_layer = nn.LayerNorm(normalized_shape=embedding_dimension)

        self.MSA_layer = nn.MultiheadAttention(embed_dim=embedding_dimension,
                                               num_heads=num_heads,
                                               dropout=attn_dropout,
                                               batch_first=True)

    def forward(self, x: torch.tensor) -> torch.tensor:

        x = self.norm_layer(x)
        attn_output, _ = self.MSA_layer(query=x, ## that _ is for weights although I have explicitly wriiten False for it but it still gonna written None and this will dump it
                                     key=x,
                                     value=x,
                                     need_weights=False)

        return attn_output
    
####### MLP Block #######

class MLPBlock(nn.Module):

    def __init__(self, 
                 embedding_dimension, 
                 mlp_size: int = 3072,
                 dropout: float = 0.1):
        
        super().__init__()

        self.norm_layer = nn.LayerNorm(normalized_shape=embedding_dimension)

        self.mlp = nn.Sequential(
            nn.Linear(in_features=embedding_dimension,
                      out_features=mlp_size),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=mlp_size,
                      out_features=embedding_dimension),
            nn.Dropout(p=dropout)
        )

    def forward(self, x: torch.tensor) -> torch.tensor:

        x = self.norm_layer(x)
        x = self.mlp(x)
        
        return x

####### Patch Embedding #######

class PatchEmbedding(nn.Module):

    def __init__(self, patch_size, in_channels, out_channels):
        super().__init__()

        self.patcher = nn.Conv2d(in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=patch_size,
                        stride=patch_size,
                        padding=0)
        
        self.flatten = nn.Flatten(start_dim=2,
                            end_dim=3)

    @staticmethod
    def create_class_token(embedded_image_shape):

        batch_size = embedded_image_shape[0]
        embedding_dimension = embedded_image_shape[2]

        class_token = nn.Parameter(torch.randn(size=(batch_size, 1, embedding_dimension)),
                                   requires_grad=True).to(device)
        
        return class_token
    
    @staticmethod
    def create_positional_token(embedded_image_shape):

        batch_size = embedded_image_shape[0]
        num_of_patches = embedded_image_shape[1]
        embedding_dimension = embedded_image_shape[2]

        position_token = nn.Parameter(torch.randn(size=(batch_size, num_of_patches, embedding_dimension)),
                                      requires_grad=True).to(device)

        return position_token

    def forward(self, x):

        image_resolution = x.shape[-1]

        assert image_resolution % patch_size == 0, f"Image size must be divisible by patch size, image shape: {image_resolution}, patch_size: {patch_size}"

        patch_embedding = self.flatten(self.patcher(x)).permute(0, 2, 1)

        ## concat class token

        class_token = self.create_class_token(patch_embedding.shape)

        token = torch.cat((class_token, patch_embedding),
                           dim=1)

        ## add positional embeddings

        position_embedding = self.create_positional_token(token.shape)

        token = token + position_embedding

        return token
    

####### Transformer Encoder Block #######


class TransformerEncoderBlock(nn.Module):

    def __init__(self,
                 embedding_dimension:int = 768,
                 num_heads:int = 12,
                 mlp_size:int = 3072,
                 msa_dropout:float = 0.0,
                 mlp_dropout:float = 0.1):
        super().__init__()

        self.msa_block = MutliHeadSelfAttentionBlock(embedding_dimension=embedding_dimension,
                                                     num_heads=num_heads,
                                                     attn_dropout=msa_dropout)
        
        self.mlp_block = MLPBlock(embedding_dimension=embedding_dimension,
                                  mlp_size=mlp_size,
                                  dropout=mlp_dropout)
        
    def forward(self, x:torch.tensor) -> torch.tensor:
        
        msa_output = self.msa_block(x) + x

        mlp_output = self.mlp_block(msa_output) + msa_output

        return mlp_output


####### ViT #######


class ViT(nn.Module):

    def __init__(self,
                 patch_size:int = 16,
                 in_channels:int = 3,
                 out_channels:int = 768,
                 embedding_dimension:int = 768,
                 num_heads:int = 12,
                 attn_dropout:float = 0.0,
                 mlp_size:int = 3072,
                 dropout:float = 0.1,
                 transformer_encoder_layers:int = 12,
                 num_classes:int = 3):
        super().__init__()

        self.patch_embedding = PatchEmbedding(patch_size=patch_size,
                                              in_channels=in_channels,
                                              out_channels=out_channels)


        self.transformer_encoder = nn.Sequential(*[TransformerEncoderBlock(embedding_dimension=embedding_dimension,
                                                                          num_heads=num_heads,
                                                                          mlp_size=mlp_size,
                                                                          msa_dropout=attn_dropout,
                                                                          mlp_dropout=dropout) for _ in range(transformer_encoder_layers)])

        self.classifier_head = nn.Sequential(

            nn.LayerNorm(normalized_shape=embedding_dimension),
            nn.Linear(in_features=embedding_dimension,
                      out_features=num_classes)
        )

    def forward(self, x: torch.tensor) -> torch.tensor:

        tokens = self.patch_embedding(x)
    
        embeddings = self.transformer_encoder(tokens)

        x = self.classifier_head(embeddings[:, 0])

        return x