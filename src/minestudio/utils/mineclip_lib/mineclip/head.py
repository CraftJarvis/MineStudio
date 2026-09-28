import torch
import torch.nn as nn

from minestudio.utils.mineclip_lib.utils import build_mlp


class CLIPScoreHead(nn.Module):
    def __init__(
        self,
        clip_model,
        *,
        video_adapter_layers,
        text_adapter_layers,
        feature_dim,
    ):
        super().__init__()
        self.clip_model = clip_model
        self.video_residual_weight = None
        self.text_residual_weight = None

        if video_adapter_layers == 0:
            self.video_adapter = nn.Identity()
        else:
            self.video_adapter = build_mlp(
                input_dim=feature_dim,
                output_dim=feature_dim,
                hidden_dim=feature_dim,
                num_layers=video_adapter_layers,
                add_input_activation=False,
            )
            self.video_residual_weight = nn.Parameter(torch.tensor(4.0))

        if text_adapter_layers == 0:
            self.text_adapter = nn.Identity()
        else:
            self.text_adapter = build_mlp(
                input_dim=feature_dim,
                output_dim=feature_dim,
                hidden_dim=feature_dim,
                num_layers=text_adapter_layers,
                add_input_activation=False,
            )
            # input * sigmoid(res_weight) + MLP(input) * (1-sigmoid(res_weight))
            # initialize res_weight to be positive so sigmoid(res_weight) is close to 1
            self.text_residual_weight = nn.Parameter(torch.tensor(4.0))

    def forward(self, video_feature, texts):
        if self.video_residual_weight is None:
            adapted_img = self.video_adapter(video_feature)
        else:
            res = torch.sigmoid(self.video_residual_weight)
            adapted_img = res * video_feature + (1.0 - res) * self.video_adapter(
                video_feature
            )
        text_feature = self.clip_model.encode_text(texts)
        if self.text_residual_weight is None:
            adapted_text = self.text_adapter(text_feature)
        else:
            res = torch.sigmoid(self.text_residual_weight)
            adapted_text = res * text_feature + (1.0 - res) * self.text_adapter(
                text_feature
            )
        logits_per_video, logits_per_text = self.clip_model(adapted_img, adapted_text)
        return logits_per_video, logits_per_text
