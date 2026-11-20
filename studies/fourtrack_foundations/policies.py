"""LeRobot's pretrained ACT and Diffusion Policy as downstream four-track adapters."""
import torch
from torch.nn import functional as F
from safetensors.torch import load_file
from .common import WORK,P,ensure

def normalized_image(rgb,size):
    v=F.interpolate(rgb.clamp(0,1),size=size,mode='bilinear',align_corners=False)
    return (v-v.new_tensor([.485,.456,.406])[None,:,None,None])/v.new_tensor([.229,.224,.225])[None,:,None,None]

class Policy:
    def __init__(self,arm):
        from lerobot.utils.constants import OBS_IMAGES,OBS_STATE
        self.images,self.state=OBS_IMAGES,OBS_STATE
        self.arm=arm
        folder=WORK/('upstream-'+arm); ref=P[arm]
        ensure(folder,{'model.safetensors':ref['sha256']})
        if arm=='act':
            from lerobot.policies.act.configuration_act import ACTConfig as Config
            from lerobot.policies.act.modeling_act import ACTPolicy as Native
        else:
            from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig as Config
            from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy as Native
        cfg=Config.from_pretrained(str(folder),local_files_only=True);cfg.device='cuda'
        cfg.pretrained_backbone_weights=None  # Entire trained ResNet is present in the pinned official policy.
        if arm=='diffusion':
            cfg.noise_scheduler_type='DDIM';cfg.num_inference_steps=4
        policy=Native(cfg)
        weights=load_file(str(folder/'model.safetensors'),device='cpu')
        # Legacy releases optionally serialized source-robot normalizers inside the policy.
        # Only those buffers are outside the current LeRobot policy; require every model tensor.
        normalization=[k for k in weights if k.split('.')[0] in {'normalize_inputs','normalize_targets','unnormalize_outputs'}]
        for k in normalization: del weights[k]
        policy.load_state_dict(weights,strict=True)
        del weights
        policy.eval().requires_grad_(False).cuda()
        self.cfg=cfg;self.net=policy.model if arm=='act' else policy.diffusion
        self.project=self.net.action_head if arm=='act' else self.net.unet.final_conv[-1]
        self.hook=self.project.register_forward_pre_hook(self.capture)
        self.meta=dict(base_sha256=ref['sha256'],upstream_commit=ref['revision'],
            original_policy_parameters=sum(p.numel() for p in policy.parameters()),
            native_output_dimensions=self.project.weight.shape[0],normalizer_buffers_excluded=normalization,
            upstream=ref['model'],sampler='native ACT all 100 queries' if arm=='act' else 'native LeRobot DDIM 4, fixed zero noise, horizon 16')
    def capture(self,core,args):
        z=args[0]
        self.terminal=(z[:,0] if self.arm=='act' else z[:,:,self.cfg.n_obs_steps-1]).float()
    def inputs_key(self,x):
        return x[:,:158] if self.arm=='act' else torch.cat((x[:,:144],x[:,156:157],x[:,-1:]),1)
    def __call__(self,x):
        b=x[:,:72].reshape(-1,1,6,12);dx=x[:,72:144].reshape_as(b)
        if self.arm=='act':
            im=normalized_image(torch.cat((5*b,5*(b+dx),.5+3*dx),1),(192,256))
            with torch.autocast('cuda',dtype=torch.bfloat16):
                actions,_=self.net({self.images:[im],self.state:x[:,144:158]})
            a=actions[:,0]
        else:
            pic=torch.stack((b,b+dx),1).flatten(0,1).expand(-1,3,-1,-1)
            im=normalized_image(5*pic,(96,96)).reshape(len(x),2,1,3,96,96)
            # Two public instrument readings supplied to the original two-coordinate Push-T state port.
            # Both histories also supply all 18 continuous readings, unchanged, to the four-track interface.
            state=torch.stack((x[:,156],x[:,-1]),1)[:,None].expand(-1,2,-1)
            noise=x.new_zeros((len(x),self.cfg.horizon,2))
            with torch.autocast('cuda',dtype=torch.bfloat16):
                a=self.net.generate_actions({self.images:im,self.state:state},noise=noise)[:,0]
        return {'v':self.terminal.half().cpu(),'native':a.float().cpu()}
    def projection(self):
        return {'weight':self.project.weight.detach().reshape(self.project.weight.shape[:2]).float().cpu(),
                'bias':self.project.bias.detach().float().cpu()}
