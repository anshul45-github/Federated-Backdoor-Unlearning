import json
import torch
from torch.utils.flop_counter import FlopCounterMode
from src.models.small_cnn import SmallCNN
from src.unlearning.jvp_cal import ggn_vector_product
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=SmallCNN().to(device)
x=torch.randn(64,3,32,32,device=device)
y=torch.randint(10,(64,),device=device)
out={"device":str(device),"batch_size":64,"warning":"Operator FLOP counter is incomplete for some operations; no end-to-end FLOP claim."}
try:
 model.train()
 with FlopCounterMode(display=False) as counter:
  torch.nn.functional.cross_entropy(model(x),y).backward()
 out["sgd_forward_backward_flops"]=counter.get_total_flops()
except Exception as e: out["sgd_error"]=repr(e)
try:
 model.eval()
 params={k:v.detach() for k,v in model.named_parameters()}
 buffers={k:v.detach().clone() for k,v in model.named_buffers()}
 vec={k:torch.randn_like(v) for k,v in params.items()}
 with FlopCounterMode(display=False) as counter:
  ggn_vector_product(model,params,buffers,x,vec)
 out["ggn_hvp_flops"]=counter.get_total_flops()
except Exception as e: out["ggn_error"]=repr(e)
with open("results/jacobian/followup_flop_probe.json","w") as f: json.dump(out,f,indent=2)
print(json.dumps(out,indent=2))
