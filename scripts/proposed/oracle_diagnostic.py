import argparse,json,os,torch,torch.nn as nn
from torch.utils.data import DataLoader
from src.models.small_cnn import SmallCNN
from src.unlearning.jvp_cal import ggn_vector_product,taylor_update,rescale_global
from src.training.train_utils import train_epoch

def cosine(a,b):
 x=torch.cat([v.float().flatten() for v in a.values()]);y=torch.cat([v.float().flatten() for v in b.values()]);return torch.nn.functional.cosine_similarity(x,y,dim=0).item()
def relerr(a,b):
 x=torch.cat([v.float().flatten() for v in a.values()]);y=torch.cat([v.float().flatten() for v in b.values()]);return (x-y).norm().item()/(y.norm().item()+1e-12)
def main():
 p=argparse.ArgumentParser();p.add_argument('--trajectory',default='results/jacobian/iid_attack_logged');p.add_argument('--replay',default='results/jacobian/iid_jvp_g64_o1_rescaled');p.add_argument('--out',required=True);a=p.parse_args()
 dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu');parts=torch.load(a.trajectory+'/client_partitions.pt',map_location='cpu',weights_only=False); rounds=[10,25,40]; clients=[1,5,9]; rows=json.load(open(a.out)) if os.path.exists(a.out) else []
 for r in rounds:
  old=torch.load(f'{a.trajectory}/trajectory/round_{r-1:03d}/global_model.pt',map_location=dev,weights_only=True) if r>1 else torch.load(a.trajectory+'/initial_model.pt',map_location=dev,weights_only=True)
  rec=torch.load(f'{a.replay}/round_{r-1:03d}.pt',map_location=dev,weights_only=True)
  delta_theta={k:rec[k].float()-old[k].float() for k in old if k in dict(SmallCNN().named_parameters())}
  for c in clients:
   if any(row['round']==r and row['client']==c for row in rows): continue
   hist=torch.load(f'{a.trajectory}/trajectory/round_{r:03d}/delta_client_{c:02d}.pt',map_location=dev,weights_only=True)
   model=SmallCNN().to(dev);model.load_state_dict(old);model.eval();params=dict(model.named_parameters());buffers=dict(model.named_buffers())
   d={k:delta_theta[k].to(dev) for k in params}; u={k:hist[k].to(dev) for k in params};hv=ggn_vector_product(model,params,buffers,torch.stack([parts[c][i][0] for i in range(min(8,len(parts[c])))]).to(dev),d)
   est=taylor_update(u,hv,eta=.01,steps=1,order=1)
   exactm=SmallCNN().to(dev);exactm.load_state_dict(rec);opt=torch.optim.SGD(exactm.parameters(),lr=.01,momentum=.9);loader=DataLoader(parts[c],batch_size=64,shuffle=False)
   for _ in range(2):train_epoch(exactm,loader,nn.CrossEntropyLoss(),opt,dev)
   st=exactm.state_dict();exact={k:st[k].float()-rec[k].float() for k in params}
   rows.append({'round':r,'client':c,'cos_stale':cosine(u,exact),'relerr_stale':relerr(u,exact),'cos_jvp':cosine(est,exact),'relerr_jvp':relerr(est,exact),'delta_theta_ratio':float(torch.sqrt(sum(x.square().sum() for x in d.values()))/torch.sqrt(sum(old[k].float().square().sum() for k in params))),'warning':'exact update uses full partition, whereas curvature uses 8 examples; stochastic SGD settings differ from historical client seed'})
   with open(a.out+'.tmp','w') as f:json.dump(rows,f,indent=2)
   os.replace(a.out+'.tmp',a.out)
 with open(a.out+'.DONE','w') as f:f.write('complete\n')
 print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
