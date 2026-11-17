"""Resumable JVP-Cal replay on a stored attack trajectory."""
import argparse, json, os, time, copy, random
import numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, Subset
from src.models.small_cnn import SmallCNN
from src.data.cifar10 import get_cifar10_dataloaders
from src.data.backdoor import create_asr_dataloader
from src.training.train_utils import evaluate_extended
from src.unlearning.jvp_cal import unflatten_like, ggn_vector_product, exact_hvp, taylor_update, rescale_global
from scripts.baseline.run_federaser_v5 import compute_model_delta, calibrate_update, aggregate_fedavg_modified

def run(args):
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out=args.out_dir; os.makedirs(out,exist_ok=True); marker=os.path.join(out,"progress.json")
    source=args.trajectory; src=os.path.join(source,"trajectory")
    train,test=get_cifar10_dataloaders("./data",batch_size=args.batch_size,use_fake=False)
    partitions=torch.load(os.path.join(source,"client_partitions.pt"),map_location="cpu",weights_only=False)
    if isinstance(partitions,dict): partitions=partitions.get("partitions",partitions.get("client_partitions"))
    model=SmallCNN().to(device); model.load_state_dict(torch.load(os.path.join(source,"initial_model.pt"),map_location=device,weights_only=True))
    start=1; done={}
    if os.path.exists(marker):
        done=json.load(open(marker)); start=done.get("last_round",0)+1
        if done.get("args") != vars(args):
            raise ValueError("Cannot resume with a different configuration; choose a new output directory")
        if start>1: model.load_state_dict(torch.load(os.path.join(out,"round_%03d.pt"%(start-1)),map_location=device,weights_only=True))
    counts=[len(partitions[c]) for c in range(1,10)]
    for r in range(start,51):
        tick=time.time(); ups=[]
        old_state = (torch.load(os.path.join(source,"initial_model.pt"),map_location=device,weights_only=True) if r == 1 else
                     torch.load(os.path.join(src,f"round_{r-1:03d}","global_model.pt"),map_location=device,weights_only=True))
        anchor = r > 1 and (r <= 1 + args.warmup or (args.anchor_every > 0 and (r-1) % args.anchor_every == 0))
        for c in range(1,10):
            hist=torch.load(os.path.join(src,f"round_{r:03d}",f"delta_client_{c:02d}.pt"),map_location=device,weights_only=True)
            if r==1 or (args.method=="stale" and not anchor): est=hist
            elif args.method in ("fed_e1", "fed_e2") or anchor:
                from src.training.train_utils import train_epoch
                lm=SmallCNN().to(device); lm.load_state_dict(model.state_dict()); opt=torch.optim.SGD(lm.parameters(),lr=.01,momentum=.9)
                gen=torch.Generator().manual_seed(args.seed+r*100+c); loader=DataLoader(partitions[c],batch_size=args.batch_size,shuffle=True,generator=gen)
                for _ in range(2 if args.method == "fed_e2" else 1): train_epoch(lm,loader,nn.CrossEntropyLoss(),opt,device)
                est=calibrate_update(hist,compute_model_delta(model.state_dict(),lm.state_dict()))
            else:
                lm=SmallCNN().to(device)
                point = {k: (old_state[k] + (model.state_dict()[k] - old_state[k]) * 0.5
                              if args.point == "midpoint" and old_state[k].is_floating_point()
                              else old_state[k]) for k in old_state}
                lm.load_state_dict(point); lm.eval()
                params={k:v.detach() for k,v in lm.named_parameters()}
                buffers={k:v.detach().clone() for k,v in lm.named_buffers()}
                current=model.state_dict()
                delta={k:current[k]-old_state[k] for k in params}
                delta_vec=delta
                idx=partitions[c].indices; gen=torch.Generator().manual_seed(args.seed*1000+c)
                chosen=torch.randperm(len(idx),generator=gen)[:min(args.bc,len(idx))].tolist(); samples=Subset(partitions[c],chosen)
                # Freeze one augmented batch for each HVP. Curvature uses eval-mode BN;
                # cloned buffers prevent mutation but deviate from train-mode curvature.
                batch=next(iter(DataLoader(samples,batch_size=min(args.bc,len(samples)),shuffle=False)))
                x,y=batch[0].to(device),batch[1].to(device)
                norm=float(torch.sqrt(sum(v.float().square().sum() for v in delta.values())).item())
                if norm<1e-12: corr={k:hist[k].to(device).clone() for k in params}
                else:
                    curvature=ggn_vector_product if args.curvature=="ggn" else exact_hvp
                    hv=curvature(lm,params,buffers,x,delta_vec) if args.curvature=="ggn" else curvature(lm,params,buffers,x,y,delta_vec)
                    if args.order==2:
                        hv2=curvature(lm,params,buffers,x,hv) if args.curvature=="ggn" else curvature(lm,params,buffers,x,y,hv)
                    else: hv2=None
                    corr=taylor_update({k:hist[k].to(device) for k in params},hv,hv2,args.lr,args.steps,args.order)
                est={k:(corr[k] if k in corr else hist[k].to(device)) for k in hist}
                if args.rescale: est=rescale_global(est,hist)
            ups.append(est)
        state=aggregate_fedavg_modified(model.state_dict(),ups,counts); model.load_state_dict(state)
        torch.save(model.state_dict(),os.path.join(out,f"round_{r:03d}.pt"))
        done={"last_round":r,"elapsed_s":done.get("elapsed_s",0)+time.time()-tick,"method":args.method,"args":vars(args)}
        tmp=marker+".tmp"; json.dump(done,open(tmp,"w"),indent=2); os.replace(tmp,marker)
        print(f"round={r} elapsed={time.time()-tick:.2f}s",flush=True)
    _,test=get_cifar10_dataloaders("./data",batch_size=args.batch_size,use_fake=False)
    asr=create_asr_dataloader(test,target_class=0,trigger_size=3,batch_size=args.batch_size)
    model.eval(); ca=evaluate_extended(model,test,nn.CrossEntropyLoss(),device); aa=evaluate_extended(model,asr,nn.CrossEntropyLoss(),device)
    result={"method":args.method,"ca":ca["accuracy"],"asr":aa["accuracy"],"runtime_s":done.get("elapsed_s"),"device":str(device),"config":vars(args)}
    json.dump(result,open(os.path.join(out,"metrics.json"),"w"),indent=2)
    open(os.path.join(out,"COMPLETE"),"w").write("complete\n")
    print(json.dumps(result),flush=True)

if __name__=="__main__":
 p=argparse.ArgumentParser(); p.add_argument("--trajectory",default="results/jacobian/iid_attack_logged"); p.add_argument("--out-dir",required=True); p.add_argument("--method",choices=["jvp","stale","fed_e1","fed_e2"],default="jvp"); p.add_argument("--curvature",choices=["ggn","exact"],default="ggn"); p.add_argument("--order",type=int,choices=[1,2],default=1); p.add_argument("--bc",type=int,default=64); p.add_argument("--steps",type=int,default=1); p.add_argument("--lr",type=float,default=.01); p.add_argument("--rescale",action="store_true"); p.add_argument("--batch-size",type=int,default=64); p.add_argument("--seed",type=int,default=123); p.add_argument("--point",choices=["old","midpoint"],default="old"); p.add_argument("--anchor-every",type=int,default=0); p.add_argument("--warmup",type=int,default=0); run(p.parse_args())
