PYTHONPATH=$PYTHONPATH:../../EOC GLOG_vmodule=MemcachedClient=-1 \
#Use srun if possible \
# srun -n16 --gpu \
"python -u -m prototype.solver.cls_solver --config config.yaml --evaluate"
# --recover=checkpoints/ckpt.pth.tar 
