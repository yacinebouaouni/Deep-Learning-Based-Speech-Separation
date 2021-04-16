import numpy as np 
import torch
def gain_params(s1,s2,y):

        l2_s1 = torch.norm(s1.float(),2, None)
                 
        l2_s2 = torch.norm(s2.float(),2, None)
               
        l2_Y = torch.norm(y.float(),2, None)
              
        # u is initialized by the l2-norm of the initial NMF source estimate sˆ1 
        # divided by the l2-norm of the mixed signal y
        u = torch.div(l2_s1, l2_Y) 

        # v is initialized by the same manner
        v = torch.div(l2_s2, l2_Y)

        return u,v
    
def gain_params_vec(s1,s2,y):

        l2_s1 = torch.norm(s1.float().t(),2, True)
                 
        l2_s2 = torch.norm(s2.float().t(),2, True)
               
        l2_Y = torch.norm(y.float().t(),2, True)
              
        # u is initialized by the l2-norm of the initial NMF source estimate sˆ1 
        # divided by the l2-norm of the mixed signal y
        u = torch.div(l2_s1, l2_Y) 

        # v is initialized by the same manner
        v = torch.div(l2_s2, l2_Y)

        return u,v
    
    
def feed_(x,model):
    
    f = model(x)
    f1 = f[0] 
    f2 = f[1]
    return f1,f2

def energy_1(x_source1,model):
    
    f1,f2 = feed_(x_source1,model)
    e1 = (1 - f1 ).pow(2) + f2.pow(2)
    return e1 

def energy_2(x_source2,model):

    f1,f2 = feed_(x_source2,model)
    e2 = f1.pow(2) + (1 - f2 ).pow(2) 
    return e2

def E_err(s1,s2,y):
  
    u,v = gain_params_vec(s1,s2,y) 
    return torch.norm(torch.mul(u,s1) + torch.mul(v,s2) - y,'fro',None)

def E_err_vec(s1,s2,y,u,v):
  
    return torch.norm(u*s1 + v*s2 - y,'fro',None)


def relu(x):
    return torch.nn.ReLU()(x)


def nonneg_constraint_sum(s1,s2,u,v):
    
    Rs1=min([s1.min().item(),0])**2
    Rs2=min([s2.min().item(),0])**2
    Ru=min([u,0])**2
    Rv=min([v,0])**2
    
    return Rs1+Rs2+Ru+Rv

def Criteria(s1,s2,Yabs,u,v,i,model,beta,l):
    
    # Feed forward and get energy 1 and 2
    e1 = energy_1(s1[:,i].float(),model)
    e2 = energy_2(s2[:,i].float(),model)
    
    # Get least square error :
    e_rr=E_err_vec(s1[:,i], s2[:,i], Yabs[:,i],u[i],v[i])
    
    # Non negative constraint
    R = nonneg_constraint_sum(s1[:,i], s2[:,i], u[i], v[i])
    


   
    return e1 + e2 + l*e_rr + beta*R
    