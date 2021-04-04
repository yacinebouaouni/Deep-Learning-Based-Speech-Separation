
import numpy as np
import matplotlib.pyplot as plt
import torch


def Viz_Y(t,f,Y, vmin=0, vmax=20):
    plt.figure(figsize=(20,7))
    plt.pcolormesh(t, f, Y,vmin=0, vmax=20, shading='gouraud')
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()
    

def Reconstruct(B,G,Ns,Nm,Yabs,p):
    
    numerators=[]
    B1=B[:,:Ns]
    B2=B[:,Ns:]
    G1=G[:Ns,:]
    G2=G[Ns:,:]
    
    
    numerators.append(np.power(np.matmul(B1,G1),p))
    numerators.append(np.power(np.matmul(B2,G2),p))

    denominator = np.power(np.matmul(B1,G1),p)+np.power(np.matmul(B2,G2),p)
  
    

    Sources=[]
    Masks=[]
    for i in range(2):

        Sources.append(np.multiply(numerators[i]/denominator,Yabs))
        Masks.append(numerators[i]/denominator)

    #print('Source shape = {}'.format(Sources[0].shape))
    
    return Sources,Masks


def Reconstruct_2comp(n_components,B,G, Yabs):
    
    percents=[]
    numerators=[]
    
    denominator = np.zeros((B.shape[0],G.shape[1]))
    for i in range(n_components):
    
        denominator += np.matmul(B[:,i].reshape((B.shape[0],1)),G[i,:].reshape((1,G.shape[1])))
        numerator = np.matmul(B[:,i].reshape((B.shape[0],1)),G[i,:].reshape((1,G.shape[1])))
        numerators.append(numerator)

        
    for i in range(n_components):
        
        percents.append(numerators[i]/(denominator+0.001))
    
    
    Sources=[]

    for i in range(n_components):

        Sources.append(np.multiply(percents[i],Yabs))

    print('Source shape = {}'.format(Sources[0].shape))
    
    return Sources


def SMR(speech, music):
    
    """
    Function that takes music and speech signals.
    returns SMR in db
    """
    speech_power = torch.tensor(speech,dtype=torch.float64).norm(p=2)
    music_power = torch.tensor(music,dtype=torch.float64).norm(p=2)
    SMR_db=10*np.log10(speech_power/music_power)
    print('SMR = {:.2f}'.format(SMR_db))
    
    return SMR_db

def SDR(s_est, s):
    """
    Function that takes original and estimated spectrogram
    returns SDR in DB
    """
    
    signal_power = torch.tensor(s,dtype=torch.float64).norm(p=2)
    distorsion_power = torch.tensor(s-s_est,dtype=torch.float64).norm(p=2)
    SDR_db=10*np.log10(signal_power/distorsion_power)
    
    return SDR_db

def get_mixed_signal(speech, music, SMR_db):
    """
    Function taht takes the speech and music signal alongside the SMR_db
    returns the mixed signal and the scaled speech
    """
    smr = 10**(SMR_db/10)
    speech_power = torch.tensor(speech,dtype=torch.float64).norm(p=2)
    music_power = torch.tensor(music,dtype=torch.float64).norm(p=2)
    scale = smr * music_power / speech_power
    
    if SMR_db ==0:
        mixed = speech + music
        return mixed,speech,music
    
    if SMR_db < 0 :
        mixed = scale* speech + music
        speech_scaled=scale*speech
        SMR(speech_scaled,music)
        return mixed,speech_scaled,music
    
    if SMR_db >0 :
        
        mixed =  speech + music * (1/scale)
        music_scaled=(1/scale) * music
        SMR(speech,music_scaled)
        return mixed,speech,music_scaled
     