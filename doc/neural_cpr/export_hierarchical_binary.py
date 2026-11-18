#!/usr/bin/env python3
"""Export hierarchical .npz weights as an OPM two-layer binary model."""
import argparse, struct
import numpy as np

def main():
    p=argparse.ArgumentParser(); p.add_argument('--input',required=True); p.add_argument('--output',required=True); a=p.parse_args()
    z=np.load(a.input); W=z['W'].astype(np.float32); b=z['b'].astype(np.float32)
    Wc,bc=z['Wc'],z['bc']; Wa,ba=z['Wa'],z['ba']; Wr,br=z['Wr'],z['br']
    # Runtime format contains shared dense layer then 12+108+648 score outputs.
    outW=np.concatenate([Wc,Wa.reshape(108,W.shape[0]),Wr.reshape(648,W.shape[0])]).astype(np.float32)
    outb=np.concatenate([bc,ba.reshape(108),br.reshape(648)]).astype(np.float32)
    with open(a.output,'wb') as f:
        f.write(struct.pack('<I',2))
        for rows,cols,weights,bias,act in [(W.shape[0],W.shape[1],W,b,6),(outW.shape[0],outW.shape[1],outW,outb,1)]:
            f.write(struct.pack('<IIII',3,cols,rows,rows)); f.write(weights.T.tobytes()); f.write(bias.tobytes()); f.write(struct.pack('<I',act))
    print(f'exported {a.output} with {outW.shape[0]} outputs')
if __name__=='__main__': main()
