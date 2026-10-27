#!/usr/bin/env python3
"""List GGUF tensor types without loading the weights: reads only the header/metadata."""
import sys, struct, collections

TYPES = {0:"F32",1:"F16",2:"Q4_0",3:"Q4_1",6:"Q5_0",7:"Q5_1",8:"Q8_0",9:"Q8_1",10:"Q2_K",11:"Q3_K",
         12:"Q4_K",13:"Q5_K",14:"Q6_K",15:"Q8_K",16:"IQ2_XXS",17:"IQ2_XS",18:"IQ3_XXS",19:"IQ1_S",
         20:"IQ4_NL",21:"IQ3_S",22:"IQ2_S",23:"IQ4_XS",24:"I8",25:"I16",26:"I32",27:"I64",28:"F64",
         29:"IQ1_M",30:"BF16",34:"TQ1_0",35:"TQ2_0",36:"MXFP4"}
# bytes per element for size estimation (approx, block-based)
BPW = {0:32,1:16,2:4.5,3:5,6:5.5,7:6,8:8.5,9:9,10:2.625,11:3.4375,12:4.5,13:5.5,14:6.5625,15:8.5,
       16:2.0625,17:2.3125,18:3.0625,19:1.5625,20:4.5,21:3.4375,22:2.5625,23:4.25,24:8,25:16,26:32,
       27:64,28:64,29:1.75,30:16,34:2.0625,35:2.3125,36:4.25}

class R:
    def __init__(s, f): s.f=f
    def u32(s): return struct.unpack("<I", s.f.read(4))[0]
    def u64(s): return struct.unpack("<Q", s.f.read(8))[0]
    def i64(s): return struct.unpack("<q", s.f.read(8))[0]
    def string(s): n=s.u64(); return s.f.read(n).decode("utf-8","replace")
    def skip(s, t):
        if t in (0,1): s.f.read(1)
        elif t in (2,3): s.f.read(2)
        elif t in (4,5,6): s.f.read(4)
        elif t==7: s.f.read(1)
        elif t==8: s.string()
        elif t==9:
            et=s.u32(); n=s.u64()
            for _ in range(n): s.skip(et)
        elif t in (10,11,12): s.f.read(8)
        else: raise ValueError("bad kv type %d" % t)

def main(path):
    f=open(path,"rb")
    assert f.read(4)==b"GGUF", "not a GGUF"
    ver=struct.unpack("<I",f.read(4))[0]
    r=R(f); nt=r.u64(); nkv=r.u64()
    for _ in range(nkv):
        k=r.string(); t=r.u32(); r.skip(t)
    counts=collections.Counter(); params=collections.Counter(); big=[]
    for _ in range(nt):
        name=r.string(); nd=r.u32()
        dims=[r.u64() for _ in range(nd)]
        tt=r.u32(); r.u64()  # offset
        n=1
        for d in dims: n*=d
        tn=TYPES.get(tt,"type%d"%tt)
        counts[tn]+=1
        params[tn]+=n
        big.append((n, name, tn, dims[:2]))
    print("GGUF v%d, %d tensors" % (ver, nt))
    print("by params (M):", {k: round(v/1e6,1) for k,v in params.most_common()})
    print("\nlargest tensors:")
    for n,name,tn,dims in sorted(big, reverse=True)[:8]:
        print("  %-28s %-8s %s -> %.1f M" % (name, tn, dims, n/1e6))

if __name__=="__main__": main(sys.argv[1])
