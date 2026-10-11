"""A tiny independent reproduction of the old query's four checkpoints.
No source credentials, no production access. Positive volumes only.
"""
rows=[('10:00:00','B',10),('10:29:30','S',20),('11:00:00','B',30),('11:29:20','S',40),('14:00:00','B',50),('14:59:10','S',60)]
checkpoints=['10:29','11:29','13:59','14:59']
def cumulative_then_diff():
    buy=sell=0;samples={}
    for t,side,v in sorted(rows):
        buy+=v if side=='B' else 0;sell+=v if side=='S' else 0
        if t[:5] in checkpoints and buy and sell:samples[t[:5]]=(buy,sell)
    out=[];prev=(0,0)
    for k,(b,s) in sorted(samples.items()):out.append((k,b-prev[0],s-prev[1]));prev=(b,s)
    return out

def same_contract_direct():
    # Only keep observed checkpoint minutes with both cumulative sides positive.
    ends=[]
    for cp in checkpoints:
        minute=[t for t,_,_ in rows if t[:5]==cp]
        if not minute:continue
        end=max(minute)
        if all(sum(v for t,side,v in rows if t<=end and side==s)>0 for s in ['B','S']):ends.append((cp,end))
    prev='';out=[]
    for cp,end in ends:
        out.append((cp,sum(v for t,s,v in rows if prev<t<=end and s=='B'),sum(v for t,s,v in rows if prev<t<=end and s=='S')));prev=end
    return out
if __name__=='__main__':
    a=cumulative_then_diff();b=same_contract_direct()
    assert a==b==[('10:29',10,20),('11:29',30,40),('14:59',50,60)]
    print(a);print('PASS: same four-checkpoint contract. Missing 13:59 creates no row; no production speed claim.')
