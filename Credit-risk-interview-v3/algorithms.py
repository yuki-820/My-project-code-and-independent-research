"""Readable interview references. Algorithm inputs are separate teaching fixtures."""
def three_sum(nums):
    a=sorted(nums)                     # sorted copy, keep input unchanged
    out=[]
    for i in range(len(a)-2):
        if i>0 and a[i]==a[i-1]:continue
        if a[i]>0:break                # remaining numbers cannot sum to zero
        left,right=i+1,len(a)-1
        while left<right:
            total=a[i]+a[left]+a[right]
            if total<0:left+=1
            elif total>0:right-=1
            else:
                out.append([a[i],a[left],a[right]])
                left+=1;right-=1
                while left<right and a[left]==a[left-1]:left+=1
                while left<right and a[right]==a[right+1]:right-=1
    return out

def quick_sort(nums):
    # Pedagogical three-way version, not in-place. Choose a random pivot in production
    # if adversarial inputs matter; Python's built-in sorted is usually preferable.
    if len(nums)<=1:return list(nums)
    pivot=nums[len(nums)//2]
    less=[x for x in nums if x<pivot]
    equal=[x for x in nums if x==pivot]
    greater=[x for x in nums if x>pivot]
    return quick_sort(less)+equal+quick_sort(greater)

def max_below(n,digits):
    """Largest nonnegative integer < n, unlimited digit reuse, no leading zeros.
    Zero is allowed only if 0 is in digits. None means no feasible answer.
    At most 10 allowed digits. Greedy equal-prefix walk with backtracking.
    """
    if not isinstance(n,int):raise TypeError('n must be an integer')
    if any(not isinstance(d,int) or not 0<=d<=9 for d in digits):raise ValueError('digits must be 0..9')
    allowed=sorted(set(digits),reverse=True)
    if n<=0 or not allowed:return None
    target=[int(x) for x in str(n)];L=len(target)
    def search(pos,prefix):
        if pos==L:return None           # identical n is not strictly smaller
        for d in allowed:
            if pos==0 and L>1 and d==0:continue
            if d>target[pos]:continue
            if d<target[pos]:
                return prefix+[d]+[allowed[0]]*(L-pos-1)
            result=search(pos+1,prefix+[d])
            if result is not None:return result
        return None
    out=search(0,[])
    if out is not None:return int(''.join(map(str,out)))
    for length in range(L-1,0,-1):
        first=next((d for d in allowed if d>0 or length==1),None)
        if first is not None:return int(str(first)+str(allowed[0])*(length-1))
    return None

if __name__=='__main__':
    print('A01',three_sum([-1,0,1,2,-1,-4]))
    print('A02',quick_sort([3,1,2,3,0]))
    print('A03',max_below(23121,[2,4,9]))
