
def solution(blocks):
    distr = [0 for _ in range(len(blocks))]
    distl = [0 for _ in range(len(blocks))]
    mxdist = 0
    distl[0] = 1
    distr[-1] = 1
    for i in range(1, len(blocks)):
        if(blocks[i] <= blocks[i-1]):
            distl[i] = distl[i-1] + 1
        else:
            distl[i] = 1
    for i in range(len(blocks)-2, -1, -1):
        if(blocks[i] <= blocks[i+1]):
            distr[i] = distr[i+1] + 1
        else:
            distr[i] = 1
    for i in range(len(blocks)):
        mxdist = max(mxdist, distl[i]+distr[i]-1)
    return mxdist

print(solution([1,5,5,2,6]))
print(solution([1,5,5,2,6]))
print(solution([1,2,3,4,5]))
print(solution([5,4,3,2,1]))
print(solution([2,6,8,5]))
print(solution([1,1]))