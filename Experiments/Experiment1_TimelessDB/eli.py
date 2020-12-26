
def solution(A):
    A = list(dict.fromkeys(A))
    A1 = [item for item in A if item > 0]
    A2 = [item for item in A if item < 0]

    if len(A1) == 0 or len(A2) == 0:
        return 0
    else:
        A1.sort(reverse=True)
        A2.sort()
        A2 = [x * -1 for x in A2]

        idx1 = 0
        idx2 = 0
        while len(A1) > idx1 and len(A2) > idx2:
            if A1[idx1] == A2[idx2]:
                return A1[idx1]

            elif A1[idx1] < A2[idx2]:
                idx2 += 1
            elif A1[idx1] > A2[idx2]:
                idx1 += 1

        return 0


A = [-1,1,2,3,4,6,-6]
print(A)
print(solution(A))


