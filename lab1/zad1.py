import math
def prime(n):
    isPrime = True
    for i in range(2,int(math.sqrt(n))+1):
        if n % i == 0:
            isPrime = False
    return isPrime
for i in range(100):
    print(i, prime(i))

