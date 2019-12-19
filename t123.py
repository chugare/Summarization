

a = [(i, i*2, i*3) for i in range(10)]



print(a)

ad,b,c = zip(*a)
print(ad)
print(b)
print(c)
a = 'aaa'
print(a)