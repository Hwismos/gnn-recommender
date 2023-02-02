class Node():
    def __init__(self) -> None:
        self.first=None
        self.second=None
        self.third=None
    
    def __str__(self) -> str:
        return f'first: {self.first}, second: {self.second}, third: {self.third}'

li=[]

n1=Node()
n1.first=5
n1.second=3
n1.third=1
li.append(n1)

n2=Node()
n2.first=5
n2.second=2
n2.third=1
li.append(n2)

n3=Node()
n3.first=1
n3.second=3
n3.third=1
li.append(n3)

print('정렬 전')
for l in li:
    print(l)

print('\n정렬 후')
li.sort(key=lambda x: (x.first, x.second))
for l in li:
    print(l)


