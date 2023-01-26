import csv

# PATH = '/home/hwiric/Internship/LightGCN_MovieLens/u1.base'
PATH = 'C:\\Users\\최석휘\\Desktop\\3학년 겨울방학 공학연구인턴십\\Internship\\LightGCN_MovieLens\\u1.base'

f = open(PATH, 'r')
rdr = csv.reader(f)

# 유저-아이템 딕셔너리 생성
d = {}
for l in rdr:
    s = ''.join(l).split('\t')

    # 레이팅이 3 이상인 아이템들로만 구성
    if int(s[2]) < 3:
        continue

    s = s[:2]
    if s[0] in d:
        d[s[0]].append(s[1])
    else:
        d[s[0]] = []

# print(d)

f.close()

# txt 파일 생성
with open(PATH+'.txt', 'w', encoding='UTF-8') as f:
    for user, item in d.items():
        user = str(int(user)-1)
        item = ' '.join(item)
        f.write(f'{user} {item}\n')
