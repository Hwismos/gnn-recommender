import collections
import csv

PATH = '/home/hwiric/Internship/LastFM/lastfm/test1.txt'

f = open(PATH, 'r')
rdr = csv.reader(f)

# 유저-아이템 딕셔너리 생성
d = {}
users=[]
for l in rdr:
    s = ''.join(l).split()
    # print(s)
    if s[0] in d:
        d[s[0]].append(s[1])
    else:
        d[s[0]] = [s[1]]
    users.append(d[s[0]])
f.close()

# print(len(users))

sorted_dict=sorted(d.items(), key=lambda x: int(x[0]))

# print(sorted_dict)

# cnt=0
# txt 파일 생성
with open('/home/hwiric/Internship/LastFM/lastfm/test.txt', 'w', encoding='UTF-8') as f:
    for line in sorted_dict:
        user, item=line[0], line[1]

        user = str(int(user)-1)
        item = ' '.join(item)
        f.write(f'{user} {item}\n')
# print(cnt)