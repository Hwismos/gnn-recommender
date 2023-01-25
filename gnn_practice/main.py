
# 모듈을 불러오는 것
import networkx as nx
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# node2vec 모듈로부터 Node2Vec 클래스를 불러오는 것
from node2vec import Node2Vec

'''
- 모듈은 제3자가 만든 파이썬 파일
- 모듈을 모아둔 것이 패키지
- 모듈 안에는 메소드, 변수, 클래스 등이 존재함
'''

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

# print(plt.__version__)

# 가라테 클럽 그래프 생성
G = nx.karate_club_graph()

# 노드들의 라벨을 int에서 string으로 변환
G = nx.relabel_nodes(G, {n: str(n) for n in G.nodes()})

# node2vec 객체 생성
node2vec = Node2Vec(graph=G,
                    dimensions=50, # 임베딩 차원
                    walk_length=10, # 이웃노드 수
                    p=1,
                    q=-0.0001,
                    weight_key="None",
                    num_walks=2000,
                    workers=1,
                    )

for i, each_walk in enumerate(node2vec.walks):
    print(f"{i:0>2d}, {each_walk}")
    if i > 1:
        break
'''
each_walk = ['19', '33', '28', '2', '9', '33', '8', '2', '13', '33']
'''

# walk를 사용해서 학습
# gensim은 통계 관련 라이브러리
model1 = node2vec.fit()

K = 5
kmeans = KMeans(n_clusters=K, random_state=0).fit(model1.wv.vectors)

for n, label in zip(model1.wv.index_to_key, kmeans.labels_):
    G.nodes[n]['label'] = label

# plt.figure(figsize=(12, 6))
nx.draw_networkx(G,
                 pos=nx.layout.spring_layout(G),
                 node_color=[n[1]['label'] for n in G.nodes(data=True)],
                 cmap=plt.cm.rainbow)
# plt.axis('off')
plt.show()
