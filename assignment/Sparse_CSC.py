from Matrix import Matrix
from utils import Node

class Sparse_CSC(Matrix):
    def __init__(self) -> None:
        super().__init__()
    
    def __str__(self) -> str:
        info=self.values[2:]
        nnz_info=''
        for x in info:
            x=list(map(str, x))
            x=' '.join(x) + '\n'
            nnz_info+=x
        return f'{self.format}\n{self.shape[0]} {self.shape[1]}\n{nnz_info}'
    
    def read_input(self, input):
        self.values=input

        self.format=''.join(self.values[0])  # 문자열로 변환
        self.shape=self.values[1]
        self.nnz=self.values[2]
        
        self.indices=self.values[3]
        self.indptr=self.values[4]
        self.nodes=[]                   # 각 노드(성분)이 존재하는 위치를 저장
        self.mat=None                   # dense 매트릭스
        self._make_entry_nodes()        # dense 행렬을 만들기 위한 헤더 생성

    def get_dense(self):
        row, col=self.shape[0], self.shape[1]       # 행렬의 행과 열
        mat=[[0]*col for _ in range(row)]           # 0 행렬 생성
        for node in self.nodes:
            mat[node.row][node.column]=node.value
        mat.insert(0, [str(row), str(col)])         # shape 추가
        return mat
    
    # self.nodes 초기화
    def _make_entry_nodes(self):                    # -> None
        # 논제로(value)를 이용해 노드 객체를 생성
        # 노드 객체들을 리스트에 저장
        for d in self.nnz:
            node=Node()
            node.set_value(d)
            self.nodes.append(node)
        
        # row 정보를 노드 객체에 저장
        for idx, row in enumerate(self.indices):
            self.nodes[idx].set_row(row)

        cur=0                                       # nodes 리스트에 접근할 인덱스
        for col, num in enumerate(self.indptr[1:]):
                                                    # 각 행에 존재하는 성분 수만큼을 nodes에서 가져옴
                                                    # 가져온 노드들의 row를 설정
            for node in self.nodes[cur:num]:     
                node.set_column(col)
            cur=num                                 # 누적하면 인덱스 범위를 넘어감
        
    def read_mat(self, dense_mat):
        nrow, ncol=dense_mat[0][0], dense_mat[0][1]
        nodes=[]
        nnz=[]
        indices=[]
        indptr=[0]*(ncol+1)                         # 0번째 컬럼 이전 값에 대한 정보도 갖고 있기 때문
        for row, line in enumerate(dense_mat[1:]):
            for col, element in enumerate(line):
                if element != 0:
                    node=Node()    
                    node.set_row(row)
                    node.set_column(col)
                    node.set_value(element)
                    nodes.append(node)
        nodes.sort(key=lambda x: (x.column, x.row)) # row 순서대로 정렬
        for node in nodes:
            indices.append(node.row)
            nnz.append(node.value)
            indptr[node.column+1]+=1
        
        # indptr 리스트 정제
        temp=[]
        x=indptr[0]
        for y in indptr[1:]:
            temp.append(x+y)
            x+=y
        indptr=temp
        indptr.insert(0, 0)

        # print(nnz)
        # print(indices)
        # print(indptr)
        # exit()
    
# dense_mat=[[4, 6],[0, 0, 3, 0, 4, 0],[0, 5, 0, 7, 0, 0],[0, 0, 0, 0, 0, 1],[2, 6, 0, 0, 0, 0,]]
# obj=Sparse_CSC(dense_mat)
# obj.read_mat(dense_mat)