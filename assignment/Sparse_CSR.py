from Matrix import Matrix
from utils import Node

class Sparse_CSR(Matrix):
    def __init__(self, values) -> None:
        super().__init__()
        self.values=values

        self.format=''.join(values[0])  # 문자열로 변환
        self.shape=values[1]
        self.nnz=values[2]
        
        self.indices=values[3]
        self.indptr=values[4]
        self.nodes=[]       # 각 노드(성분)이 존재하는 위치를 저장
        self.mat=None       # dense 매트릭스
        self._make_entry_nodes()      # dense 행렬을 만들기 위한 헤더 생성
        
    def __str__(self) -> str:
        info=self.values[2:]
        nnz_info=''
        for x in info:
            x=list(map(str, x))
            x=' '.join(x) + '\n'
            nnz_info+=x
        return f'{self.format}\n{self.shape[0]} {self.shape[1]}\n{nnz_info}'

    def getDense(self):
        row, col=self.shape[0], self.shape[1]     # 행렬의 행과 열
        mat=[[0]*col for _ in range(row)]      # 0 행렬 생성
        for node in self.nodes:
            mat[node.row][node.column]=node.value
        mat.insert(0, [str(row), str(col)])     # shape 추가
        return mat

    def _make_entry_nodes(self):
        # 논제로(value)를 이용해 노드 객체를 생성
        # 노드 객체들을 리스트에 저장
        for d in self.nnz:
            node=Node()
            node.set_value(d)
            self.nodes.append(node)
        
        # column 정보를 노드 객체에 저장
        for idx, col in enumerate(self.indices):
            self.nodes[idx].set_column(col)

        cur=0       # nodes 리스트에 접근할 인덱스
        for row, num in enumerate(self.indptr[1:]):
            # 각 행에 존재하는 성분 수만큼을 nodes에서 가져옴
            # 가져온 노드들의 row를 설정
            for node in self.nodes[cur:num]:     
                node.set_row(row)
            cur=num     # 누적하면 인덱스 범위를 넘어감
