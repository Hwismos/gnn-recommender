from Matrix import Matrix

class Sparse_LIL(Matrix):
    def __init__(self, values) -> None:
        super().__init__()
        self.values=values
        
        self.format=self.values[0]
        self.shape=self.values[1]
        self.nnz=self.values[2:]
    
    def __str__(self) -> str:
        # nnz를 표현하는 부분
        nnz_info=''
        for x in self.values[2:]:
            x=list(map(str, x))
            x=' '.join(x) + '\n'
            nnz_info+=x
        return f'{self.format}\n{self.shape[0]} {self.shape[1]} {self.shape[2]}\n{nnz_info}'

    def getDense(self):
        row, col=int(self.shape[0]), int(self.shape[1])     # 행렬의 행과 열
        mat=[[0]*col for _ in range(row)]      # 0 행렬 생성
        for node in self.nnz:
            mat[int(node[0])][int(node[1])]=int(node[2])
        mat.insert(0, [str(row), str(col)])     # shape 추가
        return mat
        
