from Matrix import Matrix

class Dense(Matrix):
    def __init__(self) -> None:
        super().__init__()
    
    def __str__(self) -> str:
        nnz_info=''                                         # matrix 출력용
        for x in self.nnz:
            x=list(map(str, x))
            x=' '.join(x) + '\n'
            nnz_info+=x
        return f'{self.format}\n{self.shape[0]} {self.shape[1]}\n{nnz_info}'
    
    def read_input(self, input):
        self.format=''.join(input[0])                      # 문자열로 변환
        self.shape=input[1]
        self.nnz=input[2:]
    
    def get_dense(self):
        row, col=int(self.shape[0]), int(self.shape[1])     # 행렬의 행과 열
        mat=[]
        for node in self.nnz:
            mat.append(list(map(int, node)))
        mat.insert(0, [str(row), str(col)])                 # shape 추가
        return mat
