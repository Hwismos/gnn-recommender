from Matrix import Matrix

class Dense(Matrix):
    def __init__(self, values) -> None:
        super().__init__()
        self.format=''.join(values[0])  # 문자열로 변환
        self.shape=values[1]
        self.nnz=values[2:]
    
    def __str__(self) -> str:
        mat=''  # matrix 출력용
        for x in self.nnz:
            x=list(map(str, x))
            x=' '.join(x) + '\n'
            mat+=x
        
        return f'{self.format}\n{self.shape[0]} {self.shape[1]}\n{mat}'
    
    def getDense(self):
        return self.nnz