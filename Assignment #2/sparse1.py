import argparse
import os


# 파일 입출력을 위한 인터페이스 정의
# 출력이 콘솔인지 모르고 FileHandler를 만들었음
    # 입력만이면 메소드로도 가능했음
class FileReader():
    def __init__(self, path) -> None:
        self.values=[]      # 행렬 정보가 담길 리스트
        self.read(path)     # 파일 읽기

    def read(self, path):
        # '\sparse1.py' 제거하고 입력 파일을 붙여서 읽을 파일의 경로를 설정
        path=os.path.join(__file__[:-11], path)     
        
        with open(path, 'r') as f:
            for line, value in enumerate(f):
                value=value.rstrip('\n')        # 개행 제거
                
                # format 체크
                if line == 0:
                    if value in ('dense', 'lil', 'csr', 'csc'):
                        self.format=value
                    else:
                        print('존재하지 않는 포맷 입니다.')
                        exit()

                # format을 지정하는 라인이 아닌 경우 int 타입의 리스트로 저장
                if line != 0:
                    value=list(map(int, value.split(' ')))
                    self.values.append(value)


class Node():
    def __init__(self) -> None:
        self.value=None
        self.row=None
        self.column=None
    
    def set_value(self, value):
        self.value=value

    def set_row(self, row):
        self.row=row

    def set_column(self, column):
        self.column=column

class LIL():
    pass


class CSR():
    def __init__(self, values) -> None:
        self.shape=values[0]
        self.data=values[1]
        self.indices=values[2]
        self.indptr=values[3]
        self.nodes=[]       # 각 노드(성분)이 존재하는 위치를 저장
        self.mat=None       # dense 매트릭스
        self.make_matrix_info()      # dense 행렬을 만들기 위한 헤더 생성
        
    def make_matrix_info(self):
        # 논제로(value)를 이용해 노드 객체를 생성
        # 노드 객체들을 리스트에 저장
        for d in self.data:
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
        
        self.mat=self.make_matrix()

    def make_matrix(self):
        row, col=self.shape[0], self.shape[1]     # 행렬의 행과 열

        mat=[[0]*col for _ in range(row)]      # 0 행렬 생성
        for node in self.nodes:
            mat[node.row][node.column]=node.value
        
        return mat



class CSC():
    def __init__(self, values) -> None:
        self.shape=values[0]
        self.data=values[1]
        self.indices=values[2]
        self.indptr=values[3]
        self.nodes=[]       # 각 노드(성분)이 존재하는 위치를 저장
        self.mat=None       # dense 매트릭스
        self.make_matrix_info()      # dense 행렬을 만들기 위한 헤더 생성
        
    def make_matrix_info(self):
        # 논제로(value)를 이용해 노드 객체를 생성
        # 노드 객체들을 리스트에 저장
        for d in self.data:
            node=Node()
            node.set_value(d)
            self.nodes.append(node)
        
        # column 정보를 노드 객체에 저장
        for idx, row in enumerate(self.indices):
            self.nodes[idx].set_row(row)

        cur=0       # nodes 리스트에 접근할 인덱스
        for col, num in enumerate(self.indptr[1:]):
            # 각 행에 존재하는 성분 수만큼을 nodes에서 가져옴
            # 가져온 노드들의 row를 설정
            for node in self.nodes[cur:num]:     
                node.set_column(col)
            cur=num     # 누적하면 인덱스 범위를 넘어감
        
        self.mat=self.make_matrix()

    def make_matrix(self):
        row, col=self.shape[0], self.shape[1]     # 행렬의 행과 열

        mat=[[0]*col for _ in range(row)]      # 0 행렬 생성
        for node in self.nodes:
            mat[node.row][node.column]=node.value
        
        return mat


# 콘솔에 결과값 출력
# 아직 dense 행렬은 안 만들었음
def print_console(format, values, dense_info, dense_matrix=None):
    print(format)
    
    for value in values:
        for v in value:
            print(v, end=' ')
        print()
    print()

    print('dense')
    row, col=dense_info[0], dense_info[1]
    print(f'{row} {col}')
    for row in dense_matrix:
        for elem in row:
            print(elem, end=' ')
        print()


# 입력 파일 처리
# txt 파일이 아니면 에러 메시지 출력 후 프로그램 종료
def parse_arg():
    parser=argparse.ArgumentParser()
    parser.add_argument('file')
    args=parser.parse_args()

    if args.file[-3:] != 'txt' :
        print('txt 파일을 입력해주세요.')
        exit()

    return args.file


def main():
    # 파일 경로 저장
    PATH=parse_arg()

    # 파일 읽기
    f=FileReader(PATH)

    # dense 포맷이면 바로 출력하고 종료
    if f.format == 'dense':
        pass
    
    # 포맷 딕셔너리를 이용해서 객체 생성 
    formats={'lil': LIL, 'csr': CSR, 'csc': CSC}
    dense_info=formats[f.format](f.values).shape      # 행과 열 값 반환 → 콘솔 출력용  
    dense_result=formats[f.format](f.values).mat      # dense 매트릭스로 전환

    # 콘솔에 출력 결과 확인
    print_console(f.format, f.values, dense_info, dense_result)


if __name__=='__main__':
    main()
