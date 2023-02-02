# ===============================NODE====================================

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


class ElementsNode(Node):
    def __init__(self) -> None:
        super().__init__()
        self.down=None
        self.right=None
    
    def set_down(self, down):
        self.down=down

    def set_right(self, right):
        self.right=right

class HeadNode(ElementsNode):
    def __init__(self) -> None:
        super().__init__()
        self.next=None
    
    def set_next(self, next):
        self.next=next

# =======================================================================
# ===============================FILE====================================

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
        path=os.path.join(__file__[:-9], path)     
        
        with open(path, 'r') as f:
            for line, value in enumerate(f):
                value=value.rstrip('\n')        # 개행 제거
                
                # format 체크
                if line == 0:
                    if value in ('dense', 'lil', 'csr', 'csc'):
                        self.format=value
                        self.values.append(value)
                    else:
                        print('존재하지 않는 포맷 입니다.')
                        exit()

                # format을 지정하는 라인이 아닌 경우 int 타입의 리스트로 저장
                if line != 0:
                    value=list(map(int, value.split(' ')))
                    self.values.append(value)

# =======================================================================
# ===============================PARSE===================================

import argparse

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

# =======================================================================
