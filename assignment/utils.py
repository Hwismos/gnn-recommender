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
        self.right=None         # None is object
    
    def set_down(self, down):
        self.down=down

    def set_right(self, right):
        self.right=right
    
    # 성분 노드 구조 확인
    def __str__(self):
        return '[{0}, {1}, {2}]\n[{3}, {4}]\n'.format(self.row, self.column, self.value, hex(id(self.down)), hex(id(self.right)))

class HeadNode(ElementsNode):
    def __init__(self) -> None:
        super().__init__()
        self.next=None      
    
    def set_next(self, next):
        self.next=next
    
    def __str__(self):
        return '[{0}]\n[{1}, {2}]\n'.format(hex(id(self.next)), hex(id(self.down)), hex(id(self.right)))

# =======================================================================
# ===============================FILE====================================

import os
from itertools import chain                     # flattening 2d list 

# 파일 입출력을 위한 인터페이스 정의
# 출력이 콘솔인지 모르고 FileHandler를 만들었음
    # 입력만이면 메소드로도 가능했음
class FileReader():
    def __init__(self, path) -> None:
        self.values=[]                          # 행렬 정보가 담길 리스트
        self.read(path)                         # 파일 읽기

    def read(self, path):                       # -> int list
        path=os.path.join(__file__[:-9], path)  # '\utils.py' 제거하고 입력 파일을 붙여서 읽을 파일의 경로를 설정     
        with open(path, 'r') as f:
            for line, value in enumerate(f):
                value=value.rstrip('\n')        # 개행 제거
                if line == 0:                   # format 체크
                    if value in ('dense', 'lil', 'csr', 'csc'):
                        self.values.append(value)
                    else:
                        print('존재하지 않는 포맷 입니다.')
                        exit()
                if line != 0:                   # format을 지정하는 라인이 아닌 경우 int 타입의 리스트로 저장
                    value=list(map(int, value.split(' ')))
                    self.values.append(value)

class FileWriter():
    def __init__(self, path, mat_d=None, mat_s=None, op=None, msg=None):
        self.op=op
        self.__set_path(path)

        if isinstance(msg, str):
            self.msg=msg
            self.__error_handling()

        self.mat_d=mat_d
        self.mat_s=mat_s
        self.__write()

    def __set_path(self, path):
            if self.op == 'add':
                self.op='a'
            if self.op == 'mul':
                self.op='m'

            output_file_path='output'+path[-7:-5]+self.op+'.txt'
            dir_path=__file__[:-9]+'\sparse2_outputs'
            os.makedirs(dir_path, exist_ok=True)
            self.path=os.path.join(dir_path, output_file_path)
    
    def __error_handling(self):
            with open(self.path, 'w') as f:
                f.write(self.msg+'\n')
            exit()

    def __write(self):
        with open(self.path, 'w') as f:
            f.write(self.mat_s[0]+'\n')                     # 파일 포맷 출력

            if self.mat_s[0] == 'lil':                      # lil이면서 0 행렬인 경우
                if self.mat_s[1][0][2] != 0:     
                    for line in chain.from_iterable(self.mat_s[1:]):
                        line=list(map(str, line))
                        line=' '.join(line)+'\n'
                        f.write(line)
                else:
                    line=' '.join(list(map(str, self.mat_s[1][0])))+'\n'
                    f.write(line)
            elif self.mat_s[0] == 'dense':                  # dense이면서 0 행렬인 경우
                for line in chain.from_iterable(self.mat_d[1:]):
                    line=list(map(str, line))
                    line=' '.join(line)+'\n'
                    f.write(line)
            else:
                if len(self.mat_s[1][1]) != 0:              # csc, csr이면서 0 행렬인 경우
                    for line in chain.from_iterable(self.mat_s[1:]):
                        line=list(map(str, line))
                        line=' '.join(line)+'\n'
                        f.write(line)
                else:
                    line=' '.join(list(map(str, self.mat_s[1][0])))+'\n'
                    f.write(line)
            
            f.write('\n'+self.mat_d[0]+'\n')
            for line in chain.from_iterable(self.mat_d[1:]):
                line=list(map(str, line))
                line=' '.join(line)+'\n'
                f.write(line)

# =======================================================================
# ===============================PARSE===================================

import argparse

# 입력 파일 처리
# txt 파일이 아니면 에러 메시지 출력 후 프로그램 종료
def parse_arg():
    parser=argparse.ArgumentParser()
    parser.add_argument('arg1')
    '''
    # ! nargs
    # Number of times the argument can be used
    '''
    parser.add_argument('arg2', nargs='?', default=None)    
    parser.add_argument('opt', nargs='?',default=None)
    args=parser.parse_args()

    if args.arg2 != None:
        return args
    else:
        return args.arg1

# =======================================================================
