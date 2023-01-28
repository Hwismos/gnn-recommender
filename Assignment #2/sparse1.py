import utils
from utils import FileReader
from Sparse_CSR import Sparse_CSR
from Sparse_CSC import Sparse_CSC

class LIL():
    def __init__(self, values) -> None:
        self.values=values
        self.nrow=self.values[0][0]
        self.ncol=self.values[0][1]
        self.nnz=self.values[0][2]

        self.hoh=ElementsNode()     # hoh 노드 생성
        self.set_hoh()              # hoh 노드 세팅
        
    def set_hoh(self):
        self.hoh.set_row(self.nrow)         # row 수
        self.hoh.set_column(self.ncol)      # column 수
        self.hoh.set_value(self.nnz)        # nnz
        self.hoh.down=None
        self.hoh.right=HeadNode()           # first rights head
        self.hoh.right

        # right heads 초기화
        self.init_column_heads(self.hoh.right)  
        
        # 성분 노드와의 연결 시작  
        self.make_node_connection(self.hoh.right) 
    
    def init_column_heads(self, first_column_head):
        nmax=max(self.nrow, self.ncol)      # 더 큰 값으로 해드 개수 설정

        cur_node=first_column_head
        for c in range(nmax-1):
            cur_node.set_down(cur_node)
            cur_node.set_next(HeadNode())
            cur_node=cur_node.next
        
        # 마지막 헤드 노드 처리
        cur_node.set_down(cur_node)
        cur_node.set_next(self.hoh)
    
    def make_node_connection(self, first_column_head):
        # 모든 성분들에 대하여 (row, col) 기준으로 정렬된 리스트 저장
        elem_nodes=self.make_element_node()

        # 테스트 → 잘 나옴
        # for e in elem_nodes:
        #     print(f'[row({e.row}), col({e.column}), val({e.value})]\n[down({e.down}), right({e.right}))]\n\n')

        # 성분 노드들과 col_head(cur_head) 연결
        for elem in elem_nodes:
            # 모든 성분 노드들에 대해, 성분 노드의 컬럼 값까지 헤드 노드 이동
            cur_node=first_column_head  # 포커싱 중인 노드
            cur_head=cur_node   # 포커싱 중인 노드에 대한 헤드 노드
            for _ in range(elem.column):
                cur_node=cur_node.next

            # 첫 연결일 때
            if cur_node.down == None:
                cur_node.down=elem
            else:
                while cur_node.down != None:
                    cur_node=cur_node.down
                cur_node.down=elem
                cur_node=cur_node.down
                cur_node.down=cur_head      # 마지막 노드에서 다시 헤드 노드로 circular 하게 연결

        
    
    def make_element_node(self):
        data=self.values[1:]
        elem_nodes=[]
        for d in data:
            node=ElementsNode()
            node.set_row(d[0])
            node.set_column(d[1])
            node.set_value(d[2])
            elem_nodes.append(node)
        
        # 성분 노드들 정렬
        elem_nodes.sort(key=lambda x: (x.column, x.row))

        return elem_nodes

    # def getDense(self):
    #     mat=[[0]*self.ncol for _ in range(self.nrow)]   # 0 행렬 생성

    #     cur_head=self.hoh.right
    #     while cur_head.next != None:
    #         pass


def main():
    PATH=utils.parse_arg()
    f=FileReader(PATH)

    formats={'csr': Sparse_CSR, 'csc': Sparse_CSC}
    
    # 객체 생성
    obj=formats[f.format](f.values)
    print(obj)
    print('dense')
    for row in obj.getDense():
        row=list(map(str, row))
        row=' '.join(row)
        print(row)


if __name__=='__main__':
    main()
