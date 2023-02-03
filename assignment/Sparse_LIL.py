from utils import ElementsNode, HeadNode

class Sparse_LIL():
    def __init__(self, values) -> None:
        self.values=values

        self.format=self.values[0]
        self.nrow=self.values[1][0]
        self.ncol=self.values[1][1]
        self.num_nnz=self.values[1][2]
        self.nnz=self.values[2:]

        self.hoh=ElementsNode()     # hoh 노드 생성
        # print(f'head of heads node 객체: {self.hoh}\n')
        self.set_hoh()              # hoh 노드 세팅
        
    # head of heads
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
        nmax=max(int(self.nrow), int(self.ncol))      # 더 큰 값으로 해드 개수 설정

        cur_node=first_column_head
        for _ in range(nmax-1):
            cur_node.set_down(cur_node)     #  자기 자신을 가리키도록 세팅
            cur_node.set_next(HeadNode())
            # print(cur_node)
            cur_node=cur_node.next
        
        # 마지막 헤드 노드 처리
        cur_node.set_down(cur_node)
        cur_node.set_next(self.hoh)
        # print(cur_node)
        # print()
        # print(hex(id(self.hoh)))
    
    def make_node_connection(self, first_column_head):
        # 모든 성분들에 대하여 (row, col) 기준으로 정렬된 리스트 저장
        elem_nodes=self.make_element_node()
        
        '''
        [3, 0, 2]
        [None, None]

        [1, 1, 5]
        [None, None]

        [3, 1, 6]
        [None, None]

        [0, 2, 3]
        [None, None]

        [1, 3, 7]
        [None, None]

        [0, 4, 4]
        [None, None]

        [2, 5, 1]
        [None, None]
        '''
        # for node in elem_nodes:
        #     print(node)
        # exit()

        # 성분 노드들과 col_head(cur_head) 연결
        for elem in elem_nodes:
            cur_node=None       # 출력용
            
            # 모든 성분 노드들에 대해, 성분 노드의 컬럼 값까지 헤드 노드 이동
            cur_head=first_column_head  # 포커싱 중인 헤드 노드
            
            # print(f'{int(elem.column)}번째 컬럼')       # 포커싱 컬럼 확인
            
            for i in range(int(elem.column)):
                cur_head=cur_head.next

            # 첫 연결일 때
            # 자기 자신일 때
            if cur_head.down == cur_head:
                cur_head.down=elem
                elem.down=cur_head
                
                cur_node=elem
            # ! 밑으로 연결이 안됨
            # else:
            #     cur_node=cur_head.down
            #     while cur_node.down == cur_head:   # 자기 자신이 되기 전까지
            #         print('밑으로!!!!!!')
            #         cur_node=cur_node.down
            #     cur_node.down=elem
            #     cur_node=cur_node.down
            #     cur_node.down=cur_head      # 마지막 노드에서 다시 헤드 노드로 circular 하게 연결
            
            print(f'헤드 객체 위치: {hex(id(cur_head))}\n헤드 노드: {cur_head}\n포커싱 객체 위치: {hex(id(cur_node))}\n포커싱 노드\n{cur_node}============================================')
        
        # # ! 연결 테스트
        # cur_head=first_column_head
        # cur_node=cur_head.down      # 성분 노드
        # while cur_head.next != self.hoh:
        #     while cur_node.down != cur_head:
        #         cur_node=cur_node.down
        #         print(cur_node)
        #     cur_head=cur_head.next
        # # while cur_node.down != cur_head:
        # #     cur_node=cur_node.down
        # #     print(cur_node)
        exit()          # !
        
    
    # 성분 노드 세팅
    def make_element_node(self):
        data=self.nnz
        elem_nodes=[]
        for d in data:
            node=ElementsNode()
            node.set_row(d[0])
            node.set_column(d[1])
            node.set_value(d[2])
            elem_nodes.append(node)

        # 성분 노드들 정렬
        elem_nodes.sort(key=lambda x: (x.column, x.row))

        # for node in elem_nodes:
        #     print(node)

        return elem_nodes
