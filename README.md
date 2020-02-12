# BERT_NER

## 이 repository는 "BERT 실용교육 한국인공지능아카데미"에서 진행하는 미니해커톤에서 사용한 코드를 정리한 repository입니다.
## 관련 코드는 https://github.com/dmis-lab/biobert 을 참고하여 수정, 작성하였습니다

### 주제

: 

    : BERT를 이용한 키워드 추출

### 목적
    1) BERT모델을 이용하여 BIO tagging data로 fine-tuning 
    	→ layer 하나를 추가하여 다양한 NLP task를 수행할 수 있는 bert모델의 특성을 이해하고,
    	직접 키워드 추출이 가능한 layer를 구현해봄
    2) 기존의 biobert는 tensorflow로 이루어져 있어 코드를 이해하는데 어려움 多 
        → 이를 keras로 수정하여 코드 이해도를 높임

### 주제 설명
    : 고려대학교 biobert weight를 이용하여 pre-train -> fine-tuning을 위해 마지막 layer를 추가하여 키워드 추출을 할 수 있도록 함.
    : biobert 사용 이유
      --> 사용하는 data는 의학 논문 데이터로 다양한 의학 언어를 포함하고 있음.  
          biobert weight는 PubMed, PMC로 pretrain된 weight이므로 본 프로젝트에서 사용하는 	   의학 논문 데이터에서 키워드 추출을 더 효과적으로 할 수 있음.
    :fine tuning에 사용한 data - BC5CDR-disease의 train, validation, test data
    							(BIO tagging 된 data)


#### 최종 발표까지의 계획
    (2/11 기준)
    2/6(목) 
         : keras bert에 들어갈 수 있는 data로 data 변환
         : input data, target data 형식으로 변환
    2/7(금)
         : output layer를 keras로 구현 -> 실패 : bert_model.fit 과정에서 error발생 
    2/10(월)
    	 : 실패 원인 분석 및 fine-tuning 완료
    2/11(화)
    	 : test data로 accuracy 확인


#### 진행상황 
    < Data 전처리: keras bert에 알맞은 input data로 변환>
    1. 본래 data를 읽어와 max_sequence_len(30) 길이의 문장들로 자르기 
    -> [문장, labels] 형식의 examples 생성
    2. 각각의 example들을 WordPieceTokenizer로 토큰화
    3. 토큰화된 단어들을 word vocab를 통해 인덱스화
    4. 
    
    < output layer를 keras로 구현 >
    1. 
    


​        


​          




