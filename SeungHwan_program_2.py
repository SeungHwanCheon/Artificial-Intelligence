import pandas as pd

lottery = pd.read_csv('lottery.csv')
lottery2 = pd.read_csv('lottery.csv')

lottery["win"] = 1
lottery2["win"] = 0

lottery2["first"] = 1
lottery2["second"] = 2
lottery2["third"] = 3
lottery2["fourth"] = 4
lottery2["fifth"] = 5
lottery2["sixth"] = 6
lottery2["bonus"] = 7

for i in range (959) :

    cnt = 0

    for j in ['first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'bonus'] :

        if lottery [j][i] == lottery2 [j][i] :

            cnt = cnt + 1

        #Because 7th number is bonus.
        if cnt == 6 :

            lottery2[win][i] = 1

lottery3 = pd.concat([lottery, lottery2])

#inplace = True
lottery3.sort_values(by = ["round", "win"], ascending = [False, False], inplace = True)

#drop은 인덱스로 세팅한 열을 df내에서 삭제할지 여부 결정
#inplace는 원본 객체를 변경할지 여부 결정 (원본 데이터 변경 or 복사본 반환)
lottery3.reset_index(drop = True, inplace = True)

print(lottery3)