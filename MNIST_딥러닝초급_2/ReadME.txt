zip 파일을 unpacking 하면 그 해당 데이터가 사라지기 때문에 다음 epoch에서 학습을 할 수 없다.
b = [y for _, y in train_loader]
a = [x for x, _ in train_loader]

이런 형식으로 바꾸어 data를 풀어보려했지만 이것도 한번 풀리면 그다음 len(a) == 0이 되어 실패했다.