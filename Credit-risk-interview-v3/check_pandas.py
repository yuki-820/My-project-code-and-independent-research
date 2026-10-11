import sys
import pandas as pd
import pandas_lab as answer
import pandas_exercises as work
qid=sys.argv[1].upper() if len(sys.argv)>1 else 'P01'
if qid not in [f'P{i:02}' for i in range(1,9)]:raise SystemExit('Use P01..P08')
try:
    got=getattr(work,qid)(answer.load());expected=getattr(answer,qid)(answer.load())
    pd.testing.assert_frame_equal(got,expected,check_dtype=False,check_categorical=False,rtol=1e-7,atol=1e-6)
    print('PASS',qid,'现在关掉答案，解释每一步的粒度和边界。')
except Exception as e:
    print('RETRY',qid,type(e).__name__,str(e));sys.exit(1)
