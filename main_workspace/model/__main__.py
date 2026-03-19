"""python -m model — 默认运行主 pipeline：排卵检测→月经预测。"""
from .experiment.run_final_ov_menses import main

if __name__ == "__main__":
    main()
