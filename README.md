## 📄项目结构和关键代码说明           
├── FeAT/ # FeAT 模块代码  
├── ICRM/ # ICRM 核心代码
      ICRM/
         main.py  #主要启动脚本，包含训练、验证、测试全流程
         networks.py #模型结构代码，关键是class GPT2Transformer下的forward函数的实现
         algorithms.py # 主要算法代码  其中关键的是 1. class ERM(Algorithm) 下的update 函数，是训练的时候调用的。 2. class ICRM(ERM) 下的evaluate 和 _evaluate_robust 函数，是验证和测试的时候用的，以及 predict函数：是在训练、验证和测试的时候都用同一个predict函数
         dataset.py  # ColoredMNIST的数据集在该方法的最后


## 🚀示例命令
在-CoT-ICRM/ICRM/ICRM下运行：  
```
python -m main --data_dir=/mnt/data02/gll_yong/ICRM/data/MNIST --algorithm ICRM --dataset ColouredMNIST
```
使用ColoredMNIST数据集，注意--data_dir 要切换成真实的MNIST数据集的路径

## ⚙️ 环境依赖

**本地开发环境版本**（建议服务器保持一致）：
- Python 3.10
- 其它依赖见 `requirements.txt`

**安装依赖**
```bash
pip install -r requirements.txt
