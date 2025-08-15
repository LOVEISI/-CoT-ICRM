## 📄项目结构           
├── FeAT/ # FeAT 模块代码  
├── ICRM/ # ICRM 核心代码


## 🚀示例命令
在-CoT-ICRM/ICRM/ICRM下运行：  
```
python -m main --data_dir=/mnt/data02/gll_yong/ICRM/data/MNIST --algorithm ICRM --dataset ColouredMNIST
```
使用ColoredMNIST数据集

## ⚙️ 环境依赖

**本地开发环境版本**（建议服务器保持一致）：
- Python 3.10
- PyTorch 2.4.1
- scikit-learn 1.3.2
- 其它依赖见 `requirements.txt`

**安装依赖**
```bash
pip install -r requirements.txt
