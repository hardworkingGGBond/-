房价预测系统
项目简介
本项目基于真实房产数据，构建了一个 端到端的房价预测系统：
从 数据清洗、特征工程 到 模型训练、评估与部署，全流程自动化。
采用 Scikit-learn Pipeline + RandomForest 架构，提升数据处理与模型迭代效率。
使用 Streamlit 搭建可交互的网页界面，实现用户输入特征后实时预测房价。
功能亮点
数据处理：缺失值填补、正则提取、类别特征降维
机器学习：自动化预处理、模型调参、评估指标输出
可视化部署：一键启动的网页应用，实时预测与展示结果
环境依赖
建议使用 Python 3.9+。
主要依赖：
pandas, numpy
scikit-learn
streamlit
运行步骤
模型训练
python train_model.py
生成 house_price_model.pkl。
启动可视化网页
streamlit run app.py
打开浏览器访问提示的本地地址。
项目展示
运行后可在页面输入房屋面积、卧室数、位置等特征，系统会实时输出预测价格。
核心技术栈
Python / Pandas / NumPy：数据清洗与特征工程
Scikit-learn：Pipeline、RandomForestRegressor、模型评估
Streamlit：交互式 Web 前端部署
